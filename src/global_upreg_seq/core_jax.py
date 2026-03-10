import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import numpyro.optim
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro.infer.initialization import init_to_value
import optax

from .model_jax import jax_glob_seq_model, jax_glob_seq_model_nosf


def jax_generate_simulated_data(
    group_size,
    gene_size=20000,
    median_log_upreg=0.2,
    non_de_fraction=0.25,
    a_0=0.01,
    a_1=5,
    log_base_mean_val=2.0,
    seed=0,
):
    """Generate simulated RNA-seq count data with differential expression.

    Parameters match generate_simulated_data from core.py exactly.

    Returns
    -------
    counts_observed : jnp.ndarray [2*group_size, gene_size]
    labels          : jnp.ndarray [2*group_size]
    (log_fc, size_factors, log_base_means) : ground truth tuple
    """
    key = jax.random.PRNGKey(seed)
    key, k1, k2, k3, k4, k5 = jax.random.split(key, 6)

    log_base_means = jnp.abs(jax.random.normal(k1, (gene_size,)) * log_base_mean_val) - 2

    raw_fc = jax.random.normal(k2, (gene_size,)) + median_log_upreg
    mask   = jax.random.uniform(k3, (gene_size,)) >= non_de_fraction 
    log_fc = jnp.where(mask, raw_fc, 0.0)

    base_mean  = jnp.exp(log_base_means)
    upreg_mean = jnp.exp(log_base_means + log_fc)

    gene_effect    = 2 * jax.random.uniform(k4, (gene_size,))
    overdispersion = ((a_1 / base_mean) + a_0) * gene_effect

    base_samples  = dist.NegativeBinomial2(base_mean,  overdispersion).sample(k5, (group_size,))
    key, k6 = jax.random.split(key)
    upreg_samples = dist.NegativeBinomial2(upreg_mean, overdispersion).sample(k6, (group_size,))

    combined = jnp.concatenate([base_samples, upreg_samples], axis=0).astype(jnp.float32)
    labels   = jnp.concatenate([jnp.zeros(group_size, dtype=jnp.int32),
                                 jnp.ones(group_size,  dtype=jnp.int32)])

    key, k7 = jax.random.split(key)
    size_factors    = dist.LogNormal(0., 1.).sample(k7, (group_size * 2,))
    counts_observed = jnp.round(combined * size_factors[:, None])

    return counts_observed, labels, (log_fc, size_factors, log_base_means)


def jax_prepare_initialization(counts_observed, labels):
    """Data-driven init matching prepare_initialization from core.py.

    Returns (log_fc_init, log_mu0) as jnp arrays.
    """
    base_mask  = ~labels.astype(bool)
    upreg_mask = labels.astype(bool)

    log_mu0     = jnp.mean(jnp.log(counts_observed[base_mask]  + 0.1), axis=0)
    log_fc_init = jnp.mean(jnp.log(counts_observed[upreg_mask] + 0.1), axis=0) - log_mu0

    return log_fc_init, log_mu0


def jax_prepare_norm_mode(counts, labels):
    """Compute log_fc, log_mu0, and size factor estimates via mode of log count ratios.

    Parameters
    ----------
    counts : jnp.ndarray [N, P]
    labels : jnp.ndarray [N]

    Returns
    -------
    log_fc  : jnp.ndarray [P]
    log_mu0 : jnp.ndarray [P]
    sf_est  : jnp.ndarray [N]
    """
    log_counts   = jnp.where(counts > 0, jnp.log(counts), jnp.nan)
    counts_gm    = jnp.nanmean(log_counts, axis=0)               # [P]
    counts_ratio = log_counts - counts_gm                         # [N, P]

    counts_ratio_clean = jnp.nan_to_num(counts_ratio, nan=0.0)

    # use finite edges — clip values outside range before histogramming
    lo, hi = -7.5, 7.5
    counts_ratio_clipped = jnp.clip(counts_ratio_clean, lo, hi)
    bins = jnp.linspace(lo, hi, 152)                              # 152 edges -> 151 bins

    N = counts_ratio_clipped.shape[0]
    hist_counts = jnp.stack([
        jnp.histogram(counts_ratio_clipped[i], bins=bins)[0]
        for i in range(N)
    ])                                                             # [N, 151]

    # zero out center bins to fix sparse-count artifact
    center      = hist_counts.shape[1] // 2
    hist_counts = hist_counts.at[:, center - 1:center + 1].set(0)

    peaks = jnp.argmax(hist_counts, axis=1)                       # [N]

    # remove samples whose peak landed in the first or last bin
    # (these are samples where the mode estimation failed)
    valid_mask = (peaks > 0) & (peaks < 150)
    assert jnp.all(valid_mask), \
        f"peak in edge bin for {jnp.sum(~valid_mask)} samples — check input counts"

    bin_centers = (bins[:-1] + bins[1:]) / 2                     # [151]
    sf_est = bin_centers[peaks]                                    # [N]

    true_counts = jnp.log(counts + 0.1) - sf_est[:, None]        # [N, P]

    base_mask  = ~labels.astype(bool)
    upreg_mask = labels.astype(bool)

    log_mu0 = jnp.mean(true_counts[base_mask],  axis=0)
    log_fc  = jnp.mean(true_counts[upreg_mask], axis=0) - log_mu0
    variance = jnp.var(jnp.exp(true_counts[base_mask]), axis=0)
    #alpha_est = (variance-jnp.exp(log_mu0))/jnp.square(jnp.exp(log_mu0))
    alpha_est = jnp.clip((variance - jnp.exp(log_mu0)) / jnp.square(jnp.exp(log_mu0)), 0.0001, 15.0)

    return log_fc, log_mu0, sf_est, alpha_est


def jax_run_pyro(counts, labels, key, cutoff=np.log(2.0), iterations=3000, device="cpu", use_size_factor_model=True, use_mode_norm=True):
    """Train NumPyro model. Drop-in replacement for run_pyro from core.py.

    Parameters
    ----------
    counts     : array-like [N, P]
    labels     : array-like [N]
    key        : jax.random.PRNGKey — pass different key per replicate for true independence
    cutoff     : float, significance cutoff in natural log scale
    iterations : int
    device     : str, accepted for API compatibility — set platform before
                 JAX import via numpyro.set_platform() at top of script

    Returns
    -------
    results_dict : {'log2fc': ndarray [P], 'significant': ndarray [P] bool}
    losses       : list of float
    svi_result   : numpyro SVIRunResult (replaces guide in pyro version)
    """
    y = jnp.array(counts, dtype=jnp.float32)   # [N, P] counts
    x = jnp.array(labels, dtype=jnp.float32)   # [N]    labels

    if x.ndim == 1:
        x = x[:, None]                          # [N, F=1] design matrix

    N, P = y.shape
    F    = x.shape[1]

    assert N > 0 and P > 0 and F > 0
    
    #initialize model object
    model = None
    #initialize the values
    log_fc_init = None
    log_mu0 = None
    alpha_est = jnp.ones(P)

    s = jnp.ones(N)
    
    #make the mode norm be set
    if use_mode_norm:
        log_fc_init, log_mu0, s, alpha_est = jax_prepare_norm_mode(y, jnp.array(labels))
    else:
        log_fc_init, log_mu0 = jax_prepare_initialization(y, jnp.array(labels))
    
    #make the model be set
    if not use_size_factor_model:
        model = jax_glob_seq_model_nosf
    elif use_size_factor_model:
        #set things to zero to make sf stable and force it to "jump" out
        #this will allow us to be conservative in finding true values
        log_fc_init = jnp.zeros_like(log_fc_init)
        model = jax_glob_seq_model

    init_vals = {
        "log_mu0": log_mu0,
        "alpha":   alpha_est,
        "log_fc":  log_fc_init[None, :],        # [F, P]
        #after this we get sample specific, will ignore if not found in model
        "s":       s,
        "sample_alpha": jnp.square(s)[:, None],
        "pi":      jnp.full((F,), 0.5),
    }

    guide     = AutoNormal(model,
                           init_loc_fn=init_to_value(values=init_vals),
                           init_scale=0.1)
    #optimizer = numpyro.optim.ClippedAdam(step_size=0.01, clip_norm=1.5,
    #                                      lrd=0.0005 ** (1.0 / iterations))

    optimizer = numpyro.optim.optax_to_numpyro(
        optax.chain(
            optax.clip_by_global_norm(1.5),
            optax.adam(
                learning_rate=optax.exponential_decay(
                    init_value=0.01,
                    transition_steps=iterations,
                    decay_rate=0.0005
                )
            )
        )
    )
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    result = svi.run(key, iterations, x, y, N, P, F, progress_bar=True)
    losses = result.losses.tolist()

    params = result.params

    log_fc_loc   = params["log_fc_auto_loc"]    # [F, P]
    log_fc_scale = params["log_fc_auto_scale"]  # [F, P]

    cutoff_val = jnp.array(cutoff)
    posterior  = dist.Normal(jnp.abs(log_fc_loc), log_fc_scale)
    p_lesser   = posterior.cdf(cutoff_val)

    log_fc_np = np.array(log_fc_loc).squeeze()  # [P]

    return {
        "log2fc":      log_fc_np / np.log(2.0),
        "plesser": np.array(p_lesser).squeeze(),
    }, losses, result
