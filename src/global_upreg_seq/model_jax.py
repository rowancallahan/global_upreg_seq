import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist


def jax_glob_seq_model(x, y, N, P, F):
    """
    NumPyro translation of glob_seq_model (FactorModelNBspike).

    x : [N, F]  design matrix (condition labels)
    y : [N, P]  observed counts
    N, P, F     static ints — pass explicitly to avoid Python ops in XLA kernel

    - log_mu0, alpha : gene-level priors [P]
    - pi             : per-factor slab probability [F]
    - z              : per-gene soft gate [F, P] via RelaxedBernoulli(pi)
    - log_fc         : MixtureSameFamily(z -> slab 5.0, 1-z -> spike 0.05) [F, P]
    - s              : sample-level size factors [N]
    - obs            : NegativeBinomial2(mean=exp(log_mu), concentration=alpha)
    """
    assert x.shape == (N, F), f"x must be [N={N}, F={F}], got {x.shape}"
    assert y.shape == (N, P), f"y must be [N={N}, P={P}], got {y.shape}"

    # gene-level priors: [P]
    with numpyro.plate("gene", P):
        log_mu0 = numpyro.sample("log_mu0", dist.Normal(0., 10.))
        alpha   = numpyro.sample("alpha",   dist.LogNormal(-1., 2.0))

    # factor-level pi [F], gene-level z and log_fc [F, P]
    with numpyro.plate("factor", F, dim=-2):
        pi = numpyro.sample("pi", dist.Beta(1.5, 1.5))  # [F]

        with numpyro.plate("gene", P, dim=-1):
            z = numpyro.sample("z",
                dist.RelaxedBernoulli(temperature=0.3, probs=pi[..., None]))  # [F, P]
            log_fc = numpyro.sample("log_fc",
                dist.MixtureSameFamily(
                    dist.Categorical(probs=jnp.stack([z, 1 - z], axis=-1)),
                    dist.Normal(0., jnp.array([5.0, 0.05]))
                ))  # z -> slab (5.0), 1-z -> spike (0.05)

    # sample-level size factors: [N]
    with numpyro.plate("sample", N):
        s = numpyro.sample("s", dist.Normal(0., 5.))

    # [N, F] @ [F, P] -> [N, P]
    log_mu = log_mu0 + x @ log_fc + s[:, None]
    log_mu = jnp.clip(log_mu, -7., 25.)

    assert log_mu.shape == (N, P), f"log_mu shape mismatch: {log_mu.shape}"

    with numpyro.plate("sample", N, dim=-2):
        with numpyro.plate("gene", P, dim=-1):
            numpyro.sample("obs",
                dist.NegativeBinomial2(jnp.exp(log_mu), alpha),
                obs=y)

def jax_glob_seq_model_nosf(x, y, N, P, F):

    with numpyro.plate("gene", P, dim=-1):
        log_mu0    = numpyro.sample("log_mu0",    dist.Normal(0., 10.))
        alpha_gene = numpyro.sample("alpha_gene", dist.LogNormal(-1., 2.0))

    midpoint   = numpyro.sample("uninformative_cutoff", dist.Normal(0., 1.))
    gate_scale = numpyro.sample("uninformative_scale",  dist.LogNormal(0., 1.))
    size_gate  = jax.nn.sigmoid((log_mu0 - midpoint) / gate_scale)  # [P]
    eps = 5e-2

    with numpyro.plate("factor_plate", F, dim=-2):
        tau_g = numpyro.sample("tau_g", dist.HalfNormal(3.0))  # [F]
        with numpyro.plate("gene", P, dim=-1):
            tau_l  = numpyro.sample("tau_l",  dist.HalfNormal(3.0))  # [F, P]
            log_fc = numpyro.sample("log_fc",
                dist.Normal(0., eps + size_gate * tau_g[..., None] * tau_l))  # [F, P]

    with numpyro.plate("data", N, dim=-2):
        sample_alpha = numpyro.sample("sample_alpha", dist.HalfNormal(1.))  # [N]

        with numpyro.plate("gene", P, dim=-1):
            log_mu = log_mu0 + x @ log_fc   # [N, F] @ [F, P] -> [N, P]
            log_mu = jnp.clip(log_mu, -7., 25.)
            alpha = alpha_gene * (1.0 + sample_alpha)  # [N, P]

            numpyro.sample("obs",
                dist.NegativeBinomial2(jnp.exp(log_mu), alpha), obs=y)
