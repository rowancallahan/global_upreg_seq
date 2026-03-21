# GlobSeq

Bayesian differential expression analysis for RNA-seq under global upregulation. Uses a spike-and-slab prior to jointly estimate fold changes and size factors, avoiding normalization bias when most genes are DE.

## Installation

```bash
git clone https://github.com/rowancallahan/global_upreg_seq.git
cd global_upreg_seq
pip install -e .
pip install "jax[cpu]==0.9.1" numpyro==0.20.0
```

Requires Python ≥ 3.10. For GPU, use `jax[cuda12]` instead of `jax[cpu]`.

## Quick Start

```python
import numpy as np
import pandas as pd
import jax
from global_upreg_seq import jax_run_pyro

counts = pd.read_csv("counts.csv", index_col=0)  # genes × samples
labels = np.array([0, 0, 0, 1, 1, 1])             # 0=control, 1=treatment

key = jax.random.PRNGKey(0)
results, losses, svi = jax_run_pyro(
    counts.values.T,  # [N, P] — samples × genes
    labels, key, iterations=3000, use_size_factor_model=True,
)

de_results = pd.DataFrame({
    "gene": counts.index,
    "log2fc": results["log2fc"],
    "plesser": results["plesser"],
})
de_results["significant"] = (de_results["plesser"] < 0.05) & (de_results["log2fc"].abs() > 1)
```

## Interpreting Results

- **`log2fc`** — posterior mean log₂ fold change per gene
- **`plesser`** — P(|log FC| < ln 2), the probability the effect is small. Lower = more significant. Call DE at `plesser < 0.05`

### Finding Stably Expressed Genes

Unlike frequentist methods that can only fail to reject the null, `plesser` directly quantifies the probability a gene's fold change is small — giving positive evidence for stability.

```python
de_results["stable"] = de_results["plesser"] > 0.95  # >95% probability of no change
```

## Design Matrices

GlobSeq accepts arbitrary design matrices `[N, F]`. A 1D label array is reshaped to `[N, 1]` automatically. For categorical variables, use one-hot encoding with K−1 columns (drop one category as the reference). Libraries like `formulaic` or `patsy` handle this automatically with formula syntax.

```python
from formulaic import model_matrix  # pip install formulaic

metadata = pd.DataFrame({
    "condition": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
    "batch":     ["x", "x", "y", "x", "y", "y", "x", "y", "y"],
})

# Drop intercept — GlobSeq has its own (log_mu0)
X = np.array(model_matrix("~ condition + batch", metadata))[:, 1:]

results, losses, svi = jax_run_pyro(counts.values.T, X, key, iterations=3000)
# results["log2fc"] is [F, P], results["plesser"] is [F, P]
```

### Pairwise Comparisons from Multi-Category Designs

With 4 categories (A, B, C, D) and A as reference, the design matrix gives 3 factors: B−A, C−A, D−A. Results vs the reference come directly from the output. For non-reference pairwise comparisons (e.g. B vs C), subtract the posterior parameters — variances add under the independent normal variational guide:

```python
import jax.numpy as jnp
import numpyro.distributions as dist
from itertools import combinations

def pairwise_plesser(svi, factor_names, cutoff=jnp.log(2.0)):
    """Compute plesser for all pairwise comparisons from a multi-category fit.

    Args:
        svi: SVIRunResult from jax_run_pyro
        factor_names: list of factor names matching design matrix columns
        cutoff: significance cutoff in natural log scale (default ln(2))

    Returns:
        dict of {("B", "C"): plesser_array, ...} for all pairs
    """
    lfc_loc = svi.params["log_fc_auto_loc"]      # [F, P]
    lfc_scale = svi.params["log_fc_auto_scale"]  # [F, P]
    results = {}
    for i, j in combinations(range(len(factor_names)), 2):
        diff_loc = lfc_loc[i] - lfc_loc[j]
        diff_scale = jnp.sqrt(lfc_scale[i]**2 + lfc_scale[j]**2)
        results[(factor_names[i], factor_names[j])] = dist.Normal(
            jnp.abs(diff_loc), diff_scale
        ).cdf(cutoff)
    return results

# Usage
pw = pairwise_plesser(svi, ["B", "C", "D"])
for pair, plesser in pw.items():
    print(f"{pair[0]} vs {pair[1]}: {(plesser < 0.05).sum()} DE genes")
```

## Model Variants

```python
# With size factor estimation (default) — for small datasets where you may have
# large bias in sample handling or processing between conditions
results, losses, svi = jax_run_pyro(counts.T, labels, key, use_size_factor_model=True)

# Without size factors — useful when you have a lot of data to make training
# more stable, with a more flexible representation of the data
results, losses, svi = jax_run_pyro(counts.T, labels, key, use_size_factor_model=False)
```

## Replicates

Different random keys give independent inference runs:

```python
for rep in range(5):
    key = jax.random.fold_in(jax.random.PRNGKey(0), rep)
    results, losses, svi = jax_run_pyro(counts.T, labels, key)
```

## Simulated Data

```python
from global_upreg_seq import jax_generate_simulated_data

counts, labels, (log_fc_true, size_factors, base_means) = jax_generate_simulated_data(
    group_size=10, gene_size=30000, median_log_upreg=1.5,
    non_de_fraction=0.25, seed=42,
)
```

## Differences from DESeq2 and Other Methods

- **Minimal filtering:** Only filter genes with fewer than 10 total counts across all samples. Unlike DESeq2, GlobSeq does not use geometric means for normalization, so genes with zero counts in some samples are handled naturally — no need for the strict filtering that DESeq2 requires to avoid undefined geometric means.
- **No normalization step:** GlobSeq estimates size factors and fold changes jointly within the model, rather than as a separate preprocessing step. This avoids the circular dependency where normalization assumes most genes are not DE.
- **Direct posterior inference:** Instead of p-values from a frequentist test, GlobSeq returns `plesser` — the posterior probability that a gene's fold change is small. This means you can make positive claims about non-DE genes (high `plesser`), not just fail to reject the null.
- **Robust to global upregulation:** When a large fraction of genes are DE in the same direction, median-of-ratios normalization (DESeq2, edgeR) systematically underestimates fold changes. GlobSeq's spike-and-slab prior separates DE from non-DE genes during inference, avoiding this bias.

## Platform

```bash
export JAX_PLATFORMS=cpu   # laptops
export JAX_PLATFORMS=cuda  # GPU
```

## Citation

Manuscript in preparation.
