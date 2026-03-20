# GlobSeq

Bayesian differential expression analysis for RNA-seq data under global upregulation. GlobSeq uses a spike-and-slab prior to jointly estimate gene-level fold changes and sample-level size factors, avoiding the median-of-ratios normalization bias that affects standard methods (DESeq2, edgeR) when most genes are DE.

## Installation

```bash
# Clone and install
git clone https://github.com/rowancallahan/global_upreg_seq.git
cd global_upreg_seq
pip install -e .

# For CPU-only (recommended for laptops)
pip install "jax[cpu]==0.9.1" numpyro==0.20.0
```

Requires Python ≥ 3.10. If you have GPU access, install `jax[cuda12]` instead of `jax[cpu]`.

## Quick Start

### Analyzing Real RNA-seq Data

```python
import numpy as np
import pandas as pd
import jax
from global_upreg_seq import jax_run_pyro

# Load count matrix (genes × samples) and condition labels
counts = pd.read_csv("counts.csv", index_col=0)  # genes × samples
labels = np.array([0, 0, 0, 1, 1, 1])  # 0=control, 1=treatment

# Run GlobSeq (pass counts as samples × genes)
key = jax.random.PRNGKey(0)
results, losses, svi = jax_run_pyro(
    counts.values.T,  # [N, P] — samples × genes
    labels,
    key,
    iterations=3000,
    use_size_factor_model=True,
)

# Build results table
de_results = pd.DataFrame({
    "gene": counts.index,
    "log2fc": results["log2fc"],
    "plesser": results["plesser"],
})

# Call significant genes (plesser < 0.05 and |log2fc| > 1)
de_results["significant"] = (de_results["plesser"] < 0.05) & (de_results["log2fc"].abs() > 1)
print(f"DE genes: {de_results['significant'].sum()}")
```

### Interpreting Results

- **`log2fc`**: Posterior mean log₂ fold change per gene
- **`plesser`**: P(|log fold change| < ln(2)) — the probability the true effect is small
  - Lower = more significant. Call DE at `plesser < 0.05`
  - Analogous to a p-value but derived from the full posterior

### Finding Stably Expressed (Non-DE) Genes

A key advantage of GlobSeq's Bayesian framework is that you can directly identify genes with high confidence of *no* differential expression — useful for reference gene selection, normalization controls, or stable biomarker discovery.

```python
# plesser = P(|log_fc| < ln(2)) — HIGH plesser means confidently non-DE
de_results["stable"] = de_results["plesser"] > 0.95  # >95% probability of no change
stable_genes = de_results[de_results["stable"]].sort_values("plesser", ascending=False)
print(f"Stably expressed genes: {len(stable_genes)}")
```

Unlike frequentist methods that can only fail to reject the null, `plesser` directly quantifies the probability that a gene's fold change is small — giving positive evidence for stability.

### Using Design Matrices

GlobSeq supports arbitrary design matrices, not just two-group comparisons. The `labels` argument is actually a design matrix `x` of shape `[N, F]` where F is the number of factors. When you pass a 1D array of 0s and 1s, it's automatically reshaped to `[N, 1]`.

For multi-factor designs, build your design matrix with [patsy](https://patsy.readthedocs.io/) or [formulaic](https://matthewwardrop.github.io/formulaic/):

```python
import pandas as pd
import numpy as np
from formulaic import model_matrix  # pip install formulaic

# Sample metadata
metadata = pd.DataFrame({
    "condition": ["ctrl", "ctrl", "ctrl", "treat", "treat", "treat",
                  "ctrl", "ctrl", "ctrl", "treat", "treat", "treat"],
    "batch":     ["A", "A", "A", "A", "A", "A",
                  "B", "B", "B", "B", "B", "B"],
})

# Two-group (simple case) — same as passing labels directly
X_simple = model_matrix("~ condition", metadata)
# Drops intercept since GlobSeq has its own (log_mu0)
X = np.array(X_simple)[:, 1:]  # keep only the condition column, shape [12, 1]

# Multi-factor: condition + batch effect
X_multi = model_matrix("~ condition + batch", metadata)
X = np.array(X_multi)[:, 1:]  # drop intercept, shape [12, 2]
# Column 0 = condition effect, Column 1 = batch effect
```

Then pass the design matrix directly:

```python
results, losses, svi = jax_run_pyro(
    counts.values.T,  # [N, P]
    X,                 # [N, F] design matrix
    key,
    iterations=3000,
)

# results["log2fc"] is now [F, P] — one fold-change per factor per gene
# results["plesser"] is [F, P] — significance per factor per gene
# Column 0 = condition effect, Column 1 = batch effect
```

**With patsy** (alternative to formulaic):

```python
import patsy

# Build design matrix from formula
X = patsy.dmatrix("~ condition + batch", metadata, return_type="dataframe")
X = np.array(X)[:, 1:]  # drop intercept
```

**Important:** Always drop the intercept column from the design matrix — GlobSeq models its own intercept via `log_mu0`. The remaining columns correspond to the factors in `log_fc [F, P]`, so the first factor's results are in `results["log2fc"][0, :]` and `results["plesser"][0, :]`.

### Model Variants

```python
# With size factor estimation (default) — use when normalization bias is expected
results, losses, svi = jax_run_pyro(counts.T, labels, key, use_size_factor_model=True)

# Without size factors — use when data is already normalized or bias is minimal
results, losses, svi = jax_run_pyro(counts.T, labels, key, use_size_factor_model=False)
```

### Running Replicates

Different random keys give independent inference runs:

```python
for rep in range(5):
    key = jax.random.fold_in(jax.random.PRNGKey(0), rep)
    results, losses, svi = jax_run_pyro(counts.T, labels, key)
```

### Simulated Data

```python
from global_upreg_seq import jax_generate_simulated_data

counts, labels, (log_fc_true, size_factors, base_means) = jax_generate_simulated_data(
    group_size=10,          # samples per condition
    gene_size=30000,        # number of genes
    median_log_upreg=1.5,   # effect size (natural log)
    non_de_fraction=0.25,   # fraction with no DE (75% upregulated)
    seed=42,
)
```

### Setting the Compute Platform

```bash
# Set before running Python (CPU)
export JAX_PLATFORMS=cpu

# For GPU
export JAX_PLATFORMS=cuda
```

## Citation

Manuscript in preparation.
