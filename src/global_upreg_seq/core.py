import torch
from tqdm import tqdm

import pyro
import pyro.distributions as dist
from pyro.infer import Predictive, SVI, Trace_ELBO, TraceMeanField_ELBO, Importance, TraceEnum_ELBO
from pyro.infer.autoguide import AutoNormal, AutoDelta
from pyro.optim import Adam, ClippedAdam

import pyro.poutine as poutine
from .models import base_guide

import numpy as np
import matplotlib.pyplot as plt

def generate_simulated_data(group_size,
                            gene_size=40000,
                            distribution_type="negative_binomial",
                            median_log_upreg = 0.7,
                            downreg_fraction = -0.66,
                            a_0= 0.01,
                            a_1 = 5,
                            log_base_mean_val = 5):

    """Generate simulated RNA-seq count data with differential expression.

    Parameters
    ----------
    group_size : int
        Number of samples per condition (base and upregulated).
    gene_size : int, default=40000
        Number of genes to simulate.
    distribution_type : str, default="negative_binomial"
        Distribution type: "poisson", "zero_inflated_poisson", "negative_binomial", or "zinb".
    median_log_upreg : float, default=0.7
        Median log fold change for upregulated genes (natural log scale).
    downreg_fraction : float, default=-0.66
        Fraction threshold for differentially expressed genes (negative selects upregulated genes).
    a_0 : float, default=0.01
        Overdispersion scaling parameter (mean-dependent component).
    a_1 : float, default=5
        Overdispersion baseline parameter.
    log_base_mean_val : float, default=5
        Scale parameter for gene expression base means (natural log scale).

    Returns
    -------
    tuple
        - counts_observed (torch.Tensor): Shape (2*group_size, gene_size) count matrix.
        - labels (torch.Tensor): Shape (2*group_size,) condition labels (0=base, 1=upreg).
        - (log_fc, size_factors, log_base_means): Ground truth parameters.
    """

    group_size=group_size
    log_base_means = torch.abs(torch.randn(gene_size) * log_base_mean_val)-2
    log_fc = torch.tensor([ sample if test >= downreg_fraction else 0
                           for test,sample in
                           zip(torch.rand(gene_size), (torch.randn(gene_size)+median_log_upreg)) ])
    
    base_dist = None
    upreg_dist = None
    if distribution_type=="poisson":
        #create a base distribution and upregulated one
        base_dist = dist.Poisson(torch.exp(log_base_means))
        upreg_dist = dist.Poisson(torch.exp(log_base_means+log_fc))

    elif distribution_type == "negative_binomial":
        constant_gate = torch.rand(gene_size)* torch.tensor(0.0)
        
        base_mean = torch.exp(log_base_means)
        overdispersion = (a_1/base_mean +a_0) * torch.randn(gene_size)
        base_dist = nb_reparam(base_mean,
                                overdispersion)

        upreg_mean = torch.exp(log_base_means + log_fc)
        overdispersion_upreg =  (a_1/upreg_mean +a_0) * torch.randn(gene_size)
        upreg_dist= nb_reparam(upreg_mean,
                                overdispersion_upreg)
    
    #now take samples from normal means and upreg mans
    base_samples = base_dist.sample((group_size,))
    upreg_samples = upreg_dist.sample((group_size,))
    combined_samples = torch.cat([base_samples, upreg_samples],dim=0)
    #now create the labels
    labels = torch.cat([torch.zeros(group_size, dtype=torch.int), torch.ones(group_size, dtype=torch.int)])
    
    #now sample the size factors and create observed counts
    size_factors_dist = dist.LogNormal(0.,1.)
    size_factors = size_factors_dist.sample((group_size*2,))
    counts_observed= (combined_samples * size_factors.unsqueeze(-1)).round()

    return (counts_observed, labels, (log_fc, size_factors, log_base_means))



def nb_reparam(mean, alpha, eps=1e-6):
    """
    probability here is flipped compared to wikipedia probability
    here am translating wikipedia nomenclature to pyro
    probs is actually the probability of failure actually
    total_counts= r
    probs = 1-p
    mean counts = (r *(1-p))/p)
    translating this into the pyro
    we get: mean counts = ((total_counts * probs)/( 1-probs)

    """
    mean = torch.tensor(mean, dtype=torch.float)
    alpha = torch.tensor(alpha, dtype=torch.float)
    alpha_inv = 1.0/alpha
    logit = torch.log(mean) + torch.log(alpha) 

    # Create ZINB distribution
    zinb = dist.NegativeBinomial(
        total_count=alpha_inv,
        logit=logit
    )
    
    return zinb


def prepare_initialization(counts_observed, labels, zero_inflated=False, group_size=None):

    #change the zero increase if you are not doing single cell,
    #TODO add padding for single cell and make this an option
    log_gm = torch.mean(torch.log(counts_observed+0.0001), dim=0)
    strict_mask = torch.isfinite(log_gm)
    happy_mask = torch.ones(counts_observed.shape[1], dtype=torch.bool)
    mask=strict_mask
    
    #now we calculate an easy best first estimate for base means here
    #TODO have this only make the count of the base version
    counts_used = counts_observed[:,mask]
    sample_means = torch.mean(counts_used, dim=0)
    log_mu0_start = torch.log(sample_means+0.01)
    
    #this always has to be a strict mask to get initialization
    #now we calculate the median of ratios to get an initialization to start off of
    #first we take the ratio against the geometric mean                           
    ratios = torch.log(counts_observed[:, strict_mask]+0.0001) - log_gm[strict_mask]        # log ratios
    log_size_factors = torch.median(ratios, dim=1).values               # per-sample log-SF

    initial_size_factors = log_size_factors - torch.where(labels.bool(), 
        log_size_factors[labels.bool()].mean(), log_size_factors[~labels.bool()].mean())
     
    return(initial_size_factors, log_mu0_start)


def train(data,
          labels,
          model,
          num_iterations=1500,
          guide=None,
          optim=None,
          loss=None):
    
    losses=[]
    logp_size_hist = []
    #SETTING UP GUIDE AND OPTIMIZER HERE
    if optim is None:
        optim=  ClippedAdam({"lr": 0.1, "clip_norm": 100.0, "lrd":0.0005**(1/num_iterations)})
    if guide is None:
        initial_sf, log_mean_start = prepare_initialization(data, labels= labels)

        guide = base_guide(model, initial_sf, log_mean_start)

    #SVI TO OPTIMIZE HERE
    svi = SVI(model,guide,optim,
    loss=Trace_ELBO(max_plate_nesting=2) if loss is None else loss)
    
    #MAIN TRAINING LOOP HERE
    pbar = tqdm(range(num_iterations))
    pbar_loss = 1e37
    for j in pbar:
        loss = svi.step(data,y=labels)
        if j %50 == 0:
            pbar_loss = loss

            
        pbar.set_description(f"Loss: {pbar_loss:.3e}")
        losses.append(loss)

    return(model, guide, (losses, logp_size_hist))


def calculate_foldchange_bf(guide, cutoff, make_plot=True, non_bf=False):
    log_fc_loc = torch.abs(guide.locs.log_fc.detach())
    log_fc_scale = guide.scales.log_fc
    cutoff = torch.tensor(cutoff)
    dist = torch.distributions.Normal(loc=log_fc_loc, scale=log_fc_scale)
    
    bayes_factor = ((1 - dist.cdf(cutoff)) / dist.cdf(cutoff)).clamp(min=1e-37)
    log10_bf = np.log10(bayes_factor.detach().cpu().numpy())
    log_fc_np = guide.locs.log_fc.detach().cpu().numpy()

    if non_bf:
        likelihood_less= (dist.cdf(cutoff)).clamp(min=1e-37)
        log10_bf = -np.log10(likelihood_less.detach().cpu().numpy())

    if make_plot:
        fig, ax = plot_volcano(log_fc_np, log10_bf, float(cutoff))  # Convert to float
        return log10_bf, log_fc_np, fig, ax
    
    return log10_bf, log_fc_np


def plot_volcano(log_fc_np, log10_bf, cutoff, bf_threshold=1.0):
    """Create a publication-quality volcano plot for differential expression analysis."""
    
    # Classify genes
    upregulated = (log_fc_np > cutoff) & (log10_bf > bf_threshold)
    downregulated = (log_fc_np < -cutoff) & (log10_bf > bf_threshold)
    not_significant = ~(upregulated | downregulated)
    
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    
    # Plot points
    ax.scatter(log_fc_np[not_significant], log10_bf[not_significant], c='#BDBDBD', 
               s=50, alpha=0.4, edgecolors='none', label=f'NS (n={not_significant.sum()})', rasterized=True)
    ax.scatter(log_fc_np[downregulated], log10_bf[downregulated], c='#2E86DE', 
               s=60, alpha=0.8, edgecolors='none', label=f'Down (n={downregulated.sum()})', rasterized=True)
    ax.scatter(log_fc_np[upregulated], log10_bf[upregulated], c='#EE5A6F', 
               s=60, alpha=0.8, edgecolors='none', label=f'Up (n={upregulated.sum()})', rasterized=True)
    
    # Threshold lines
    ax.axhline(y=bf_threshold, color='#757575', linestyle='--', linewidth=2, alpha=0.7, zorder=0)
    ax.axvline(x=cutoff, color='#757575', linestyle='--', linewidth=2, alpha=0.7, zorder=0)
    ax.axvline(x=-cutoff, color='#757575', linestyle='--', linewidth=2, alpha=0.7, zorder=0)
    
    # Styling with better fonts
    ax.set_xlabel('Log2 Fold Change', fontsize=28, fontweight='normal', family='sans-serif')
    ax.set_ylabel('Log10 Bayes Factor', fontsize=28, fontweight='normal', family='sans-serif')
    ax.set_title('Differential Expression Analysis', fontsize=32, fontweight='bold', 
                 family='sans-serif', pad=20)
    
    # Limits
    x_max = np.max(np.abs(log_fc_np)) * 1.1
    log10_bf_finite = log10_bf[np.isfinite(log10_bf)]
    y_max = np.max(log10_bf_finite) * 1.1 if len(log10_bf_finite) > 0 else 3.0
    ax.set_xlim(-x_max, x_max)
    ax.set_ylim(-0.3, max(y_max, 2.5))
    
    # Clean styling
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.tick_params(axis='both', which='major', labelsize=22, width=2, length=6)
    
    # Better legend
    ax.legend(loc='upper left', framealpha=1.0, fontsize=20, frameon=True, 
              edgecolor='black', fancybox=False, shadow=False)
    
    plt.tight_layout()
    
    return fig, ax