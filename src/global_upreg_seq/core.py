import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

import pyro
import pyro.distributions as dist

from pyro.infer import Predictive, SVI, Trace_ELBO, TraceMeanField_ELBO, Importance, TraceEnum_ELBO
from pyro.infer.autoguide import AutoNormal, AutoDelta
from pyro.optim import Adam, ClippedAdam

import pyro.poutine as poutine
from pyro.nn import PyroModule
from pyro.nn import PyroSample
from pyro.ops import einsum

def generate_simulated_data(group_size,
                            gene_size=40000,
                            distribution_type="negative_binomial"
                            median_log_upreg = 0.7
                            downreg_fraction = -0.66
                            zero_inflation = 0.2
                            overdispersion_factor = 0.2,
                            log_base_mean_val = 5)


    group_size=group_size
    
    log_base_means = torch.abs(torch.randn(gene_size) * log_base_mean_val)-2
    log_fc = torch.tensor([ sample if test >= downreg_fraction else 0
                           for test,sample in
                           zip(torch.rand(gene_size), (torch.randn(gene_size)+median_log_upreg)) ])
    
    distribution_type = "poisson"
    base_dist = None
    upreg_dist = None
    if distribution_type=="poisson":
        #create a base distribution and upregulated one
        base_dist = dist.Poisson(torch.exp(log_base_means))
        upreg_dist = dist.Poisson(torch.exp(log_base_means+log_fc))
    elif distribution_type == "zero_inflated_poisson":  
        constant_gate = torch.rand(gene_size)*zero_inflation
        base_dist = dist.ZeroInflatedPoisson(torch.exp(log_base_means), gate = constant_gate)
        upreg_dist = dist.ZeroInflatedPoisson(torch.exp(log_base_means+log_fc), gate= constant_gate)

    elif distribution_type == "negative_binomial":
        constant_gate = [0.0]*zero_inflation
        overdisperse = torch.rand(gene_size)*overdispersion_factor

        base_dist = zinb_reparam(torch.exp(log_base_means),
                                 torch.exp(log_base_means+overdisperse),
                                 constant_gate)
        upreg_dist= zinb_reparam(torch.exp(log_base_means+log_fc),
                                 torch.exp(log_base_means+log_fc+overdisperse),
                                 constant_gate)
    elif distribution_type == "zinb":
        constant_gate = torch.rand(gene_size)*zero_inflation
        overdisperse = torch.rand(gene_size)*overdispersion_factor

        base_dist = zinb_reparam(torch.exp(log_base_means),
                                 torch.exp(log_base_means+overdisperse),
                                 constant_gate)
        upreg_dist= zinb_reparam(torch.exp(log_base_means+log_fc),
                                 torch.exp(log_base_means+log_fc+overdisperse),
                                 constant_gate)
    
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

    return (counts_observed, labels)



def zinb_reparam(mean, variance, zero_inflation, eps=1e-6):
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
    variance = torch.tensor(variance, dtype=torch.float)
    zero_inflation = torch.tensor(zero_inflation, dtype=torch.float)

    # Assert variance is at least equal to mean
    assert torch.all(variance >= mean), f"Variance must be >= mean, but got min ratio: {(variance/mean).min()}"
    variance = torch.where(variance == mean, mean + eps, variance)
    
    # Calculate ZINB parameters
    total_count = torch.square(mean)/ (variance-mean) #r = mu^2/(sigma^2-mu)
    probability = mean/variance #p = mu/(sigma^2)
    probs = torch.clamp((1 - probability), 0+eps,1-eps) # success probability for Pyro
    
    # Create ZINB distribution
    zinb = dist.ZeroInflatedNegativeBinomial(
        gate=zero_inflation,
        total_count=total_count,
        probs=probs
    )
    
    return zinb


