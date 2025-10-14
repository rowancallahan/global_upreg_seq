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


def prepare_initialization(counts_observed, zero_inflated=False, labels=None, group_size=None)

    if (labels is None) and (group_size is None):
        raise ValueError("Either one of labels for deseq calculation or group size for generalized testing must be supplied")
    if labels is not None:
        raise NotImplementedError("labels not yet implemented here")

    #change the zero increase if you are not doing single cell,
    #TODO add padding for single cell and make this an option
    log_gm = torch.mean(torch.log(counts_observed+0.0001), dim=0)
    strict_mask = torch.isfinite(log_gm)
    happy_mask = torch.ones(gene_size, dtype=torch.bool)
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
    size_factors_0 = log_size_factors[0:group_size]-torch.mean(log_size_factors[0:group_size])
    size_factors_1 = log_size_factors[group_size:]-torch.mean(log_size_factors[group_size:])
    initial_size_factors = torch.cat([size_factors_0, size_factors_1])
     
    return(initial_size_factors, log_mu0_start)


def train(data,
          label,
          model,
          num_iterations=1500,
          guide=None,
          optim=None,
          loss=None):
    
    losses=[]
    logp_size_hist = []

    #SETTING UP GUIDE AND OPTIMIZER HERE
    if optim is None:
        optim=  ClippedAdam({"lr": 0.1, "clip_norm": 100.0, "lrd":0.0005**(1/train_num)})
    if guide is None:
        initial_sf, log_mean_start = prepare_initialization(data, labels=label)
        guide = base_guide(model, initial_sf, log_mean_start)
    
    #SVI TO OPTIMIZE HERE
    svi = SVI(model,guide,optim,
    loss=TraceEnum_ELBO(max_plate_nesting=2) if loss is None else loss)
    
    #MAIN TRAINING LOOP HERE
    pbar = tqdm(range(num_iterations))
    pbar_loss = 1e37
    for j in pbar:
        loss = svi.step(data,y=label)
        if j %50 == 0:
            pbar_loss = loss
            with torch.no_grad():
                trace = poutine.trace(model).get_trace(data, label)
                trace.compute_log_prob()
                logp_size = trace.nodes["log_size_factor"]["log_prob_sum"].item()
                logp_size_hist.append(logp_size)
            
        pbar.set_description(f"Loss: {pbar_loss:.3e}")
        losses.append(loss)

    return(model, guide, (losses, logp_size_hist))

