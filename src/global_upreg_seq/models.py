import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

import pyro
import pyro.distributions as dist

from pyro.infer import Predictive, SVI, Trace_ELBO, TraceMeanField_ELBO, Importance, TraceEnum_ELBO
from pyro.infer.autoguide import AutoNormal, AutoDelta
from pyro.infer.autoguide.initialization import init_to_value

from pyro.optim import Adam, ClippedAdam

import pyro.poutine as poutine
from pyro.nn import PyroModule
from pyro.nn import PyroSample
from pyro.ops import einsum


class factor_model_poisson(PyroModule):
    def forward(self, x, y, classnum=2):
        N, P = x.shape
        
        loading_plate = pyro.plate("loading_plate", P, dim=-1)
        factor_plate = pyro.plate("factor_plate", classnum-1, dim=-2)
        log_shared_mean = pyro.sample("log_shared_mean", dist.Normal(0., 100.))

        with loading_plate:
            log_mu0 = pyro.sample("log_mu0", dist.Normal(0., 100.))# baseline log-mean per feature
            
        with factor_plate:
            global_upreg_true = pyro.sample("global_upreg_true", dist.Bernoulli(0.2), infer={"enumerate": "parallel"})
            global_upreg_value = pyro.sample("global_upreg_value", dist.Normal(0,5.0))

            global_upreg = pyro.deterministic("global_upreg", global_upreg_value*global_upreg_true)

            tau_g = pyro.sample("tau_g", dist.HalfCauchy(1.))

            with loading_plate:
                tau_l = pyro.sample("tau_l", dist.HalfCauchy(1.))
                log_fc = pyro.sample("log_fc", dist.Normal(global_upreg, 1.0*tau_g*tau_l)) 


        with pyro.plate("data", N, dim=-2) as sample_index:
            log_size_factor= pyro.sample("log_size_factor", dist.Normal(0,5.0))

            mean_0 = log_size_factor[y==0].mean()
            mean_1 = log_size_factor[y==1].mean()
            mean_diff = pyro.deterministic("mean_diff", mean_1 - mean_0)
            pyro.factor("size_factor_diff_penalty",
                        dist.Normal(0., 0.001/torch.sqrt(torch.tensor(N))).log_prob(mean_diff))
            
            with loading_plate:
                log_mu = log_mu0 + (log_fc.unsqueeze(0)*y.unsqueeze(-1)) 
                log_mu = log_mu.squeeze()
                mu = torch.exp(log_mu+log_size_factor).clamp(min=1e-30, max=1e30)
                pyro.sample("obs", dist.Poisson(mu), obs=x)

class factor_model_onehot_mixedeffect(PyroModule):
    def forward(self, x, y, random_factor=0):
        N, P = x.shape
        N, F = y.shape
        
        loading_plate = pyro.plate("loading_plate", P, dim=-1)
        factor_plate = pyro.plate("factor_plate", F, dim=-2)
        random_factor_plate = pyro.plate("random_factor_plate", random_factor, dim=-2)

        #still want a base mean so that log likelihood doesn't go crazy
        with loading_plate:
            log_mu0 = pyro.sample("log_mu0", dist.Normal(0., 100.))# baseline log-mean per feature
            
        with factor_plate:
            global_upreg_true = pyro.sample("global_upreg_true", dist.Bernoulli(0.2), infer={"enumerate": "parallel"})
            global_upreg_value = pyro.sample("global_upreg_value", dist.Normal(0,5.0))
            global_upreg = pyro.deterministic("global_upreg", global_upreg_value*global_upreg_true)
            tau_g = pyro.sample("tau_g", dist.HalfCauchy(1.))

            with loading_plate:
                tau_l = pyro.sample("tau_l", dist.HalfCauchy(1.))
                log_mean = pyro.sample("log_mean", dist.Normal(global_upreg, 1.0*tau_g*tau_l)) 

        with random_factor_plate:
            tau_g = pyro.sample("tau_g", dist.HalfCauchy(1.))
            with loading_plate:
                tau_l = pyro.sample("tau_l", dist.HalfCauchy(1.))
                log_mean = pyro.sample("log_mean", dist.Normal(0, 1.0*tau_g*tau_l)) 



        with pyro.plate("data", N, dim=-2) as sample_index:
            log_size_factor= pyro.sample("log_size_factor", dist.Normal(0,5.0))

            #random_factor= pyro.sample("random_factor", dist.Normal(0,5.0))
            #multiply random factor times its features then you have mixed effects modelling?
            #Then you have the supervised portion and the un supervised portion
            #likely this will have to be second paper
            #second paper, expand further into single cell, expand further into supervised
            #go hard on optimization
            #third paper go hard on infinite vectors with infinite clusters

            #TODO figure this one out for multiple classes
            #mean_0 = log_size_factor[y==0].mean()
            #mean_1 = log_size_factor[y==1].mean()
            #mean_diff = pyro.deterministic("mean_diff", mean_1 - mean_0)
            #pyro.factor("size_factor_diff_penalty",
            #            dist.Normal(0., 0.001/torch.sqrt(torch.tensor(N))).log_prob(mean_diff))
            
            with loading_plate:
                log_mu = (log_mean.unsqueeze(0)*y.unsqueeze(-1)) #multiply each feature times either
                #its onehot encoding or its measured value
                mu = torch.exp(log_mu+log_size_factor).clamp(min=1e-30, max=1e30)
                pyro.sample("obs", dist.Poisson(mu), obs=x)




def base_guide(model, initial_size_factors, log_mu0_start):

    guide = AutoNormal(
        poutine.block(model, hide=['global_upreg_true']),
        init_loc_fn=init_to_value(values={"log_size_factor": initial_size_factors.unsqueeze(-1),
                                         "log_mu0": log_mu0_start.unsqueeze(0) })
    )
    
    return(guide)


