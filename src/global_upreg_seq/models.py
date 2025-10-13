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

