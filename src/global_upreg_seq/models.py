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

class factor_model_nb_nosf(PyroModule):
    def forward(self, x, y, classnum=2):
        N, P = x.shape

        loading_plate = pyro.plate("loading_plate", P, dim=-1)
        factor_plate = pyro.plate("factor_plate", classnum-1, dim=-2)

        #log_global_alpha= pyro.sample("log_global_alpha", dist.Normal(0,3)) 
        with loading_plate:
            log_mu0 = pyro.sample("log_mu0", dist.Normal(5,10))
            alpha_gene = pyro.sample("alpha_gene", dist.LogNormal(-2.0,2.5))
            #alpha_gene_inv = pyro.sample("alpha_gene", dist.HalfNormal(5)) 
        with factor_plate:
            #tau_g = pyro.sample("tau_g", dist.HalfCauchy(3.0))
            tau_g = pyro.sample("tau_g", dist.HalfNormal(3.0))

            with loading_plate:
                #tau_l = pyro.sample("tau_l", dist.HalfCauchy(3.0))
                tau_l = pyro.sample("tau_l", dist.HalfNormal(3.0))

                log_fc = pyro.sample("log_fc", dist.Normal(0, 1.0 * tau_g * tau_l))
                #log_fc = pyro.sample("log_fc", dist.Laplace(0, 1.0))


        with pyro.plate("data", N, dim=-2):
            size_factor_square_minus_one = pyro.sample("sample_alpha", dist.LogNormal(0,1))
            #size_factor_square = pyro.param("sample_alpha", torch.ones(N,1),
            #                                constraint=pyro.distributions.constraints.greater_than(1.))

            #log_size_factor_square= pyro.sample("sample_alpha", dist.Normal(0,2))
            #size_factor_square_raw = pyro.sample("sample_alpha_raw", dist.HalfNormal(0,2))
            #size_factor_square = pyro.deterministic("sample_alpha", torch.exp(size_factor_square_raw-1))


            with loading_plate:
                log_mu = (log_mu0).clamp(min=-4.0, max=18.0) + (log_fc.unsqueeze(0) * y.unsqueeze(-1))
                log_mu = log_mu.squeeze()
                log_mu = torch.clamp(log_mu, min = torch.tensor(-7.0), max=torch.tensor(18.))

                alpha_inv = pyro.deterministic("alpha_inv", torch.clamp(1/(
                    ((1.0+size_factor_square_minus_one) * alpha_gene).clamp(max=300.0)# +
                    #(size_factor_square-2.0*torch.sqrt(size_factor_square) +1.0).clamp(min=0.0)+
                    #( (size_factor_square -1.0)/(20.0+torch.exp(log_mu)) ).clamp(min=-100.0)
                    #(torch.sqrt(size_factor_square) -1 + 1/(10.0+torch.exp(log_mu))) *(torch.sqrt(size_factor_square)-1)
                ),min=1e-4, max=1e4))

                logits = (log_mu - torch.log(alpha_inv)).clamp(min=-15, max=18)
                if torch.isinf(logits).any() or torch.isinf(alpha_inv).any():
                    print(f"Inf detected: logits range [{logits.min()}, {logits.max()}]")
                    print(f"alpha_inv range [{alpha_inv.min()}, {alpha_inv.max()}]")
                pyro.sample("obs", dist.NegativeBinomial(total_count=alpha_inv, logits=logits), obs=x)



def base_guide(model, initial_size_factors, log_mu0_start):

    guide = AutoNormal(
        poutine.block(model, hide=['global_upreg_true']),
        init_loc_fn=init_to_value(values={"log_size_factor": initial_size_factors.unsqueeze(-1),
                                         "log_mu0": log_mu0_start.unsqueeze(0) })
    )
    
    return(guide)


