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

        with loading_plate:
            #log_mu0= pyro.sample("log_mu0", dist.Normal(0,10))
            log_mu0= pyro.sample("log_mu0", dist.Normal(0,10))

            alpha_gene = pyro.sample("alpha_gene", dist.LogNormal(-1,2.0))

        midpoint = pyro.sample("uninformative_cutoff", dist.Normal(0,1))
        gate_scale = pyro.sample("uninformative_scale", dist.LogNormal(0,1))
        with factor_plate:
            #tau_g = pyro.sample("tau_g", dist.HalfCauchy(3.0))
            tau_g = pyro.sample("tau_g", dist.HalfNormal(3.0))
            with loading_plate:
                eps = torch.tensor(5e-2)
                size_gate = torch.sigmoid((log_mu0 - midpoint ) / gate_scale)
                #tau_l = pyro.sample("tau_l", dist.HalfCauchy(3.0))
                tau_l = pyro.sample("tau_l", dist.HalfNormal(3.0))
                log_fc = pyro.sample("log_fc", dist.Normal(0, eps +(size_gate * tau_g * tau_l) ))
                #log_fc = pyro.sample("log_fc", dist.Laplace(0, 1.0))

        with pyro.plate("data", N, dim=-2):
            size_factor_square_minus_one = pyro.sample("sample_alpha", dist.HalfNormal(1))


            with loading_plate:
                log_mu = (log_mu0)+ (log_fc.unsqueeze(0) * y.unsqueeze(-1))
                log_mu = log_mu.squeeze()
                log_mu = (log_mu).clamp( min = torch.tensor(-7.0), max=torch.tensor(25.))

                alpha_inv = pyro.deterministic("alpha_inv", (1/(
                    (1.0 + size_factor_square_minus_one) * alpha_gene #+
                    #torch.exp(-1.0*(log_mu + 2.5))
                )))#.clamp(min=1e-4, max=1e4))
                logits = (log_mu - torch.log(alpha_inv))#.clamp(min=-15, max=18)

                pyro.sample("obs", dist.NegativeBinomial(total_count=alpha_inv, logits=logits), obs=x)
                #pyro.sample("obs", dist.NegativeBinomial(total_count=alpha_inv, logits=logits).mask((x>0)), obs=x)
