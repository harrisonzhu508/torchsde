import sys 
sys.path.append("../..")
import argparse
import logging
import math
import os
import random
from collections import namedtuple
from typing import Optional, Union
from multiscale_sde.util import softplus_inverse

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from torch import distributions, nn, optim

import torchsde
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs("plots/", exist_ok=True)

def manual_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

manual_seed(0)
plt.style.use('seaborn-poster')

# SDE solver config
method = "euler"
dt = 1e-3
adaptive = False
rtol = 1e-3
atol = 1e-3
adjoint = False
kl_anneal_iters = 100
dpi = 300
batch_size=512
sdeint_fn = torchsde.sdeint


# generate data
ts = np.linspace(0., 2., 100)
ts = np.sort(ts)
ts = torch.Tensor(ts).to(device)
class AgeHeartSDE(torchsde.SDEIto):

    def __init__(self, period=0.2, device="cpu"):
        super().__init__(noise_type="diagonal")
        self.device = device
        # let's do t\in [0, 2]
        self.period = period
        self.y0_mean = torch.Tensor([0, 0]).to(device) 
        self.y0_std = torch.Tensor([1, 1]).to(device) 
        

    def f(self, t, y):  # drift
        y_1 = y[:,:1]
        drift_1 = torch.ones(y_1.size(), device=self.device)
        drift_2 = y_1*20*torch.cos(2*np.pi * t / self.period)
        drift = torch.cat([drift_1, drift_2], axis=1)
        return drift
    
    def g(self, t, y):  # Diffusion.
        return torch.ones(y.size(), device=self.device)*math.sqrt(2)

    def forward(self, ts, batch_size, eps=None):
        eps = torch.randn(batch_size, 1).to(self.y0_std) if eps is None else eps
        y0 = self.y0_mean + eps * self.y0_std
        ys = sdeint_fn(
            sde=self,
            y0=y0,
            ts=ts,
            method=method,
            dt=dt,
            adaptive=adaptive,
            rtol=rtol,
            atol=atol,
            names={'drift': 'f', 'diffusion': 'g'}
        )
        return ys
dataSDE = AgeHeartSDE(period=0.2, device=device)
y = dataSDE.forward(ts[:,None], batch_size=5)
y_plot = y.cpu().detach().numpy()
ts_plot = ts.detach().cpu().numpy()
fig, (ax1, ax2) = plt.subplots(2, frameon=False, figsize=(20,10), sharex=True)
ax1.plot(ts_plot, y_plot[:,:,0])
ax2.plot(ts_plot, y_plot[:,:,1])
plt.savefig("data.pdf")
## select training data
ts_train = ts[:, None]
y_train = y[:,0,:]


##########################################
################SDE Model#################
##########################################
# pick SDEInt
sdeint_fn = torchsde.sdeint

# pick model hyperparameters
method = "euler"
dt = 1e-2
adaptive = False 
rtol = 1e-3
atol = 1e-3

class LinearScheduler(object):
    def __init__(self, iters, maxval=1.0):
        self._iters = max(1, iters)
        self._val = maxval / self._iters
        self._maxval = maxval

    def step(self):
        self._val = min(self._maxval, self._val + self._maxval / self._iters)

    @property
    def val(self):
        return self._val

def _stable_division(a, b, epsilon=1e-7):
    b = torch.where(b.abs().detach() > epsilon, b, torch.full_like(b, fill_value=epsilon) * b.sign())
    return a / b

class LatentSDE(torchsde.SDEIto):

    def __init__(self, device="cpu"):
        super(LatentSDE, self).__init__(noise_type="diagonal")

        self.device = device

        # Approximate posterior drifts
        self.net1 = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        self.net2 = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Prior drifts
        self.prior_net1 = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        self.prior_net2 = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # set prior and posterior initial conditions
        self.register_buffer("py0_mean", torch.zeros((2)))
        self.register_buffer("py0_std", torch.ones((2)))
        self.qy0_mean = nn.Parameter(torch.tensor([0., 0.]), requires_grad=True)
        self.qy0_std_unconstrained = nn.Parameter(torch.tensor([1., 1.]), requires_grad=True)
    @property
    def qy0_std(self):
        return nn.functional.softplus(self.qy0_std_unconstrained)

    def f(self, t, y):  # Approximate posterior drift.
        drift1 = self.net1(y[:,:1])
        drift2 = self.net2(y)
        drift = torch.cat([drift1, drift2], axis=1)
        return drift

    def f_prior(self, t, y):  # Prior drift.
        drift1 = self.prior_net1(y[:,:1])
        drift2 = self.prior_net2(y)
        drift = torch.cat([drift1, drift2], axis=1)
        return drift
    
    def g(self, t, y):  # Shared diffusion.
        """
        """
        # return self.sigma.repeat(y.size(0), 1)
        # drift matrix = sigma^2 * sqrt(2 / l) for Matern12 kernel (diffusion of BM is 2/l)
        return torch.ones(y.size(), device=self.device) * math.sqrt(2)

    def f_aug(self, t, y):  # Drift for augmented dynamics with logqp term.
        y = y[:,:-1]
        f, g, f_prior = self.f(t, y), self.g(t, y), self.f_prior(t, y)
        u = _stable_division(f - f_prior, g)
        f_logqp = .5 * (u ** 2).sum(dim=1, keepdim=True)
        return torch.cat([f, f_logqp], dim=1)

    def g_aug(self, t, y):  # Diffusion for augmented dynamics with logqp term.
        y = y[:,:2]
        g = self.g(t, y)
        g_logqp = torch.zeros_like(y[:,:1])
        return torch.cat([g, g_logqp], dim=1)

    def forward(self, ts, batch_size, eps=None):
        eps = torch.randn(batch_size, 1).to(self.qy0_std) if eps is None else eps
        y0 = self.qy0_mean + eps * self.qy0_std
        qy0 = distributions.Normal(loc=self.qy0_mean, scale=self.qy0_std)
        py0 = distributions.Normal(loc=self.py0_mean, scale=self.py0_std)
        logqp0 = distributions.kl_divergence(qy0, py0) # KL(t=0).

        aug_y0 = torch.cat([y0, torch.zeros(batch_size, 1).to(y0)], dim=1)
        aug_ys = sdeint_fn(
            sde=self,
            y0=aug_y0,
            ts=ts,
            method=method,
            dt=dt,
            adaptive=adaptive,
            rtol=rtol,
            atol=atol,
            names={'drift': 'f_aug', 'diffusion': 'g_aug'}
        )
        ys, logqp_path = aug_ys[:, :, :2], aug_ys[-1, :, -1]
        logqp = logqp0.sum() + logqp_path.mean(dim=0)  # KL(t=0) + KL(path).
        return ys, logqp

    def sample_p(self, ts, batch_size, eps=None, bm=None):
        eps = torch.randn(batch_size, 1).to(self.py0_mean) if eps is None else eps
        y0 = self.py0_mean + eps * self.py0_std
        
        yt = sdeint_fn(self, y0, ts, bm=bm, method='srk', dt=dt, names={'drift': 'f_prior'})
        return yt

    def sample_q(self, ts, batch_size, eps=None, bm=None):
        eps = torch.randn(batch_size, 1).to(self.qy0_mean) if eps is None else eps
        y0 = self.qy0_mean + eps * self.qy0_std
        yt = sdeint_fn(self, y0, ts, bm=bm, method='srk', dt=dt, names={"drift" : "f"})
        return yt

##########################################
################Training#################
##########################################
model = LatentSDE(device=device).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-2)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=.999)
kl_scheduler = LinearScheduler(iters=100)
for global_step in tqdm.tqdm(range(300+1)):
    optimizer.zero_grad()
    zs, kl = model(ts=ts_train, batch_size=128)
    zs = zs.squeeze()
    # zs = zs[1:-1]  # Drop first and last which are only used to penalize out-of-data region and spread uncertainty.
    likelihood_constructor = distributions.Normal
    likelihood = likelihood_constructor(loc=zs, scale=0.01)
    logpy = likelihood.log_prob(y_train[:,None,:]).sum(dim=[0,-1]).mean(dim=0)
    loss = -logpy + kl * kl_scheduler.val
    loss.backward()

    optimizer.step()
    scheduler.step()
    kl_scheduler.step()
    if global_step % 50 == 0:
        y_pred_plot = zs.cpu().detach().numpy()
        y_pred = zs.mean(1).cpu().detach().numpy()
        fig, (ax1, ax2) = plt.subplots(2, frameon=False, figsize=(20,10), sharex=True)
        ax1.plot(ts_plot, y_pred[:,0], color="C0")
        ax2.plot(ts_plot, y_pred[:,1], color="C0")

        # plot the credible interval
        y_pred_plot[:,:,0] = np.sort(y_pred_plot[:,:,0], axis=1)
        y_pred_plot[:,:,1] = np.sort(y_pred_plot[:,:,1], axis=1)
        percentile = 0.95
        idx = int((1 - percentile) / 2. * y_pred_plot.shape[1])
        y_pred_plot_bottom, y_pred_plot_top = y_pred_plot[:, idx,0], y_pred_plot[:, -idx,0]
        ax1.fill_between(ts_plot, y_pred_plot_bottom, y_pred_plot_top, alpha=0.3, color="C0")

        y_pred_plot_bottom, y_pred_plot_top = y_pred_plot[:, idx,1], y_pred_plot[:, -idx,1]
        ax2.fill_between(ts_plot, y_pred_plot_bottom, y_pred_plot_top, alpha=0.3, color="C0")

        ax1.scatter(ts_plot, y_plot[:,0,0], color="black")
        ax2.scatter(ts_plot, y_plot[:,0,1], color="black")

        ax1.set_ylabel("First Scale")
        ax2.set_ylabel("First Scale")
        
        ax1.set_xlabel("Time $t$")
        ax2.set_xlabel("Time $t$")
        plt.savefig(f"plots/{global_step}.pdf")
    