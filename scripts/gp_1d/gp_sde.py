# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Latent SDE fit to a single time series with uncertainty quantification."""
import sys 
sys.path.append("../..")
import argparse
import logging
import math
import os
import random
from collections import namedtuple
from typing import Optional, Union
import gpytorch
from gpytorch.kernels import RBFKernel, MaternKernel
from multiscale_sde.util import softplus_inverse



import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from torch import distributions, nn, optim
import torchsde
from multiscale_sde.gp import MaternSDEKernel

# w/ underscore -> numpy; w/o underscore -> torch.
Data = namedtuple('Data', ['ts_', 'ts_ext_', 'ts_vis_', 'ts', 'ts_ext', 'ts_vis', 'ys', 'ys_'])


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


class EMAMetric(object):
    def __init__(self, gamma: Optional[float] = .99):
        super(EMAMetric, self).__init__()
        self._val = 0.
        self._gamma = gamma

    def step(self, x: Union[torch.Tensor, np.ndarray]):
        x = x.detach().cpu().numpy() if torch.is_tensor(x) else x
        self._val = self._gamma * self._val + (1 - self._gamma) * x
        return self._val

    @property
    def val(self):
        return self._val


def str2bool(v):
    """Used for boolean arguments in argparse; avoiding `store_true` and `store_false`."""
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')


def manual_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _stable_division(a, b, epsilon=1e-7):
    b = torch.where(b.abs().detach() > epsilon, b, torch.full_like(b, fill_value=epsilon) * b.sign())
    return a / b

class LatentSDE(torchsde.SDEIto):

    def __init__(self, smoothness=0.5, variance=1, lengthscale=1, fix_variance=True, fix_lengthscale=True, device="cpu"):
        super(LatentSDE, self).__init__(noise_type="diagonal")
        # define kernel function
        self.kernel = MaternSDEKernel(smoothness=smoothness, variance=variance, lengthscale=lengthscale, fix_variance=fix_variance, fix_lengthscale=fix_lengthscale, device=device)
        self.device = device
        
        # initial distribution p(y0).
        # self.register_buffer("py0_mean", torch.tensor([[0.0]]))
        self.register_buffer("py0_mean", torch.zeros((self.kernel.state_dim())))

        # Approximate posterior drift: Takes in 2 positional encodings and the state.
        self.net = nn.Sequential(
            nn.Linear(2 + self.kernel.state_dim(), 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, self.kernel.state_dim())
        )
        # Initialization trick from Glow.
        # self.net[-1].weight.data.fill_(0.)
        # self.net[-1].bias.data.fill_(0.)

        # q(y0).
        # logvar = math.log(1/ 2)
        
        # self.qy0_mean = nn.Parameter(torch.tensor([[0.0]]), requires_grad=True)
        # self.qy0_logvar = nn.Parameter(torch.tensor([[logvar]]), requires_grad=True)

        self.qy0_mean = nn.Parameter(torch.zeros((self.kernel.state_dim())), requires_grad=True)
        std_init = torch.ones((self.kernel.state_dim()))
        std_init = softplus_inverse(std_init)
        self.qy0_std_unconstrained = nn.Parameter(std_init, requires_grad=True)

    @property
    def qy0_std(self):
        return nn.functional.softplus(self.qy0_std_unconstrained)

    def f(self, t, y):  # Approximate posterior drift.
        if t.dim() == 0:
            t = torch.full_like(y[:,:1], fill_value=t)
        # Positional encoding in transformers for time-inhomogeneous posterior.
        return self.net(torch.cat((torch.sin(t), torch.cos(t), y), dim=-1))

    def g(self, t, y):  # Shared diffusion.
        """Define this using GP Kernel methods
        """
        # return self.sigma.repeat(y.size(0), 1)
        # drift matrix = sigma^2 * sqrt(2 / l) for Matern12 kernel (diffusion of BM is 2/l)
        return self.kernel.g(t, y)
        # return torch.ones(y.size(0), 1, device=self.device) * math.sqrt(2 / (self.lengthscale))

    def h(self, t, y):  # Prior drift.
        """Define this using GP Kernel methods
        """
        return self.kernel.f(t, y)
        # return -1/self.lengthscale * y

    def f_aug(self, t, y):  # Drift for augmented dynamics with logqp term.
        y = y[:, 0:self.kernel.state_dim()]
        f, pseudo_inv, h = self.f(t, y), self.kernel.diffusion_pseudoinverse(y), self.h(t, y)
        # u = _stable_division(f - h, g)

        # pseudo_inv: batch x state_dim
        # f,h: batch x state_dim
        u = pseudo_inv * (f-h)
        f_logqp = .5 * (u ** 2).sum(dim=1, keepdim=True)
        return torch.cat([f, f_logqp], dim=1)

    def g_aug(self, t, y):  # Diffusion for augmented dynamics with logqp term.
        y = y[:, 0:self.kernel.state_dim()]
        g = self.g(t, y)
        g_logqp = torch.zeros_like(y[:,:1])
        return torch.cat([g, g_logqp], dim=1)

    def forward(self, ts, batch_size, eps=None):
        eps = torch.randn(batch_size, 1).to(self.qy0_std) if eps is None else eps
        y0 = self.qy0_mean + eps * self.qy0_std
        qy0 = distributions.Normal(loc=self.qy0_mean, scale=self.qy0_std)
        py0 = distributions.Normal(loc=self.py0_mean, scale=self.kernel.stationary_covariance().sqrt())
        logqp0 = distributions.kl_divergence(qy0, py0).sum(dim=1)  # KL(t=0).

        aug_y0 = torch.cat([y0, torch.zeros(batch_size, 1).to(y0)], dim=1)
        aug_ys = sdeint_fn(
            sde=self,
            y0=aug_y0,
            ts=ts,
            method=args.method,
            dt=args.dt,
            adaptive=args.adaptive,
            rtol=args.rtol,
            atol=args.atol,
            names={'drift': 'f_aug', 'diffusion': 'g_aug'}
        )
        ys, logqp_path = aug_ys[:, :, :self.kernel.state_dim()], aug_ys[-1, :, -1]
        logqp = (logqp0 + logqp_path).mean(dim=0)  # KL(t=0) + KL(path).
        out = torch.einsum("ij, tbj -> tbi", self.kernel.measurement_model(), ys)
        return out, logqp

    ### skipping operations
    def forward_skip(self, batch_size, ts_segments, eps=None):

        # suppose we learn the stationary distribution
        eps = torch.randn(batch_size, 1).to(self.qy0_std) if eps is None else eps
        y0 = self.qy0_mean + eps * self.qy0_std
        qy0 = distributions.Normal(loc=self.qy0_mean, scale=self.qy0_std)
        py0 = distributions.Normal(loc=self.py0_mean, scale=self.kernel.stationary_covariance().sqrt())
        logqp0 = distributions.kl_divergence(qy0, py0).sum(dim=1)  # KL(t=0).
        aug_y0 = torch.cat([y0, torch.zeros(batch_size, 1).to(y0)], dim=1)
        
        for i, ts_segment in enumerate(ts_segments):
            # ts_segment = ts[(ts >= t_starts[i]) & (ts <= t_ends[i])]     
            # ts_segment[0] -= mixing_time
            # ts_segment[-1] += mixing_time
            # ts_segment = ts_segments[i]
            aug_ys = sdeint_fn(
                sde=self,
                y0=aug_y0,
                ts=ts_segment,
                method=args.method,
                dt=args.dt,
                adaptive=args.adaptive,
                rtol=args.rtol,
                atol=args.atol,
                names={'drift': 'f_aug', 'diffusion': 'g_aug'}
            )
            ys_segment, logqp_path = aug_ys[:, :, 0:self.kernel.state_dim()], aug_ys[-1, :, 1:]
            if i == 0:
                ys = ys_segment[1:-1]
            else:
                ys = torch.concat([ys, ys_segment[1:-1]], axis=0)
            eps = torch.randn(batch_size, 1).to(self.qy0_std)
            y0 = self.qy0_mean + eps * self.qy0_std
            aug_y0 = torch.cat([y0, logqp_path], dim=1)
        logqp_path = aug_ys[-1, :, -1]
        logqp = (logqp0 + logqp_path).mean(dim=0)  # KL(t=0) + KL(path).
        out = torch.einsum("ij, tbj -> tbi", self.kernel.measurement_model(), ys)
        return out, logqp

    def sample_p(self, ts, batch_size, eps=None, bm=None):
        eps = torch.randn(batch_size, 1).to(self.py0_mean) if eps is None else eps
        y0 = self.py0_mean + torch.einsum("ij, bj -> bi", self.kernel.stationary_covariance().sqrt(), eps)
        
        yt = sdeint_fn(self, y0, ts, bm=bm, method='srk', dt=args.dt, names={'drift': 'h'})
        out = torch.einsum("ij, tbj -> tbi", self.kernel.measurement_model(), yt)
        return out

    def sample_q(self, ts, batch_size, eps=None, bm=None):
        eps = torch.randn(batch_size, 1).to(self.qy0_mean) if eps is None else eps
        y0 = self.qy0_mean + eps * self.qy0_std
        yt = sdeint_fn(self, y0, ts, bm=bm, method='srk', dt=args.dt, names={"drift" : "f"})
        out = torch.einsum("ij, tbj -> tbi", self.kernel.measurement_model(), yt)
        return out



def make_irregular_gp_data():
    k = MaternKernel(nu=0.5)
    k.lengthscale = args.lengthscale
    ts = np.concatenate(
        [
            np.random.uniform(low=0.1, high=0.5, size=5),
            np.random.uniform(low=1.5, high=1.9, size=5)
        ],
        axis=0
    )
    ts_ = np.sort(ts)
    ts_ext_ = np.array([0.] + list(ts_) + [2.0])
    ts_vis_ = np.linspace(0., 2.0, 300)
    K = k(torch.Tensor(ts_)).evaluate().double() + (1e-6*torch.eye(ts_.shape[0])).double()
    L = torch.linalg.cholesky(K)
    ys_ = L.detach().cpu().numpy() @ np.random.normal(size=(K.shape[0], 1)).astype(np.float64)

    ts = torch.tensor(ts_).float()
    ts_ext = torch.tensor(ts_ext_).float()
    ts_vis = torch.tensor(ts_vis_).float()
    ys = torch.tensor(ys_).float().to(device)
    return Data(ts_, ts_ext_, ts_vis_, ts, ts_ext, ts_vis, ys, ys_)

def make_data():
    data_constructor = {
        'irregular_gp': make_irregular_gp_data
    }[args.data]
    return data_constructor()


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.MaternKernel(nu=0.5)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def main():
    # Dataset.
    ts_, ts_ext_, ts_vis_, ts, ts_ext, ts_vis, ys, ys_ = make_data()
    t_kl_plot = torch.Tensor(np.linspace(ts_ext[0], ts_ext[-1], 100))
    print(ts_)

    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    # get exact GP posterior
    model = ExactGPModel(torch.Tensor(ts_), torch.Tensor(ys_)[:,0], likelihood)
    model.likelihood.noise = args.scale**2
    model.covar_module.lengthscale = args.lengthscale
    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()

    # Test points are regularly spaced along [0,1]
    # Make predictions by feeding model through likelihood
    test_x = torch.Tensor(ts_vis_)
    f_preds = model(test_x)
    f_mean = f_preds.mean.detach().cpu().numpy()
    f_var = f_preds.variance.detach().cpu().numpy()
    lower = f_mean - 1.96*np.sqrt(f_var)
    upper = f_mean + 1.96*np.sqrt(f_var)
    del model

    # Plotting parameters.
    vis_batch_size = 1024
    ylims = (-3.5, 3.5)
    # alphas = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
    # percentiles = [0.999, 0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    alphas = [0.55]
    percentiles = [0.95]
    vis_idx = np.random.permutation(vis_batch_size)
    # From https://colorbrewer2.org/.
    if args.color == "blue":
        sample_colors = ('#8c96c6', '#8c6bb1', '#810f7c')
        fill_color = '#9ebcda'
        mean_color = '#4d004b'
        num_samples = len(sample_colors)
    else:
        sample_colors = ('#fc4e2a', '#e31a1c', '#bd0026')
        fill_color = '#fd8d3c'
        mean_color = '#800026'
        num_samples = len(sample_colors)


    # Model.
    model = LatentSDE(device=device, lengthscale=args.lengthscale, smoothness=args.smoothness).to(device)
    eps = torch.randn(size=(vis_batch_size, model.kernel.state_dim())).to(device)  # Fix seed for the random draws used in the plots.
    bm = torchsde.BrownianInterval(
        t0=ts_vis[0],
        t1=ts_vis[-1],
        size=(vis_batch_size, model.kernel.state_dim()),
        device=device,
        levy_area_approximation='space-time'
    )  # We need space-time Levy area to use the SRK solver
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=.999)
    kl_scheduler = LinearScheduler(iters=args.kl_anneal_iters)

    logpy_metric = EMAMetric()
    kl_metric = EMAMetric()
    loss_metric = EMAMetric()

    if args.show_prior:
        with torch.no_grad():
            zs = model.sample_p(ts=ts_vis, batch_size=vis_batch_size, eps=eps, bm=bm).squeeze()
            ts_vis_, zs_ = ts_vis.cpu().numpy(), zs.cpu().numpy()
            zs_ = np.sort(zs_, axis=1)

            img_dir = os.path.join(args.train_dir, 'prior.png')
            plt.subplot(frameon=False)
            for alpha, percentile in zip(alphas, percentiles):
                idx = int((1 - percentile) / 2. * vis_batch_size)
                zs_bot_ = zs_[:, idx]
                zs_top_ = zs_[:, -idx]
                plt.fill_between(ts_vis_, zs_bot_, zs_top_, alpha=alpha, color=fill_color)

            # `zorder` determines who's on top; the larger the more at the top.
            plt.scatter(ts_, ys_, marker='x', zorder=3, color='k', s=35)  # Data.
            plt.ylim(ylims)
            plt.xlabel('$t$')
            plt.ylabel('$Y_t$')
            plt.tight_layout()
            plt.savefig(img_dir, dpi=args.dpi)
            plt.close()
            logging.info(f'Saved prior figure at: {img_dir}')
    # mixing time
    mixing_time = 0.3
    # when to begin solving
    t_starts = torch.Tensor([0, 1.65337661])
    # when to stop solving
    t_ends = torch.Tensor([0.38607575, 1.8854651])
    ## suppose we start from y0 (data driven)
    ts_segments = []
    for i in range(t_starts.size()[0]):
        ts_segment = ts[(ts >= t_starts[i]) & (ts <= t_ends[i])]     
        ts_segment = torch.cat([ts_segment[:1] -mixing_time, ts_segment])
        ts_segment = torch.cat([ts_segment, ts_segment[-1:] + mixing_time])
        ts_segments.append(ts_segment)

    for global_step in tqdm.tqdm(range(args.train_iters)):
        # Plot and save.
        if global_step % args.pause_iters == 0:
            img_path = os.path.join(args.train_dir, f'global_step_{global_step}.png')

            with torch.no_grad():
                zs = model.sample_q(ts=ts_vis, batch_size=vis_batch_size, eps=eps, bm=bm).squeeze()
                samples = zs[:, vis_idx]
                ts_vis_, zs_, samples_ = ts_vis.cpu().numpy(), zs.cpu().numpy(), samples.cpu().numpy()
                zs_ = np.sort(zs_, axis=1)
                fig, (ax1, ax2) = plt.subplots(2, frameon=False, figsize=(20,10), sharex=True)

                if args.show_percentiles:
                    for alpha, percentile in zip(alphas, percentiles):
                        idx = int((1 - percentile) / 2. * vis_batch_size)
                        zs_bot_, zs_top_ = zs_[:, idx], zs_[:, -idx]
                        ax1.fill_between(ts_vis_, zs_bot_, zs_top_, alpha=alpha, color=fill_color)

                if args.show_mean:
                    ax1.plot(ts_vis_, zs_.mean(axis=1), color=mean_color)

                if args.show_samples:
                    for j in range(num_samples):
                        ax1.plot(ts_vis_, samples_[:, j], color=sample_colors[j], linewidth=1.0)

                # if args.show_arrows:
                #     num, dt = 10, 0.14
                #     t, y = torch.meshgrid(
                #         [torch.linspace(0.2, 1.8, num).to(device), torch.linspace(-1.5, 2.5, num).to(device)]
                #     )
                #     t, y = t.reshape(-1, 1), y.reshape(-1, 1)
                #     fty = model.f(t=t, y=y)
                #     # only show the drift in the first dimension i.e. not the derivatives
                #     fty = torch.einsum("ij, tbj -> tbi", model.kernel.measurement_model(), fty)
                #     fty = fty.reshape(num, num)
                #     dt = torch.zeros(num, num).fill_(dt).to(device)
                #     dy = fty * dt
                #     dt_, dy_, t_, y_ = dt.cpu().numpy(), dy.cpu().numpy(), t.cpu().numpy(), y.cpu().numpy()
                #     ax1.quiver(t_, y_, dt_, dy_, alpha=0.3, edgecolors='k', width=0.0015, scale=200)

                if args.hide_ticks:
                    ax1.xticks([], [])
                    ax1.yticks([], [])

                # plot the prior
                zs = model.sample_p(ts=ts_vis, batch_size=vis_batch_size, eps=eps, bm=bm).squeeze()
                ts_vis_, zs_ = ts_vis.cpu().numpy(), zs.cpu().numpy()
                zs_ = np.sort(zs_, axis=1)
                
                for alpha, percentile in zip(alphas, percentiles):
                    idx = int((1 - percentile) / 2. * vis_batch_size)
                    zs_bot_ = zs_[:, idx]
                    zs_top_ = zs_[:, -idx]
                    ax1.fill_between(ts_vis_, zs_bot_, zs_top_, alpha=alpha, color="C1")

                # plot exact GP 95% posterior 
                # ax1.fill_between(ts_vis_, lower, upper, alpha=0.3, color="green")

                # plot KL at each point
                eps_normal = torch.randn(args.batch_size, 1).to(model.qy0_std)
                y0 = model.qy0_mean + eps_normal * model.qy0_std
                aug_y0 = torch.cat([y0, torch.zeros(args.batch_size, 1).to(y0)], dim=1)
                aug_ys = sdeint_fn(
                    sde=model,
                    y0=aug_y0,
                    ts=t_kl_plot,
                    method=args.method,
                    dt=args.dt,
                    adaptive=args.adaptive,
                    rtol=args.rtol,
                    atol=args.atol,
                    bm=None,
                    names={'drift': 'f_aug', 'diffusion': 'g_aug'},
                )
                cum_kl = aug_ys[:,:,1].mean(1)
                kl_list = torch.diff(cum_kl)
                ax2.plot(t_kl_plot[1:].cpu().detach().numpy(), kl_list.cpu().detach().numpy())
                ax2.plot(ts_, np.zeros_like(ys_), "x", color="black")

            
                ax1.scatter(ts_, ys_, marker='x', zorder=3, color='k', s=35)  # Data.
                ax1.set_ylim(ylims)
                ax2.set_ylim(0, 1.2)
                ax1.set_xlabel('$t$')
                ax1.set_ylabel('$Y_t$')
                # plt.tight_layout()
                plt.savefig(img_path, dpi=args.dpi)
                plt.close()
                logging.info(f'Saved figure at: {img_path}')

                if args.save_ckpt:
                    torch.save(
                        {'model': model.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         'scheduler': scheduler.state_dict(),
                         'kl_scheduler': kl_scheduler},
                        os.path.join(ckpt_dir, f'global_step_{global_step}.ckpt')
                    )

        # Train.
        
        optimizer.zero_grad()
        if args.skip == True:
            # # mixing time
            # mixing_time = 0.3
            # # when to begin solving
            # t_starts = torch.Tensor([0.0000, 0.4661])
            # # when to stop solving
            # t_ends = torch.Tensor([1.5302, 2])
            zs, kl = model.forward_skip(batch_size=args.batch_size, ts_segments=ts_segments)
            zs = zs.squeeze()
        else:
            # print("\n\n", model.kernel.variance, model.qy0_std, model.qy0_mean)
            zs, kl = model(ts=ts_ext, batch_size=args.batch_size)
            zs = zs.squeeze()
            zs = zs[1:-1]  # Drop first and last which are only used to penalize out-of-data region and spread uncertainty.
        likelihood_constructor = {"laplace": distributions.Laplace, "normal": distributions.Normal}[args.likelihood]
        likelihood = likelihood_constructor(loc=zs, scale=args.scale)
        logpy = likelihood.log_prob(ys).sum(dim=0).mean(dim=0)
        loss = -logpy + kl * kl_scheduler.val
        loss.backward()

        optimizer.step()
        scheduler.step()
        kl_scheduler.step()

        logpy_metric.step(logpy)
        kl_metric.step(kl)
        loss_metric.step(loss)
        
        logging.info(
            f'global_step: {global_step}, '
            f'logpy: {logpy_metric.val:.3f}, '
            f'kl: {kl_metric.val:.3f}, '
            f'loss: {loss_metric.val:.3f}'
        )


if __name__ == '__main__':
    # The argparse format supports both `--boolean-argument` and `--boolean-argument True`.
    # Trick from https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse.
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-gpu', type=str2bool, default=False, const=True, nargs="?")
    parser.add_argument('--gpu', type=str, default="cpu")
    parser.add_argument('--debug', type=str2bool, default=False, const=True, nargs="?")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--train-dir', type=str, required=True)
    parser.add_argument('--save-ckpt', type=str2bool, default=False, const=True, nargs="?")

    parser.add_argument('--data', type=str, default='irregular_gp', choices=['segmented_cosine', 'irregular_sine', 'irregular_gp'])
    parser.add_argument('--kl-anneal-iters', type=int, default=100, help='Number of iterations for linear KL schedule.')
    parser.add_argument('--train-iters', type=int, default=5000, help='Number of iterations for training.')
    parser.add_argument('--pause-iters', type=int, default=50, help='Number of iterations before pausing.')
    parser.add_argument('--batch-size', type=int, default=512, help='Batch size for training.')
    parser.add_argument('--likelihood', type=str, choices=['normal', 'laplace'], default='laplace')
    parser.add_argument('--scale', type=float, default=0.01, help='Scale parameter of Normal and Laplace.')

    parser.add_argument('--adjoint', type=str2bool, default=False, const=True, nargs="?")
    parser.add_argument('--adaptive', type=str2bool, default=False, const=True, nargs="?")
    parser.add_argument('--skip', type=str2bool, default=False, const=True, nargs="?")
    parser.add_argument('--method', type=str, default='euler', choices=('euler', 'milstein', 'srk'),
                        help='Name of numerical solver.')
    parser.add_argument('--dt', type=float, default=1e-2)
    parser.add_argument('--rtol', type=float, default=1e-3)
    parser.add_argument('--atol', type=float, default=1e-3)
    parser.add_argument('--smoothness', type=float, default=0.5)
    parser.add_argument('--lengthscale', type=float, default=1)

    parser.add_argument('--show-prior', type=str2bool, default=True, const=True, nargs="?")
    parser.add_argument('--show-samples', type=str2bool, default=True, const=True, nargs="?")
    parser.add_argument('--show-percentiles', type=str2bool, default=True, const=True, nargs="?")
    parser.add_argument('--show-arrows', type=str2bool, default=True, const=True, nargs="?")
    parser.add_argument('--show-mean', type=str2bool, default=False, const=True, nargs="?")
    parser.add_argument('--hide-ticks', type=str2bool, default=False, const=True, nargs="?")
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--color', type=str, default='blue', choices=('blue', 'red'))
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_gpu else 'cpu')
    manual_seed(args.seed)

    if args.debug:
        logging.getLogger().setLevel(logging.INFO)

    ckpt_dir = os.path.join(args.train_dir, 'ckpts')
    os.makedirs(ckpt_dir, exist_ok=True)

    sdeint_fn = torchsde.sdeint_adjoint if args.adjoint else torchsde.sdeint

    main()
