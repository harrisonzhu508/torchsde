import torchsde 
import torch
from torch import distributions, nn
import math 
from multiscale_sde.util import _stable_division


class GPSDE(torchsde.SDEIto):

    def __init__(self, sdeint_fn, method="euler", adaptive="False", rtol=1e-3, atol=1e-3, dt=1e-2, theta=1.0, mu=0.0, sigma=1, lengthscale=0.3, device="cpu"):
        super().__init__(noise_type="diagonal")
        self.sdeint_fn = sdeint_fn
        self.dt = dt
        self.method = method
        self.adaptive = adaptive
        self.rtol = rtol
        self.atol = atol
        logvar = math.log(sigma ** 2 / (2. * theta))
        # logvar = 1
        std = 1

        # Prior drift.
        self.register_buffer("theta", torch.tensor([[theta]]))
        self.register_buffer("mu", torch.tensor([[mu]]))
        self.register_buffer("sigma", torch.tensor([[sigma]]))

        # p(y0).
        self.register_buffer("py0_mean", torch.tensor([[mu]]))
        # self.register_buffer("py0_logvar", torch.tensor([[logvar]]))
        self.register_buffer("py0_std", torch.tensor([[std]]))

        # Approximate posterior drift: Takes in 2 positional encodings and the state.
        self.net = nn.Sequential(
            nn.Linear(3, 200),
            nn.Tanh(),
            nn.Linear(200, 200),
            nn.Tanh(),
            nn.Linear(200, 1)
        )
        # Initialization trick from Glow.
        self.net[-1].weight.data.fill_(0.)
        self.net[-1].bias.data.fill_(0.)

        # q(y0).
        self.qy0_mean = nn.Parameter(torch.tensor([[mu]]), requires_grad=True)
        self.qy0_logvar = nn.Parameter(torch.tensor([[logvar]]), requires_grad=True)
        self.lengthscale = lengthscale
        self.device = device

    def f(self, t, y):  # Approximate posterior drift.
        if t.dim() == 0:
            t = torch.full_like(y, fill_value=t)
        # Positional encoding in transformers for time-inhomogeneous posterior.
        return self.net(torch.cat((torch.sin(t), torch.cos(t), y), dim=-1))

    def g(self, t, y):  # Shared diffusion.
        # return self.sigma.repeat(y.size(0), 1)
        # drift matrix = sigma^2 * sqrt(2 / l) for Matern12 kernel (diffusion of BM is 2/l)
        return torch.ones(y.size(0), 1, device=self.device) * math.sqrt(2 / self.lengthscale)

    def h(self, t, y):  # Prior drift.
        return -1/self.lengthscale * y

    def f_aug(self, t, y):  # Drift for augmented dynamics with logqp term.
        y = y[:, 0:1]
        f, g, h = self.f(t, y), self.g(t, y), self.h(t, y)
        u = _stable_division(f - h, g)
        f_logqp = .5 * (u ** 2).sum(dim=1, keepdim=True)
        return torch.cat([f, f_logqp], dim=1)

    def g_aug(self, t, y):  # Diffusion for augmented dynamics with logqp term.
        y = y[:, 0:1]
        g = self.g(t, y)
        g_logqp = torch.zeros_like(y)
        return torch.cat([g, g_logqp], dim=1)

    def forward(self, ts, batch_size, eps=None):
        eps = torch.randn(batch_size, 1).to(self.qy0_std) if eps is None else eps
        y0 = self.qy0_mean + eps * self.qy0_std
        qy0 = distributions.Normal(loc=self.qy0_mean, scale=self.qy0_std)
        py0 = distributions.Normal(loc=self.py0_mean, scale=self.py0_std)
        logqp0 = distributions.kl_divergence(qy0, py0).sum(dim=1)  # KL(t=0).

        aug_y0 = torch.cat([y0, torch.zeros(batch_size, 1).to(y0)], dim=1)
        aug_ys = self.sdeint_fn(
            sde=self,
            y0=aug_y0,
            ts=ts,
            method=self.method,
            dt=self.dt,
            adaptive=self.adaptive,
            rtol=self.rtol,
            atol=self.atol,
            names={'drift': 'f_aug', 'diffusion': 'g_aug'}
        )
        ys, logqp_path = aug_ys[:, :, 0:1], aug_ys[-1, :, 1]
        logqp = (logqp0 + logqp_path).mean(dim=0)  # KL(t=0) + KL(path).
        return ys, logqp

    def sample_p(self, ts, batch_size, eps=None, bm=None):
        eps = torch.randn(batch_size, 1).to(self.py0_mean) if eps is None else eps
        y0 = self.py0_mean + eps * self.py0_std
        return self.sdeint_fn(self, y0, ts, bm=bm, method='srk', dt=self.dt, names={'drift': 'h'})

    def sample_q(self, ts, batch_size, eps=None, bm=None):
        eps = torch.randn(batch_size, 1).to(self.qy0_mean) if eps is None else eps
        y0 = self.qy0_mean + eps * self.qy0_std
        return self.sdeint_fn(self, y0, ts, bm=bm, method='srk', dt=self.dt)

    ### skipping operations
    def forward_skip(self, batch_size, ts_segments, eps=None):

        # suppose we learn the stationary distribution
        eps = torch.randn(batch_size, 1).to(self.qy0_std) if eps is None else eps
        y0 = self.qy0_mean + eps * self.qy0_std
        qy0 = distributions.Normal(loc=self.qy0_mean, scale=self.qy0_std)
        py0 = distributions.Normal(loc=self.py0_mean, scale=self.py0_std)
        logqp0 = distributions.kl_divergence(qy0, py0).sum(dim=1)  # KL(t=0).
        aug_y0 = torch.cat([y0, torch.zeros(batch_size, 1).to(y0)], dim=1)
        
        for i, ts_segment in enumerate(ts_segments):
            # ts_segment = ts[(ts >= t_starts[i]) & (ts <= t_ends[i])]     
            # ts_segment[0] -= mixing_time
            # ts_segment[-1] += mixing_time
            # ts_segment = ts_segments[i]
            aug_ys = self.sdeint_fn(
                sde=self,
                y0=aug_y0,
                ts=ts_segment,
                method=self.method,
                dt=self.dt,
                adaptive=self.adaptive,
                rtol=self.rtol,
                atol=self.atol,
                names={'drift': 'f_aug', 'diffusion': 'g_aug'}
            )
            ys_segment, logqp_path = aug_ys[:, :, 0:1], aug_ys[-1, :, :1]
            if i == 0:
                ys = ys_segment[1:-1]
            else:
                ys = torch.concat([ys, ys_segment[1:-1]], axis=0)
            eps = torch.randn(batch_size, 1).to(self.qy0_std)
            y0 = self.qy0_mean + eps * self.qy0_std
            aug_y0 = torch.cat([y0, logqp_path], dim=1)

        logqp = (logqp0 + logqp_path).mean(dim=0)  # KL(t=0) + KL(path).
        return ys, logqp

    # @property
    # def py0_std(self):
    #     return torch.exp(.5 * self.py0_logvar)

    @property
    def qy0_std(self):
        return torch.exp(.5 * self.qy0_logvar)