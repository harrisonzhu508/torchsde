## Define an SDE kernel with 
## methods drift: f()
## diffusion: g() [make sure the Brownian diffusion is included too]
## 
import torch
import math
from torch import nn
from multiscale_sde.util import softplus_inverse

class MaternSDEKernel(nn.Module):
    """Based on bayesnewton's kernel implementation
    """
    def __init__(self, smoothness=0.5, lengthscale=1, variance=1, fix_lengthscale=False, fix_variance=False, device="cpu") -> None:
        super().__init__()
        self.smoothness = smoothness 
        assert self.smoothness in [0.5, 1.5], "GP Kernel not implemented"
        
        self.fix_lengthscale = fix_lengthscale
        self.fix_variance = fix_variance
        
        if self.fix_lengthscale is True:
            self.lengthscale_unconstrained = nn.Parameter(softplus_inverse(torch.Tensor([lengthscale])), requires_grad=False)
        else:
            self.lengthscale_unconstrained = nn.Parameter(softplus_inverse(torch.Tensor([lengthscale])), requires_grad=True)
        
        if self.fix_variance is True:
            self.variance_unconstrained = nn.Parameter(softplus_inverse(torch.Tensor([[variance]])), requires_grad=False)
        else:
            self.variance_unconstrained = nn.Parameter(softplus_inverse(torch.Tensor([[variance]])), requires_grad=True)
        
        self.register_buffer("lengthscale", nn.functional.softplus(self.lengthscale_unconstrained))
        self.register_buffer("variance", nn.functional.softplus(self.variance_unconstrained))
        self.device = device
    # @property
    # def lengthscale(self):
    #     return torch.softplus(self.lengthscale_unconstrained)

    # @property
    # def variance(self):
    #     return torch.softplus(self.variance_unconstrained)

    def f(self, t, y):
        if self.smoothness == 0.5:
            return -1/self.lengthscale * y
        elif self.smoothness == 1.5:
            lam = 3.0 ** 0.5 / self.lengthscale
            return torch.Tensor([[0.0, 1.0], [-(lam**2), -2 * lam]])

    def g(self, t, y):
        if self.smoothness == 0.5:
            Qc = 2 * self.variance.reshape(-1) / self.lengthscale
            return torch.ones(y.size(0), 1, device=self.device) * Qc.sqrt()
        elif self.smoothness == 1.5:
            Qc = torch.Tensor([[12.0 * 3.0**0.5 / self.lengthscale**3.0 * self.variance.reshape(-1)]])
            return Qc * torch.Tensor([[0.0], [1.0]])

    def stationary_covariance(self):
        if self.smoothness == 0.5:
            return self.variance
        elif self.smoothness == 1.5:
            Pinf = torch.Tensor(
                [[self.variance.reshape(-1), 0.0], [0.0, 3.0 * self.variance.reshape(-1) / self.lengthscale**2.0]]
            )
            return Pinf

    def measurement_model(self):
        if self.smoothness == 0.5:
            H = torch.Tensor([[1.0]])
        elif self.smoothness == 1.5:
            H = torch.Tensor([[1.0, 0.0]])
        return H
