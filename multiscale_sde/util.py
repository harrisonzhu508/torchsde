import torch 
import random 
import numpy as np 

def manual_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _stable_division(a, b, epsilon=1e-7):
    b = torch.where(b.abs().detach() > epsilon, b, torch.full_like(b, fill_value=epsilon) * b.sign())
    return a / b


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

def softplus_inverse(x):
    """PyTorch translation of tensorflow implementation
    """
    threshold = np.log(torch.finfo(x.dtype).eps) + 2.
    is_too_small = x < np.exp(threshold)
    is_too_large = x > -threshold
    too_small_value = torch.log(x)
    too_large_value = x
    # This `where` will ultimately be a NOP because we won't select this
    # codepath whenever we used the surrogate `ones_like`.
    x = torch.where(is_too_small | is_too_large, torch.ones((1), dtype=x.dtype), x)
    y = x + torch.log(-torch.expm1(-x))  # == log(expm1(x))
    return torch.where(is_too_small,
                too_small_value,
                torch.where(is_too_large, too_large_value, y))