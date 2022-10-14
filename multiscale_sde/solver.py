from torchsde._core.base_solver import BaseSDESolver
from torchsde._core.base_sde import BaseSDE
from torchsde._brownian import BaseBrownian
from torchsde.types import Scalar, Dict, Tensor, Tuple
from torchsde._core import interp
import torch


class SkipSDESolver(BaseSDESolver):
    def integrate(self, y0: Tensor, ts: Tensor, extra0: Tensor) -> Tuple[Tensor, Tensor]:
        """Integrate along trajectory by skipping over pre-defined regions

        Args:
            y0: Tensor of size (batch_size, d)
            ts: Tensor of size (T,).
            extra0: Any extra state for the solver.

        Returns:
            ys, where ys is a Tensor of size (T, batch_size, d).
            extra_solver_state, which is a tuple of Tensors of shape (T, ...), where ... is arbitrary and
                solver-dependent.
        """
        step_size = self.dt

        prev_t = curr_t = ts[0]
        prev_y = curr_y = y0
        curr_extra = extra0

        ys = [y0]
        for out_t in ts[1:]:
            while curr_t < out_t:
                next_t = min(curr_t + step_size, ts[-1])
                prev_t, prev_y = curr_t, curr_y
                curr_y, curr_extra = self.step(curr_t, next_t, curr_y, curr_extra)
                curr_t = next_t
            ys.append(interp.linear_interp(t0=prev_t, y0=prev_y, t1=curr_t, y1=curr_y, t=out_t))

        return torch.stack(ys, dim=0), curr_extra