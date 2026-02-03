from __future__ import annotations

from typing import List, Tuple, Dict, Any
import numpy as np
import torch
from scipy.optimize import least_squares


class FairGrad:

    def __init__(self, n_tasks: int, device: torch.device, alpha: float = 1.0, max_norm: float = 1.0, eps: float = 1e-8):
        self.n_tasks = int(n_tasks)
        self.device = device
        self.alpha = float(alpha)
        self.max_norm = float(max_norm)
        self.eps = float(eps)

    @staticmethod
    def _grad2vec(shared_params: List[torch.nn.Parameter], grads: torch.Tensor, grad_dims: List[int], task: int):
        grads[:, task].fill_(0.0)
        cnt = 0
        for p in shared_params:
            g = p.grad
            if g is not None:
                g = g.detach().view(-1)
                beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                end = sum(grad_dims[: (cnt + 1)])
                grads[beg:end, task].copy_(g)
            cnt += 1

    def _overwrite_grad(self, shared_params: List[torch.nn.Parameter], newgrad: torch.Tensor, grad_dims: List[int]):
        newgrad = newgrad * self.n_tasks
        cnt = 0
        for p in shared_params:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            end = sum(grad_dims[: (cnt + 1)])
            p.grad = newgrad[beg:end].contiguous().view_as(p.data).detach().clone()
            cnt += 1

    def _solve_weights(self, GG: np.ndarray) -> np.ndarray:
        K = GG.shape[0]
        x0 = np.ones(K, dtype=np.float64) / K
        alpha = float(self.alpha)
        eps = float(self.eps)

        def objfn(x):
            x = np.maximum(x, eps)
            return GG.dot(x) - np.power(1.0 / x, 1.0 / alpha)

        res = least_squares(objfn, x0, bounds=(0.0, np.inf))
        x = np.maximum(res.x, eps)
        return x

    def backward(
        self,
        losses: torch.Tensor,
        shared_parameters: List[torch.nn.Parameter],
        task_specific_parameters=None,
        last_shared_parameters=None,
        representation=None,
        **kwargs,
    ) -> Tuple[None, Dict[str, Any]]:
        assert losses.dim() == 1 and losses.numel() == self.n_tasks, \
            f"FairGrad expects losses shape ({self.n_tasks},), got {tuple(losses.shape)}"

        shared_params = list(shared_parameters)
        grad_dims = [p.data.numel() for p in shared_params]
        D = int(sum(grad_dims))

        with torch.no_grad():
            active_mask = (losses.detach() > 0).to(torch.bool)

        active_idx = torch.nonzero(active_mask, as_tuple=False).view(-1)

        if active_idx.numel() <= 1:
            losses.sum().backward()
            if self.max_norm > 0:
                torch.nn.utils.clip_grad_norm_(shared_params, self.max_norm)

            w_full = np.zeros(self.n_tasks, dtype=np.float64)
            if active_idx.numel() == 1:
                w_full[int(active_idx.item())] = 1.0
            return None, {"weights": w_full, "GTG": None, "alpha": self.alpha}

        K = int(active_idx.numel())
        grads = torch.zeros(D, K, device=self.device, dtype=torch.float32)

        for j, ti in enumerate(active_idx.tolist()):
            retain = (j < K - 1)
            losses[ti].backward(retain_graph=retain)
            self._grad2vec(shared_params, grads, grad_dims, j)
            for p in shared_params:
                p.grad = None

        GG_t = grads.t().mm(grads).detach().cpu().numpy()
        w_active = self._solve_weights(GG_t)

        ww = torch.tensor(w_active, device=grads.device, dtype=grads.dtype)
        g = (grads * ww.view(1, -1)).sum(dim=1)  # [D]
        self._overwrite_grad(shared_params, g, grad_dims)

        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(shared_params, self.max_norm)

        w_full = np.zeros(self.n_tasks, dtype=np.float64)
        for j, ti in enumerate(active_idx.tolist()):
            w_full[int(ti)] = float(w_active[j])

        return None, {"weights": w_full, "GTG": GG_t, "alpha": self.alpha}
