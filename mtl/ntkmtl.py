import numpy as np
import torch


class NTKMTL:

    def __init__(self, n_tasks: int, device: torch.device,
                 max_norm: float = 1.0, ntk_exp: float = 0.5):
        self.n_tasks = n_tasks
        self.device = device
        self.max_norm = max_norm
        self.ntk_exp = ntk_exp

    @staticmethod
    def _collect_grad_dims(params):
        return [p.data.numel() for p in params]

    @staticmethod
    def _grad2vec(shared_params, grads_tensor, grad_dims, task_idx):
        grads_tensor[:, task_idx].fill_(0.0)
        cnt = 0
        for param in shared_params:
            grad = param.grad
            if grad is not None:
                g_flat = grad.data.detach().view(-1)
                beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                en = sum(grad_dims[:cnt + 1])
                grads_tensor[beg:en, task_idx].copy_(g_flat)
            cnt += 1

    @staticmethod
    def _overwrite_grad(shared_params, new_grad, grad_dims, n_tasks):

        new_grad = new_grad * n_tasks
        cnt = 0
        for param in shared_params:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = new_grad[beg:en].contiguous().view_as(param.data)
            param.grad = this_grad.clone()
            cnt += 1

    def _ntkgrad(self, grads):

        GTG = grads.t().mm(grads)
        ntk_trace = torch.trace(GTG)
        diag = GTG.diagonal()

        w = ntk_trace / (diag + 1e-8)
        w = torch.pow(w, self.ntk_exp)

        g = grads @ w
        return g, w.detach().cpu().numpy()

    def backward(
        self,
        losses: torch.Tensor,
        shared_parameters,
        task_specific_parameters=None,
        last_shared_parameters=None,
        representation=None,
    ):
        assert losses.dim() == 1 and losses.numel() == self.n_tasks, \
            f"NTKMTL expects losses of shape (n_tasks,), got {losses.shape}"

        shared_params = list(shared_parameters)
        grad_dims = self._collect_grad_dims(shared_params)
        grads = torch.zeros(sum(grad_dims), self.n_tasks, device=self.device)

        for i in range(self.n_tasks):
            if i < self.n_tasks - 1:
                losses[i].backward(retain_graph=True)
            else:
                losses[i].backward()

            self._grad2vec(shared_params, grads, grad_dims, i)

            for p in shared_params:
                p.grad = None

        g, w_cpu = self._ntkgrad(grads)

        self._overwrite_grad(shared_params, g, grad_dims, self.n_tasks)

        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(shared_params, self.max_norm)

        return w_cpu
