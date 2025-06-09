import torch
from torch.optim.optimizer import Optimizer

class Lion(Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad

                # Get or init momentum
                state = self.state[p]
                if 'momentum' not in state:
                    state['momentum'] = torch.zeros_like(p)
                m = state['momentum']

                # Update
                update = (1 - beta1) * grad + beta1 * m
                update = update.sign()

                # Weight decay
                if weight_decay != 0:
                    update += weight_decay * p

                # Apply update
                # p = p - lr * update
                p.add_(update, alpha=-lr)


                # Momentum update
                # m = beta2 * m + (1 - beta2) * grad
                m.mul_(beta2).add_(grad, alpha=1 - beta2)
