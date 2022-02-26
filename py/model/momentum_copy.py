"""
The slower they change, the easier they are to learn.
"""



import torch
import torch.nn as nn



class MomentumCopy(nn.Module):
    """
    `MomentumCopy(module, momentum = .999)`

    A version of the `module` that is updated slowly, by making its weights the exponentially-moving-average of `module`'s weights. This can help stabilize training, when used as a moving target for prediction.

    Remember to `.update()`, either at call, or near the optimizer's step.
    """
    def __init__(self, f, momentum = .999):
        import copy
        super().__init__()
        self.f = f
        self.g = copy.deepcopy(f)
        self.fp, self.gp = list(self.f.parameters()), list(self.g.parameters())
        for p in self.gp: p.requires_grad_(False)
        self.momentum = momentum
    def forward(self, x):
        return self.g(x)
    def update(self):
        with torch.no_grad():
            for i in range(len(self.fp)):
                self.gp[i][:] = self.gp[i] * self.momentum + (1-self.momentum) * self.fp[i]