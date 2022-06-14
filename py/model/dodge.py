"""
Implements directional gradient descent, in `DODGE`, along with a few useful things for it.

May be useful for training RNNs, because unlike truncated BPTT, this method neither consumes a lot of memory nor stalls periodically to backprop through it, though this directional-descent is quite a bit less exact.
"""



import torch
import torch.nn as nn
import torch.autograd.forward_ad as fw



class DODGE:
    """
    `DODGE(model, optim = None).restart()`

    Implements [directional gradient descent](https://openreview.net/forum?id=5i7lJLuhTm): pick a direction [(a random unit vector by default)](https://arxiv.org/abs/2202.08587), receive a (forward-derivative) scalar feedback to correct it, and assign the gradients.

    It's a memory-less alternative to backpropagation, suitable for unrolling very long RNNs.

    Use this within `with torch.no_grad(): with torch.autograd.forward_ad.dual_level(): ...`. Probably in a very long loop, which always does `.minimize(...)` and sometimes does `.restart(...)`.

    Parameters:
    - `model` can be a dict of learnable params and `torch.nn.Module`s, or just one module. `loss` must access learnable params through this dict, or modules.
    - `optim` can be a function from a list of params to a PyTorch optimizer, such as `lambda p: torch.optim.Adam(p, lr=1e-3)`. If not provided, do optimizer steps manually.

    Methods:
    - `.restart(direction = ...)`: starts optimization, by setting forward-derivatives to slices of `direction` (a good & default candidate for which is `torch.randn(dodge.sz)`; see the linked papers for other alternatives). Please note that DODGE is not magic, and can't learn inter-restart dependencies, because directions are different.
    - `.minimize(loss)`: updates `model`'s parameters, and steps `optim` if provided.
        - If RNN state is reset with `detach` after this, then having many calls to this without restarts is equivalent to just one call of this just before the restart, with the sum of losses — but updates params slightly faster.
    - `.stop()`: stops optimization, unless restarted again. No need to stop right before a restart.
    """
    def __init__(self, model, optim=None):
        if not isinstance(model, dict): model = { 'model': model }
        sz, responsibility = 0, [(model, [])]
        params = []
        for k,v in model.items():
            if isinstance(v, nn.Module):
                for mod in v.modules():
                    responsibility.append((mod, []))
                    for name, param in mod.named_parameters(recurse=False):
                        responsibility[-1][1].append((name, param))
                        params.append(param)
                        sz += param.numel()
            else:
                responsibility[0][1].append((k, v))
                params.append(v)
                sz += v.numel()
        self.sz, self.responsibility = sz, responsibility
        if callable(optim): optim = optim(params)
        self.optim = optim
    def restart(self, direction=...):
        # Changes the direction, and sets it as tangents of `model`'s parameters.
        if direction is ...:
            direction = torch.randn(self.sz)
        assert len(direction.shape) == 1 and direction.shape[0] == self.sz
        n = 0
        for mod, ps in self.responsibility:
            for name, param in ps:
                tangent = direction[n : n+param.numel()].reshape(*param.shape)
                if isinstance(mod, dict):
                    mod[name] = fw.make_dual(param, tangent)
                else:
                    delattr(mod, name)
                    setattr(mod, name, fw.make_dual(param, tangent))
                n += param.numel()
        return self
    def minimize(self, loss):
        # Given the scalar feedback `loss_tangent`, finalizes the pre-decided direction's contribution.
        _, loss_tangent = fw.unpack_dual(fw_unnan(loss))
        assert loss_tangent is not None, "The computation doesn't use learnable params, so we can't optimize it"
        with torch.no_grad():
            for mod, ps in self.responsibility:
                for name, param in ps:
                    dual = mod[name] if isinstance(mod, dict) else getattr(mod, name)
                    _, d = fw.unpack_dual(dual)
                    assert d is not None, "DODGE optimization is not enabled; `.restart()` it"
                    grad = d * loss_tangent
                    param.grad = grad if param.grad is None else param.grad + grad
        if self.optim:
            self.optim.step()
            self.optim.zero_grad(True)
    def stop(self):
        # Removes forward-gradients from NN params.
        for mod, ps in self.responsibility:
            for name, param in ps:
                if isinstance(mod, dict):
                    mod[name] = param
                else:
                    delattr(mod, name)
                    setattr(mod, name, param)



class LayerNorm(nn.Module):
    """A re-implementation of PyTorch's layer-norm, done so that forward-diff can work with slightly older PyTorch versions."""
    def __init__(self, sz):
        super().__init__()
        self.mult = nn.parameter.Parameter(torch.ones(sz), requires_grad=True)
        self.add = nn.parameter.Parameter(torch.zeros(sz), requires_grad=True)
    def forward(self, x):
        x = x - x.mean()
        y = x / (x.square().sum().sqrt() + 1e-5)
        return y * self.mult + self.add
class Softmax(nn.Module):
    """A re-implementation of PyTorch's softmax, done so that forward-diff can work with slightly older PyTorch versions."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        if x.numel() > 0:
            x = x - x.max()
        x = x.exp()
        return x / (x.sum(self.dim, keepdim=True) + 1e-5)
class ReLU(nn.Module):
    """A re-implementation of PyTorch's relu, done so that forward-diff can work with slightly older PyTorch versions."""
    def __init__(self): super().__init__()
    def forward(self, x):
        return (detach(x) > 0.).float() * x
def detach(x):
    return fw.unpack_dual(x)[0].detach()
def fw_unnan(x):
    """Removes `nan`s from `x`'s tangents (forward-gradient)."""
    primal, tangent = fw.unpack_dual(x)
    return fw.make_dual(primal, torch.nan_to_num(tangent)) if tangent is not None else primal



if __name__ == '__main__': # pragma: no cover
    # Every step, we input a bit, then ask the `net` to output the bit from 8 steps ago.
    #   Loss is 1+ if no learning, else lower.
    import random
    class SkipConnection(nn.Module):
        def __init__(self, *fns): super().__init__();  self.fn = nn.Sequential(*fns)
        def forward(self, x): return self.fn(x)
    recall_len = 1 # 0 for the current bit, 1 for the 1-step-ago bit, etc.
    truncation_len = 2 # Has to be at least one more than `recall_len` to learn.
    n = 8
    p = .1
    lr = 1e-3
    net = nn.Sequential(
        SkipConnection(nn.Linear(n, n)),
        SkipConnection(ReLU(), LayerNorm(n), nn.Linear(n, n)),
        SkipConnection(ReLU(), LayerNorm(n), nn.Linear(n, n)),
    )
    params = {
        'initial_state': torch.randn(1, n, requires_grad=True),
        'net': net,
    }
    opt = torch.optim.Adam([params['initial_state'], *net.parameters()], lr=lr)
    state = torch.randn(1, n)
    past_bits = []
    def loss(state):
        # Env.
        next_bit = 1 if random.randint(0,1)==1 else -1
        past_bits.append(next_bit)
        if len(past_bits) > recall_len+1: del past_bits[0]
        past_bit = past_bits[0]

        # RNN.
        state = torch.cat((torch.full((state.shape[0], 1), next_bit), state[..., 1:]), -1)
        state = net(state)
        pred = state[..., 0] # Only the first number predicts.
        state = params['initial_state']*p + (1-p)*state # Soft-resetting.
        loss = (pred - past_bit).square().sum()

        return loss, state
    with fw.dual_level():
        with torch.no_grad():
            dodge = DODGE(params).restart()
            for iter in range(50000):
                l2, state = loss(state)
                state = fw_unnan(state)

                dodge.minimize(l2)
                if iter % truncation_len == 0:
                    dodge.restart()

                print(str(iter).rjust(5), 'L2', l2, state[0,0])
                opt.step();  opt.zero_grad(True)