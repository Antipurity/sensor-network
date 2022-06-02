"""
TODO:
"""



import torch
import torch.nn as nn
import torch.autograd.forward_ad as fw



def DODGE(loss_fn, model, direction_fn = lambda sz: torch.randn(sz)):
    """
    Implements [directional gradient descent](https://openreview.net/forum?id=5i7lJLuhTm): pick a direction [(a random unit vector by default)](https://arxiv.org/abs/2202.08587), compute a forward-derivative for it, receive a scalar feedback to correct it, and assign the gradients.

    Needs `loss_fn(…)→loss` and `model` for parameters. Do an optimizer step on those parameters yourself.

    `loss_fn` can also return a `(loss, ...)` tuple, if extra return info is desired.

    The `model` arg can be a dict of learnable params and `torch.nn.Module`s (or just one module). `loss` must access learnable params through this dict, or modules.
    """
    sz = 0
    if not isinstance(model, dict): model = {'model': model}
    responsibility = [(model, [])]
    for k,v in model.items():
        if isinstance(v, nn.Module):
            for mod in v.modules():
                responsibility.append((mod, []))
                for name, param in mod.named_parameters(recurse=False):
                    responsibility[-1][1].append((name, param))
                    sz += param.numel()
        else:
            responsibility[0][1].append((k, v))
            sz += v.numel()
    def loss_wrapper(*args, **kwargs):
        direction = direction_fn(sz)

        direction = (direction - direction.mean()) / (direction.std() + 1e-8)
        n = 0
        # Give the `direction` to `model`'s parameters.
        for mod, ps in responsibility:
            for name, param in ps:
                tangent = direction[n : n+param.numel()].reshape(*param.shape)
                if isinstance(mod, dict):
                    mod[name] = fw.make_dual(param, tangent)
                else:
                    delattr(mod, name)
                    setattr(mod, name, fw.make_dual(param, tangent))
                n += param.numel()
        result = loss_fn(*args, **kwargs)
        loss = result[0] if isinstance(result, tuple) else result
        _, loss_tangent = fw.unpack_dual(loss)
        assert loss_tangent is not None, "The computation doesn't use learnable params, so we can't optimize it"
        # Approximate the gradient by multiplying forward-gradient by the `loss_tangent` number.
        for mod, ps in responsibility:
            for name, param in ps:
                dual = mod[name] if isinstance(mod, dict) else getattr(mod, name)
                _, d = fw.unpack_dual(dual)
                grad = d * loss_tangent
                param.grad = grad if param.grad is None else param.grad + grad
                if isinstance(mod, dict):
                    mod[name] = param
                    # TODO: …Wait, if we handle RNN state like this, then should we also leave in the forward-gradient of params, and just re-assign it on the next iteration?… …But commenting this resetting out doesn't seem to help any…
                    #   TODO: …Eh, remove this resetting, both for efficiency and for preserving the "params got updated" semantics.
                else:
                    delattr(mod, name)
                    setattr(mod, name, param)
        return result
    return loss_wrapper



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



if __name__ == '__main__': # pragma: no cover
    # Every step, we input a bit, then ask the `net` to output the bit from 8 steps ago.
    #   Loss is 1 if no learning, else lower.
    import random
    class SkipConnection(nn.Module):
        def __init__(self, *fns): super().__init__();  self.fn = nn.Sequential(*fns)
        def forward(self, x): return self.fn(x)
    recall_len = 1 # 0 for the current bit, 1 for the 1-step-ago bit, etc. # TODO:
    n = 8
    p = .1
    lr = 1e-3
    net = nn.Sequential(
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
        loss = 0

        for _ in range(1):
            next_bit = 1 if random.randint(0,1)==1 else -1
            past_bits.append(next_bit)
            if len(past_bits) > recall_len+1: del past_bits[0]
            past_bit = past_bits[0]

            state = torch.cat((torch.full((state.shape[0], 1), next_bit), state[..., 1:]), -1)
            state = net(state)
            pred = state[..., 0] # Only the first number predicts.
            state = params['initial_state']*p + (1-p)*state # Soft-resetting.
            loss = loss + (pred - past_bit).square().sum()

        return loss, state
    loss = DODGE(loss, params)
    with fw.dual_level():
        with torch.no_grad():
            for iter in range(50000):
                l2, state = loss(state)

                print(str(iter).rjust(5), 'L2', l2, fw.unpack_dual(state)[1].abs().sum()) # TODO:
                opt.step();  opt.zero_grad(True)
    # TODO: (…If this soft-resetting fails, should we try episodes? …And then, should we try episodes-in-`loss`, which should definitely be mathematically grounded since direction is the same?…)
    #   …Same-bit prediction now works… But…
    #   …Loss settles to a bit below (inconsistent) 1 in the current configuration, for predicting the 1-step-past bit… Something is definitely wrong…
    #   TODO: Why does loss eventually go to `nan` when we don't converge (`recall_len>0`)?…
    #     TODO: …Why does it kinda look like it's starting to learn before that happens?… …Though on that one time where it didn't `nan`, L2 was only able to reach unstable .5 (then revert to stable 1 at 40k), nothing like .001 of more-steps-in-`loss` — hardly suitable for optimization…
    # TODO: …If we make `loss` do 2 steps inside itself, then we can learn 1-in-the-past dependencies, albeit very slowly (at 10k) — but 2-in-the-past remain out of reach (L2 is 2 at 50k) — which 3-steps-in-`loss` can learn, super-slowly (at 45k)…
    #   w e l l   w h a t   g o o d   i s   d o d g e   t h e n
    #   d o   w e   n e e d   r e p t i l e   a f t e r   a l l
    #   TODO: Re-read the DODGE paper to confirm this understanding.