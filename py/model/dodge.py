"""
TODO:
"""



import torch
import torch.nn as nn
import torch.autograd.forward_ad as fw



class DODGE:
    """
    `DODGE(model, optim = None).restart()`

    Implements [directional gradient descent](https://openreview.net/forum?id=5i7lJLuhTm): pick a direction [(a random unit vector by default)](https://arxiv.org/abs/2202.08587), receive a (forward-derivative) scalar feedback to correct it, and assign the gradients.

    It's a memory-less alternative to backpropagation, suitable for unrolling very long RNNs.

    Use this within `with torch.no_grad(): with torch.autograd.forward_ad.dual_level(): ...`.

    Parameters:
    - `model` can be a dict of learnable params and `torch.nn.Module`s, or just one module. `loss` must access learnable params through this dict, or modules.
    - `optim` can be a function from a list of params to a PyTorch optimizer, such as `lambda p: torch.optim.Adam(p, lr=1e-3)`. If not provided, do optimizer steps manually.

    Methods:
    - `.restart(direction = ...)`: starts optimization, by setting forward-derivatives to slices of `direction` (a good & default candidate for which is `torch.randn(dodge.sz)`; see the linked papers for other alternatives). Please note that DODGE is not magic, and can't learn inter-restart dependencies, because directions are different.
    - `.minimize(loss)`: updates `model`'s parameters. Note that having many calls to this without restarts is equivalent to just one call of this just before the restart, with the sum of losses — but updates params slightly faster.
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
        assert self.direction is not None
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



# TODO: Remove this, and Reptile.
def make_functional(model, params=...):
    """A polyfill of `functorch.make_functional`: `make_functional(model) → fn, params`, where `out = model(*a, **kw)` becomes `out = fn(params, *a, **kw)`. `model` can't mutate its params, but `params` will be treated as differentiable and get any set-inside-the-call `.grad` copied to outside unless within `with torch.no_grad(): ...`.

    Optionally pass the `params` arg (`model.parameters()` by default) to override the parameter-list.

    Convenient if advanced `params` manipulations are necessary (including soft-resetting, weighted averages, etc)."""
    if params is ...: params = model.parameters()
    orig_params = list(params)
    def fn(params, *a, **kw):
        assert len(orig_params) == len(params)
        old_params = []
        with torch.no_grad():
            for i in range(len(params)):
                old_params.append(orig_params[i].clone())
                orig_params[i].copy_(params[i])
                orig_params[i].requires_grad_(True)
        out = model(*a, **kw)
        with torch.no_grad():
            for i in range(len(params)):
                params[i].grad, orig_params[i].grad = orig_params[i].grad, None
                orig_params[i].detach_()
                params[i].copy_(orig_params[i])
                orig_params[i].copy_(old_params[i])
        return out
    return fn, orig_params



def SGD(lr):
    def do_SGD(params):
        with torch.no_grad():
            for p in params:
                if p.grad is not None:
                    p.data += lr * p.grad
                    p.grad = None
    return do_SGD
def Reptile(loss_fn, params, steps=3, inner_optim=SGD(.01), outer_optim=None):
    """
    `Reptile(loss_fn, params, steps=3, inner_optim=SGD(lr=.01), outer_optim=None)`

    Implements [Reptile](https://openai.com/blog/reptile/), which performs at least 2 gradient updates then updates initial parameters toward final ones.

    (A simple meta-learning algorithm, AKA an "RNN"-optimizer optimizer, which simply ignores the gradient from the inner update for speed.)

    Call the resulting func as `fn(*a, **kw)`, and `loss_fn(*a, **kw)→loss` (must not be `make_functional`) will be called `steps` times, and `params` will get updated in-place. No result.
    """
    def optimize_loss_fn(*a, **kw):
        result = loss_fn(*a, **kw)
        loss = result[0] if isinstance(result, tuple) else result
        loss.backward()
    optimize_loss_fn = make_functional(optimize_loss_fn, params=params)[0]
    def do_reptile(*a, **kw):
        params_now = [p.clone() for p in params]
        for _ in range(steps):
            optimize_loss_fn(params_now, *a, **kw)
            inner_optim(params_now)
        with torch.no_grad():
            for i in range(len(params)):
                params[i].grad = params[i] - params_now[i]
        if outer_optim is not None: outer_optim(params)
    return do_reptile



# TODO: Make our test use the (better) class-`DODGE`. (With an embedded optimizer.)
if __name__ == '__main__': # pragma: no cover
    # Every step, we input a bit, then ask the `net` to output the bit from 8 steps ago.
    #   Loss is 1 if no learning, else lower.
    import random
    class SkipConnection(nn.Module):
        def __init__(self, *fns): super().__init__();  self.fn = nn.Sequential(*fns)
        def forward(self, x): return self.fn(x)
    recall_len = 0 # 0 for the current bit, 1 for the 1-step-ago bit, etc. # TODO:
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
    loss = DODGE(loss, params) # TODO: …No longer the `loss` decorator, it's now a class…
    with fw.dual_level():
        with torch.no_grad():
            for iter in range(50000):
                l2, state = loss(state)
                state = fw_unnan(state)

                print(str(iter).rjust(5), 'L2', l2, state[0,0])
                opt.step();  opt.zero_grad(True)
    # TODO: …If we make `loss` do 2 steps inside itself, then we can learn 1-in-the-past dependencies, albeit very slowly (at 10k) — but 2-in-the-past remain out of reach (L2 is 2 at 50k) — which 3-steps-in-`loss` can learn, super-slowly (at 45k)…
    #   w e l l   w h a t   g o o d   i s   d o d g e   t h e n
    #   d o   w e   n e e d   r e p t i l e   a f t e r   a l l

    # TODO: Implement the `DODGE`-like `Reptile(loss_fn, model, steps=3, optim=...)` [decorator](https://openai.com/blog/reptile/), which remembers ALL initial parameters, does several steps of computing the loss and minimizing it, then resets the optimizer state and sets params to initial-nudged-to-final (or sets to initial and sets grad and does an optimizer step).
    # TODO: Update the DODGE in `test.py` to use the new class-interface.
    #   (TODO: …Also, should we *maybe* allow non-digital queries (controlled by a special name-part), for potential image generation?… If we don't, generating the correct bytes would likely be too expensive…)
    #     (…Maybe even allow `sn` instances to have arbitrary metadata attached, like a string "label cells with 'discrete', and querying them give you a discrete value"… (Even though this is kinda growing into "can attach arbitrary de/serialization code" for efficiency of digital transfers…))
    #       (…And, can GANs be measured not by a separate 0|1 NN, but by distance? …In fact, can't self-imitation be turned into GAN-like learning via adding DDPG that minimizes distance — if it's not done already? And by giving random noise as input, of course… Is this all we need for a unified analog/digital interface…)
    #     (…Maybe have not just the `'goal'` name-part, but make it 'analog'|'analog_goal'|'digital'|'digital_goal'…)
    #   (…Also, shouldn't reverse the zero-padded bit patterns, since that makes it *less* robust to changes in bits-per-cell, not *more*.)
    #   (…Also, `data = np.nan_to_num(data.clip(-1., 1.), copy=False)`.)
    #   (…Also, at least make a note to create trivial non-digital-action environments.)
    #   (…Also, probably make non-1-cell data/queries reserve a spot in the name, because otherwise, not only would we have to always perform a full query even if a result midway is `None`, but also, dynamically-sized queries are impossible because the end-of-sequence action may become `None`. …Meaning that we may indeed want to shuffle queries, and be able to do filling-in-of-`None`s.)

    # TODO: …Or can we repurpose `DODGE` to do very-long unrolls? …How would we know the targets during these unrolls though, particularly, distances? Not like we can retroactively relabel anything if learning online, so computing losses at goal-switching won't be enough… And very-long-unrolls-of-the-past can't scale to millions of timesteps anyway…
    #   (…Maybe self-imitation *could* learn good-for-remembering actions as if they were RNN state, saving us from having to do a perfect solution… Like an explicit notepad of facts and to-do tasks… Analogous to how humans use language to augment their learning, and aren't natively super-intelligences…)