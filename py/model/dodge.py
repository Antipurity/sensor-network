"""
TODO:
"""



from importlib.metadata import requires
import torch
import torch.autograd.forward_ad as fw



def DODGE(loss_fn, model, direction_fn = lambda sz: torch.randn(sz)):
    """
    Implements [directional gradient descent](https://openreview.net/forum?id=5i7lJLuhTm): pick a direction [(a random unit vector by default)](https://arxiv.org/abs/2202.08587), compute a forward-derivative for it, receive a scalar feedback to correct it, and assign the gradients.

    Needs `loss_fn(…)→loss` and `model` for parameters. Do an optimizer step on those parameters yourself.
    """
    responsibility = []
    sz = 0
    for mod in model.modules():
        responsibility.append((mod, []))
        for name, param in mod.named_parameters(recurse=False):
            responsibility[-1][1].append((name, param))
            sz += param.numel()
    def loss_wrapper(*args, **kwargs):
        direction = direction_fn(sz)

        direction = (direction - direction.mean()) / (direction.std() + 1e-8)
        n = 0
        # Give the `direction` to `model`'s parameters.
        for mod, ps in responsibility:
            for name, param in ps:
                assert getattr(mod, name) is param
                tangent = direction[n : n+param.numel()].reshape(*param.shape)
                delattr(mod, name)
                setattr(mod, name, fw.make_dual(param, tangent))
                n += param.numel()
        loss = loss_fn(*args, **kwargs)
        _, loss_tangent = fw.unpack_dual(loss)
        # Approximate the gradient by multiplying forward-gradient by the `loss_tangent` number.
        for mod, ps in responsibility:
            for name, param in ps:
                _, d = fw.unpack_dual(getattr(mod, name))
                grad = d * loss_tangent
                param.grad = grad if param.grad is None else param.grad + grad
                delattr(mod, name)
                setattr(mod, name, param)
    return loss_wrapper



if __name__ == '__main__': # pragma: no cover
    # Every step, we input a bit, then ask the `net` to output the bit from 8 steps ago.
    import torch.nn as nn
    import random
    class SkipConnection(nn.Module):
        def __init__(self, *fns): super().__init__();  self.fn = nn.Sequential(*fns)
        def forward(self, x): return self.fn(x)
    recall_len = 8
    n = 32
    p = .001
    net = nn.Sequential(
        SkipConnection(nn.ReLU(), nn.LayerNorm(n), nn.Linear(32, 32)),
        SkipConnection(nn.ReLU(), nn.LayerNorm(n), nn.Linear(32, 32)),
    )
    opt = torch.optim.Adam(net.parameters(), le=1e-3)
    initial_state = torch.randn(1, n, requires_grad=True)
    state = torch.randn(1, n)
    past_bits = [0]
    def loss(pred, target):
        return (pred - target).square().sum()
    loss = DODGE(loss, net)
    with fw.dual_level():
        for _ in range(50000):
            bit = random.randint(0,1)

            state = torch.cat((torch.full((state.shape[0], 1), bit), state[..., 1:]), -1)
            state = net(state)
            state = initial_state*p + (1-p)*state # Soft-resetting.

            print('L2', loss(state[0], past_bits[0]))
            opt.step();  opt.zero_grad(True)

            past_bits.append(bit)
            if len(past_bits) > recall_len: del past_bits[0]