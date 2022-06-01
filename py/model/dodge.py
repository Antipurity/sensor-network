"""
TODO:
"""



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
    0
    # TODO: Test: a 32→32 simple RNN with a skip connection and soft-resetting of state; every step, we input a bit, and ask to output the bit from 8 steps ago. No backprop, only forward-prop.