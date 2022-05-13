"""
TODO:
"""



import torch
import torch.nn as nn



def cat(xs, dim=-1): return xs[0] if len(xs) == 1 else torch.cat(xs, dim)



class SkipConnection(nn.Module):
    """TODO: (At least link ResNet as evidence that learning is indeed much easier with a linear path.)"""
    def __init__(self, *fns):
        """Give it a list of submodules that will be called in sequence to compute the output's non-linear part."""
        super().__init__()
        self.fn = fns[0] if len(fns) == 1 else nn.Sequential(*fns)
    def forward(self, *xs):
        """`y = x + f(x)`, slicing/padding `x` if it is of a wrong size.

        For convenience, if many inputs are passed in, they are concatenated into one `x` along the last dimension."""
        x = cat(xs)
        y = self.fn(x)
        assert x.shape[:-1] == y.shape[:-1]
        x_sz, y_sz = x.shape[-1], y.shape[-1]
        if x_sz == y_sz: return x + y
        if x_sz > y_sz: return x[..., :y_sz] + y
        return cat(x, torch.zeros(*x.shape[:-1], y_sz - x_sz, dtype=x.dtype, device=x.device)) + y
    def came_from(self, x, *past):
        """Marks `x` as having been computed from `past` states, teleporting the gradient.

        `past` can be sampled arbitrarily, without the need for an unbroken chain of history.

        The result's value is `x`, but its gradient will be received by each `past` (since addition gives gradient to all its arguments, and each RNN state is the sum of all past states). Note that if the NN parameters have changed in the meantime, that gradient is stale, especially if pasts are from very long ago."""
        p = sum(past)
        return x + p - p.detach()

    # TODO: Also have the index-aware method…
    # TODO: Also add support for cutting-off too-large values…



# TODO: …Make a `model/grad_teleport_rnn.py` file with code for a skip-connection RNN with gradient teleportation…
#   (For the grad-tp, need 2 methods: one which assumes that all inputs (past-RNN-state, computed first) come-before, and returns the first arg (future-RNN-state) as-is but with gradient directly linked to inputs; and the other, which knows the timestamps of inputs and accumulates gradients and everything.)
#   (And need to ensure that we reset individual values when they get too high (getting reset-timestamps associated with every single RNN-state, inconveniently); and not grad-teleport if reset-timestamps don't match.)
#     TODO: …How do we implement this resetting exactly, especially considering that the forward-pass should still function like a regular skip-connection…
#     - Faraway gradient teleportation: if the RNN always *adds* to history (skip-connections yo), then the gradient of future steps (the initial `history`) should always be added to the gradient of past steps (the post-`env` `history`).
#       - Should also ensure that RNN states don't get too huge, by resetting big values (`.abs()>1000`) to 0 and keeping track of the reset-timestamp, and only allowing grad-tp when reset-timestamps match.
#       - Probably implemented across minibatches, to not hurt performance: by looking at `.grad` and saving that and timestamps, and on the next batch, incorporating old-grad to output-RNN-state, AKA adding that gradient whenever our timestamp is lower. Needs a quadratic loop to incorporate.
#         - To change `O(B**2)` to `O(B log(B))`, sort: when remembering grad, `.sort()` by timestamps and and calc cumsum of all this-and-above-our-timestamp gradients; when incorporating, pick from here at `.searchsorted(…, side='right')` indices.
#         - To trade off staleness for lower variance, could actually incorporate old-grad into new-grad (with a multiplier like `.95`) in the same way as it's incorporated into a replay-step.


if __name__ == '__main__': # pragma: no cover
    pass
    # TODO: How to test this?
    #   Have to test the simplest all-in-the-past case first, to make sure that learning *can* actually happen.
    #     (Should: the concept is basically `f(y[n]) = f(x + sum(… f(y[i]).detach() …))`. …Though, if the network changed in the meantime, then the gradient is stale.)