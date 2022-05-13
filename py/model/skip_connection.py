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
#   (And need to ensure that we reset individual values when they get too high (getting reset-timestamps associated with every single RNN-state, inconveniently); and not grad-teleport if reset-timestamps don't match.)
#     TODO: …How do we implement this resetting exactly, especially considering that the forward-pass should still function like a regular skip-connection…
#     - Faraway gradient teleportation: if the RNN always *adds* to history (skip-connections yo), then the gradient of future steps (the initial `history`) should always be added to the gradient of past steps (the post-`env` `history`).
#       - Should also ensure that RNN states don't get too huge, by resetting big values (`.abs()>1000`) to 0 and keeping track of the reset-timestamp, and only allowing grad-tp when reset-timestamps match.
#       - Probably implemented across minibatches, to not hurt performance: by looking at `.grad` and saving that and timestamps, and on the next batch, incorporating old-grad to output-RNN-state, AKA adding that gradient whenever our timestamp is lower. Needs a quadratic loop to incorporate.
#         - To change `O(B**2)` to `O(B log(B))`, sort: when remembering grad, `.sort()` by timestamps and and calc cumsum of all this-and-above-our-timestamp gradients; when incorporating, pick from here at `.searchsorted(…, side='right')` indices.
#         - To trade off staleness for lower variance, could actually incorporate old-grad into new-grad (with a multiplier like `.95`) in the same way as it's incorporated into a replay-step.


if __name__ == '__main__': # pragma: no cover
    import random
    from log import log, finish
    batch_sz, state_sz = 64, 32
    net = SkipConnection(
        nn.Linear(state_sz + 2, state_sz),
        SkipConnection(nn.ReLU(), nn.LayerNorm(state_sz), nn.Linear(state_sz, state_sz)),
        SkipConnection(nn.ReLU(), nn.LayerNorm(state_sz), nn.Linear(state_sz, state_sz)),
        SkipConnection(nn.ReLU(), nn.LayerNorm(state_sz), nn.Linear(state_sz, state_sz)),
    )
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    replay_buffer = []
    max_replay_samples = 256

    state = torch.randn(batch_sz, state_sz)
    sum_so_far = torch.zeros(batch_sz, 1)
    for _ in range(50000):
        # Acknowledge that prior RNN steps existed.
        if len(replay_buffer) > 8:
            past = [net(random.choice(replay_buffer)) for _ in range(8)]
            state = net.came_from(state, *past)

        # Env: most values are ignored, some must be remembered in a sum, and some must output & reset that sum.
        #   The input is the marker (0|1|-1) and the value.
        p = torch.rand(batch_sz, 1)
        is_ignore, is_store, is_report = (p < .9), (p >= .9) & (p < .9666666), (p >= .9666666)
        marker = torch.where(is_ignore, torch.zeros_like(p), torch.where(is_store, torch.ones_like(p), -torch.ones_like(p)))
        value = torch.rand(batch_sz, 1)
        sum_so_far = torch.where(is_store, sum_so_far + value, sum_so_far).clamp(-3, 3)

        # Inputs to `net` must be stored in the `replay_buffer`.
        net_input = cat((state, marker, value))
        replay_buffer.append(net_input.detach())
        if len(replay_buffer) > max_replay_samples:
            random.shuffle(replay_buffer)
            replay_buffer = replay_buffer[-(max_replay_samples//2):]

        # RNN step.
        state = net(net_input)

        # Predict the sum in the correct places.
        pred = state[..., :1]
        # target = torch.where(is_report, sum_so_far, pred) # This is the correct one, not the one below.
        target = sum_so_far # TODO:
        loss = (pred - target).square().sum()
        loss = loss + pred.abs().clamp(3).sum() # Keep within -3…3.
        sum_so_far = torch.where(is_report, torch.zeros_like(p), sum_so_far)
        log(0, False, L1 = (pred - target).abs().mean().detach().cpu().numpy(), pred=pred.detach()[0].cpu().numpy(), sum=sum_so_far.detach()[0].cpu().numpy())

        loss.backward()
        opt.step();  opt.zero_grad(True)

        state = state.detach()

        # …First impressions: not encouraging…


        # TODO: Also have to limit `state` to make it never contain ridiculously-big values.
        #   …Are there really no consequences to just doing this without any gradient-filtering…
        state = torch.where(state.abs()>10, torch.zeros_like(state), state)
    finish()


    # TODO: …Last chance could be: have the "RNN-state corrector" NN, applied on `.detach()`ed-prev-state before it's sent through `net`; the loss should make next-state predict corrected-next-state.
    #   (Like synthetic gradient, but without intervening on actual gradients. More stable, maybe.)
    #   …Intuitively, how would this method help with learning to ignore inputs? A "request for a bigger result" is slowly learned and propagated, intermediate steps get bigger output for no reason, and not