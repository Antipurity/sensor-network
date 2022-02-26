"""
Testing how far we can push "compression = exploration", where we try to make the learned/abstract futures maximally sensitive to initial states (by predicting which of a diverse set of futures we will end up in, and hoping that future-diversity translates to past-sensitivity). Probably not very far.
"""



import asyncio
import random
import torch
import torch.nn as nn

import sensornet as sn
import minienv
from model.rnn import RNN
from model.momentum_copy import MomentumCopy



cell_shape, part_size = (8, 24, 64), 8
sn.shape(cell_shape, part_size)

minienv.reset(can_reset_the_world = False, allow_suicide = False)



class SkipConnection(nn.Module):
    def __init__(self, *fn): super().__init__();  self.fn = nn.Sequential(*fn)
    def forward(self, x):
        y = self.fn(x)
        return y + x if x.shape == y.shape else y
class SelfAttention(nn.Module):
    def __init__(self, *args, **kwargs): super().__init__();  self.fn = nn.MultiheadAttention(*args, **kwargs)
    def forward(self, x):
        x = torch.unsqueeze(x, -2)
        y, _ = self.fn(x, x, x, need_weights=False)
        return torch.squeeze(y, -2)
class Sum(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x): return x.sum(0)



device = 'cuda' if torch.cuda.is_available() else 'cpu'
hidden_sz, fut_sz = 128, 128
embed_data = nn.Sequential( # data → state (to concat at the end)
    nn.Linear(sum(cell_shape), hidden_sz),
    SkipConnection(
        nn.ReLU(),
        nn.LayerNorm(hidden_sz),
        nn.Linear(hidden_sz, sum(cell_shape)),
    ),
).to(device)
embed_query = nn.Sequential( # query → state (to concat at the end)
    nn.Linear(sum(cell_shape) - cell_shape[-1], hidden_sz),
    SkipConnection(
        nn.ReLU(),
        nn.LayerNorm(hidden_sz),
        nn.Linear(hidden_sz, sum(cell_shape)),
    ),
).to(device)
def f(in_sz = hidden_sz, out_sz = hidden_sz):
    return SkipConnection(
        nn.LayerNorm(in_sz),
        nn.ReLU(),
        nn.Linear(in_sz, out_sz),
    )
def h(in_sz = hidden_sz, out_sz = hidden_sz):
    return SkipConnection(
        nn.LayerNorm(in_sz),
        nn.ReLU(),
        SelfAttention(embed_dim=in_sz, num_heads=2),
        f(in_sz, out_sz),
    )
incorporate_input = h(sum(cell_shape), sum(cell_shape)).to(device)
state_transition = nn.Sequential( # state → state; RNN.
    h(sum(cell_shape), sum(cell_shape)),
    h(sum(cell_shape), sum(cell_shape)),
).to(device)
state_future = nn.Sequential( # state → future; BYOL projector.
    h(sum(cell_shape), fut_sz),
    Sum(),
).to(device)
slow_state_future = MomentumCopy(state_future)
future_transition = nn.Sequential( # future → future; BYOL predictor.
    f(fut_sz, hidden_sz),
    f(hidden_sz, fut_sz),
).to(device)
optimizer = torch.optim.Adam([
    *embed_data.parameters(),
    *embed_query.parameters(),
    *incorporate_input.parameters(),
    *state_transition.parameters(),
    *state_future.parameters(),
    *future_transition.parameters(),
], lr=1e-4)
def loss(prev_state, next_state):
    global loss_was
    A = future_transition(state_future(prev_state))
    B = slow_state_future(next_state.detach())
    slow_state_future.update()
    A = A - A.mean()
    A = A / (A.std() + 1e-5)
    B = B - B.mean()
    B = B / (B.std() + 1e-5)
    loss_was = (A - B).square().sum()
    # TODO: ...How to maximize agreement of the computed future (no-grad except to the RNN? through the momentum-copy, maybe?) and our (prev) goals (which we'll be giving as an extra arg)?
    #   Maybe just minimize L2 loss (since we now have only 1 state vector)?
    return loss_was
model = RNN(
    transition = state_transition,
    loss = loss,
    optimizer = optimizer,
    backprop_length = lambda: random.randint(2, 3), # TODO: (Figure out what's up with the crazy memory usage.)
    trace = False, # TODO: (…This doesn't help reduce memory usage… False advertising? …Or is it due to the non-RNN `embed_data` and `embed_query` and `incorporate_input`, and actually, `future_transition` and `state_future` too?)
)



state = torch.randn(16, sum(cell_shape), device=device)
# TODO: Have `goal`: a vector of length fut_sz.
# TODO: …Condition RNN state on the `goal`: concat it to each cell and put everything through an MLP…
# TODO: …Make futures predict goals, but not with all weights, but with RNN weights…
max_state_cells = 1024
feedback = None
loss_was = 0.
exploration_peaks = [0.]
async def print_loss(data_len, query_len, explored, loss, reachable):
    loss = await sn.torch(torch, loss, True)
    explored = round(explored*100, 2)
    if explored >= exploration_peaks[-1]: exploration_peaks[-1] = explored
    else: exploration_peaks.append(explored)
    if len(exploration_peaks) > 2048:
        exploration_peaks[:-1024] = [sum(exploration_peaks[:-1024]) / len(exploration_peaks[:-1024])]
    explored_avg = sum(exploration_peaks) / len(exploration_peaks)
    print(str(data_len).rjust(3), str(query_len).ljust(2), 'explored', str(explored).rjust(5)+'%', ' avg', str(round(explored_avg, 2)).rjust(5)+'%', ' reachable', str(round(reachable*100, 2)).rjust(5)+'%', '  L2', str(loss))
async def main():
    global state, feedback
    while True:
        # (Might want to also split data/query into multiple RNN updates if we have too much data.)
        #   (Let the RNN learn the time dynamics, a Transformer is more of a reach-extension mechanism.)
        # (Might also want to do proper GPT-style pre-training, predicting shifted-by-1-to-the-left input, or ensuring that resulting representations stay the same.)
        await asyncio.sleep(.05) # TODO: Remove. It serves no purpose now, other than going slower. (The fan hardly being active sure is nice, though.)
        data, query, data_error, query_error = await sn.handle(feedback)
        data = embed_data(torch.as_tensor(data, dtype=torch.float32, device=device))
        query = embed_query(torch.as_tensor(query, dtype=torch.float32, device=device))
        state = torch.cat((state, data, query), 0)[-max_state_cells:, :]
        state = incorporate_input(state)
        state = model(state)
        feedback = sn.torch(torch, state[(-query.shape[0] or max_state_cells):, :])
        asyncio.ensure_future(print_loss(data.shape[0], query.shape[0], minienv.explored(), loss_was, minienv.reachable()))

        # TODO: Change `goal` sometimes (10% iterations?).

        # import numpy as np # TODO:
        # sn.data(None, np.random.rand(1, 96)*2-1) # TODO:
        #   TODO: Instead of this, concat that noise to `data`, so that we don't get ghost steps.
asyncio.run(main())




# TODO: Fix exploration. Try different normalization schemes. Simplify the problem and/or models. Establish *some* ground that works at least a bit, or else we'll just be floundering forever.



# TODO: That "normalization of futures will lead to exploration" seems like a big weak point. So, try (reporting avg-of-peaks exploration):
# - ❌ Make future-nets have no attention, so every cell fends for itself. (Slightly better. Which makes no sense: post-RNN-transition cells are not in the same place, so we need attention.)
# - Baselines:
#     - ✓ Random agent. .37% (1.5% with allow_suicide=False)
#     - ⋯ No-BYOL-loss agent (frozen-weights RNN).
# - Normalization:
#     - ❌ Cross-cell normalization.
#         - Single-cell 4-number futures: .1%, .1%, .13%, .15%, .24%, .28% (usually doesn't converge)
#     - ❌ Per-cell normalization (LayerNorm at the end of the future-predictor).
#         - Single-cell 4-number futures: .14%, .18%, .22%, .25%, .45%
#     - ❌ Whole-state normalization (empirically the best, though not by much).
#         - Single-cell 4-number futures: .1%, .11%, .18%, .55%, .78%
#         - Single-cell 128-number futures: .23% .29% .34% .41% .5% (may seem better, but the reporting method actually changed to have less bugs and no longer penalize .1%-swathes, so, about the same as 4-num-fut)
#         - With `state_future` using attention and not just linear layers: .1% .1% .1% .1% .1% .11% .12% .15% .42%
#     - ⋯ Running statistics (BatchNorm in eval mode).
# - ❌ Very small future-representations. (No significant advantage.)
#     - ONLY if successful:
#         - Cumulative sum right before normalization?
#         - Hierarchical downsampling, which may make more sense than cumsum?
#         - Weight decay with long learning?
# - ⋯ Try having only one future vector, rather than per-cell vectors; BYOL doesn't have normalization issues, so why should we? Were we failing because our BYOL impl was wrong? (What does it even mean to encode the future state of a cell rather than a system, anyway?)
#     - (Proven to be no use by itself, but goal-directedness won't work without this.)
# - Put off:
#     - ⋯ Disallow suicide?
#     - ⋯ Try to sometimes reset state to a random vector. (And/or try making `trivial_exploration_test.py` use proper RNN machinery, and see whether its exploration disappears. Is our normalization not good enough to ensure uniform sampling in a space we should care about?)
#         - ⋯ A sensor of random data??
#     - ⋯ Try many batches at the same time, rewriting `minienv` to allow that?
#     - ⋯ Alternative losses:
#         - ⋯ Try [Barlow twins](https://arxiv.org/pdf/2103.03230.pdf), AKA "make all futures perfectly uncorrelated". (Simpler than BYOL, but also doesn't have `future_transition` unless we try it.)
#         - ⋯ If all fails, translate the "exploration = max sensitivity of future to the past" definition to code: `state.detach()` for the BYOL-loss, and to train `state`, forward-propagate its gradient to the future, and maximize the sum of its magnitudes. (Hopefully, doesn't explode.) (As a bonus, this returns us to our initial "opposing optimization forces at work" formulation. And doesn't actually contradict the BYOL-paradigm, because the BYOL paper doesn't consider gradient of inputs.)
#             - (Need `pytorch-nightly` (or 1.11) for forward-mode gradients here, though.)
#     - ⋯ …Add noise to prior state, to match [Mean Teacher](https://arxiv.org/pdf/1703.01780.pdf) better?…
# - ⋯ If all fails, go back to RL's "compression may be good for exploration" that we were trying to average-away by "compression is exploration": have a non-differentiable `goal` state (sized `cells×fut_sz`) on which state is conditioned, updated randomly sometimes, and make futures predict `goal` (possibly through a critic that predicts futures, to interfere with gradient less) (or `.detach()`ing BYOL-loss and making the RNN maximize goal-ness of the future).
#     - (A much more principled way of adding noise than "just inject noise into internal state". So it's very good and should be implemented, right?)
#     - (It's basically pretraining for RL: actual goal-maximization could be concerned with just predicting per-cell goals since they can control everything, and thus be much more efficient.)
#     - (If only that works, then we can think about 1> simplification and 2> trying to achieve exponentially-many goals in linear time.)