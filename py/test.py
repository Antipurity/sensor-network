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



device = 'cuda' if torch.cuda.is_available() else 'cpu'
hidden_sz = 128
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
        SelfAttention(embed_dim=in_sz, num_heads=2),
        f(in_sz, out_sz),
        nn.LayerNorm(out_sz),
        nn.ReLU(),
    )
incorporate_input = h(sum(cell_shape), sum(cell_shape)).to(device)
state_transition = nn.Sequential( # state → state; RNN.
    h(sum(cell_shape), sum(cell_shape)),
    h(sum(cell_shape), sum(cell_shape)),
).to(device)
state_future = nn.Sequential( # state → future; BYOL projector.
    f(sum(cell_shape)),
    f(),
    f(),
).to(device)
slow_state_future = MomentumCopy(state_future)
future_transition = nn.Sequential( # future → future; BYOL predictor.
    f(),
    f(),
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
    # TODO: Normalize both across dimension 0 (so cells have diverse futures). DON'T LayerNorm.
    loss_was = (A - B).square().sum()
    return loss_was
model = RNN(
    transition = state_transition,
    loss = loss,
    optimizer = optimizer,
    backprop_length = lambda: random.randint(1, 4),
)



state = torch.randn(16, sum(cell_shape), device=device)
max_state_cells = 1024
feedback = None
loss_was = 0.
async def print_loss(data_len, query_len, explored, loss):
    loss = await sn.torch(torch, loss, True)
    print(str(data_len).rjust(3), str(query_len).ljust(2), 'explored', str(round(explored*100, 2)).rjust(5)+'%', '  L2', str(loss))
async def main():
    global state, feedback
    while True:
        # (Might want to also split data/query into multiple RNN updates if we have too much data.)
        #   (Let the RNN learn the time dynamics, a Transformer is more of a reach-extension mechanism.)
        await asyncio.sleep(.05) # TODO: Remove. It serves no purpose now, other than going slower. (The fan hardly being active sure is nice, though.)
        data, query, data_error, query_error = await sn.handle(feedback)
        data = embed_data(torch.as_tensor(data, dtype=torch.float32, device=device))
        query = embed_query(torch.as_tensor(query, dtype=torch.float32, device=device))
        state = torch.cat((state, data, query), 0)[-max_state_cells:, :]
        state = incorporate_input(state)
        state = model(state)
        feedback = sn.torch(torch, state[(-query.shape[0] or max_state_cells):, :])
        asyncio.ensure_future(print_loss(data.shape[0], query.shape[0], minienv.explored(), loss_was))
        # TODO: Why is exploration mostly at .1%, sometimes at only a couple nodes? Can we make it work better?
        # ...I'm seeing very little actual diversity between steps. Maybe we really should maximize sensitivity of futures to observed outputs?
        # TODO: Compare to baselines: random agent, no-loss.
asyncio.run(main())
# TODO: Try different normalization schemes.



# TODO: That "normalization of futures will lead to exploration" seems like a big weak point. So, try:
# - Cross-cell normalization.
# - Per-cell normalization (LayerNorm at the end of the future-predictor).
# - Running statistics (BatchNorm in eval mode).
# - Very small future-representations.
#     - Cumulative sum right before normalization?
#     - Hierarchical downsampling, which may make more sense than cumsum?
# - Weight decay with long learning?
# - Try [Barlow twins](https://arxiv.org/pdf/2103.03230.pdf), AKA "make all futures perfectly uncorrelated". (Simpler than BYOL, but also doesn't have `future_transition` unless we try it.)
# - If all fails, translate the "exploration = max sensitivity of future to the past" definition to code: `state.detach()` for the BYOL-loss, and to train `state`, forward-propagate its gradient to the future, and maximize the sum of its magnitudes. (Hopefully, doesn't explode.) (As a bonus, this returns us to our initial "opposing optimization forces at work" formulation. And doesn't actually contradict the BYOL-paradigm, because the BYOL paper doesn't consider gradient of inputs.)
#     - (Need PyTorch Nightly (or 1.11) for forward-mode gradients, though.)
# - If ALL fails, switch to SwaV.