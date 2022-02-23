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
from model.attention import Attention
from model.momentum_copy import MomentumCopy



cell_shape, part_size = (8, 24, 64), 8
sn.shape(cell_shape, part_size)



class SkipConnection(nn.Module):
    def __init__(self, *fn): super().__init__();  self.fn = nn.Sequential(*fn)
    def forward(self, x): return self.fn(x) + x
device = 'cuda' if torch.cuda.is_available() else 'cpu'
hidden_sz = 128
embed_tokens = nn.Sequential( # tokens → input
    nn.Linear(sum(cell_shape), hidden_sz),
    SkipConnection(
        nn.ReLU(),
        nn.LayerNorm(hidden_sz),
        nn.Linear(hidden_sz, hidden_sz),
    ),
).to(device)
incorporate_input = Attention( # (input, state) → state
    kv_size = hidden_sz,
    q_size = hidden_sz,
    heads=2,
).to(device)
def h():
    return SkipConnection(
        Attention(kv_size=hidden_sz, heads=2),
        nn.ReLU(),
        nn.LayerNorm(hidden_sz),
        nn.Linear(hidden_sz, hidden_sz),
        nn.ReLU(),
        nn.LayerNorm(hidden_sz),
    )
state_transition = nn.Sequential( # state → state; RNN.
    h(),
    h(),
).to(device)
state_future = nn.Sequential( # state → future; BYOL projector.
    h(),
    h(),
).to(device)
slow_state_future = MomentumCopy(state_future)
future_transition = nn.Sequential( # future → future; BYOL predictor.
    h(),
    h(),
).to(device)
optimizer = torch.optim.SGD([
    *embed_tokens.parameters(),
    *incorporate_input.parameters(),
    *state_transition.parameters(),
    *state_future.parameters(),
    *future_transition.parameters(),
], lr=1e-3)
def loss(prev_state, next_state):
    A = future_transition(state_future(prev_state))
    B = slow_state_future(next_state)
    return (A - B).square().sum()
model = RNN(
    transition = state_transition,
    loss = loss,
    optimizer = optimizer,
    backprop_length = lambda: random.randint(1, 16),
)



state = torch.randn(16, sum(cell_shape), device=device)
async def main():
    while True:
        pass
        # TODO: Get data & query to handle.
        # TODO: embed_tokens, of data...
        #   ...What about the query though? Where would *that* be put?
        # TODO: An infinite loop where we update state, and display minienv.explored().
        # TODO: ...How to compute the actual feedback?... Wouldn't we want our `state` to contain the query in queries?...
        #   ...Should we concat, uh, something...
asyncio.run(main())
# TODO: Run & test. Try to make it work, at least via trying different normalization schemes.



# TODO: That "normalization of futures will lead to exploration" seems like a big weak point. So, try:
# - Cross-cell normalization.
# - Per-cell normalization (LayerNorm at the end of the future-predictor).
# - Running statistics (BatchNorm in eval mode).
# - Very small future-representations.
#     - Cumulative sum right before normalization?
#     - Hierarchical downsampling, which may make more sense than cumsum?
# - Weight decay with long learning?
# - If all fails, translate the "exploration = max sensitivity of future to the past" definition to code: `state.detach()` for the BYOL-loss, and to train `state`, forward-propagate its gradient to the future, and maximize the sum of its magnitudes. (Hopefully, doesn't explode.) (As a bonus, this returns us to our initial "opposing optimization forces at work" formulation. And doesn't actually contradict the BYOL-paradigm, because the BYOL paper doesn't consider gradient of inputs.)
# - If ALL fails, switch to SwaV.