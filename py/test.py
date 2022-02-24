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
def h(in_sz = hidden_sz, out_sz = hidden_sz):
    return SkipConnection(
        SelfAttention(embed_dim=in_sz, num_heads=2),
        nn.ReLU(),
        nn.LayerNorm(in_sz),
        nn.Linear(in_sz, out_sz),
        nn.ReLU(),
        nn.LayerNorm(out_sz),
    )
state_transition = nn.Sequential( # state → state; RNN.
    h(sum(cell_shape), sum(cell_shape)),
    h(sum(cell_shape), sum(cell_shape)),
).to(device)
state_future = nn.Sequential( # state → future; BYOL projector.
    h(sum(cell_shape)),
    h(),
    h(),
).to(device)
slow_state_future = MomentumCopy(state_future)
future_transition = nn.Sequential( # future → future; BYOL predictor.
    h(),
    h(),
).to(device)
optimizer = torch.optim.Adam([
    *embed_data.parameters(),
    *embed_query.parameters(),
    *state_transition.parameters(),
    *state_future.parameters(),
    *future_transition.parameters(),
], lr=1e-4)
def loss(prev_state, next_state):
    A = future_transition(state_future(prev_state))
    B = slow_state_future(next_state.detach())
    return (A - B).square().sum()
model = RNN(
    transition = state_transition,
    loss = loss,
    optimizer = optimizer,
    backprop_length = lambda: random.randint(1, 4),
)



state = torch.randn(16, sum(cell_shape), device=device)
max_state_cells = 1024
feedback = None
async def main():
    global state, feedback
    while True:
        await asyncio.sleep(.1) # TODO: Uncomment. It serves no purpose now, other than going slower.
        data, query, data_error, query_error = await sn.handle(feedback)
        print(data.shape[0], query.shape[0], ' ', len(sn.default._prev_fb)) # TODO:
        data = embed_data(torch.as_tensor(data, dtype=torch.float32, device=device))
        query = embed_query(torch.as_tensor(query, dtype=torch.float32, device=device))
        state = torch.cat((state, data, query), 0)[-max_state_cells:, :]
        state = model(state)
        feedback = sn.torch(torch, state[(-query.shape[0] or max_state_cells):, :])
        # print('explored', str(minienv.explored()*100)+'%') # TODO:
        # TODO: Okay. Now. Why is exploration at exactly 1 node literally all the time?
        #   TODO: Does a random agent achieve any exploration? Do world-resets happen every single time? (Because if random can't do anything, then we should probably make `sn` not do any fractal filling on data, and rely on users to do anything. ...Which is probably a good idea anyway, to minimize user surprise; if users need more precision, they can implement any folding themselves.)
        # ...I'm seeing very little actual diversity between steps. Maybe we really should maximize sensitivity of futures to observed outputs?
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
# - Try [Barlow twins](https://arxiv.org/pdf/2103.03230.pdf), AKA "make all futures perfectly uncorrelated". (Simpler than BYOL, but also doesn't have `future_transition` unless we try it.)
# - If all fails, translate the "exploration = max sensitivity of future to the past" definition to code: `state.detach()` for the BYOL-loss, and to train `state`, forward-propagate its gradient to the future, and maximize the sum of its magnitudes. (Hopefully, doesn't explode.) (As a bonus, this returns us to our initial "opposing optimization forces at work" formulation. And doesn't actually contradict the BYOL-paradigm, because the BYOL paper doesn't consider gradient of inputs.)
#     - (Need PyTorch Nightly (or 1.11) for forward-mode gradients, though.)
# - If ALL fails, switch to SwaV.