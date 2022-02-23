"""
Testing how far we can push "compression = exploration", where we try to make the learned/abstract futures maximally sensitive to initial states (by predicting which of a diverse set of futures we will end up in, and hoping that future-diversity translates to past-sensitivity). Probably not very far.
"""



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
embed_tokens = nn.Sequential( # input → state
    nn.Linear(sum(cell_shape), hidden_sz),
    SkipConnection(
        nn.ReLU(),
        nn.LayerNorm(hidden_sz),
        nn.Linear(hidden_sz, hidden_sz),
    ),
)
state_transition = nn.Sequential( # state → state; RNN.
    # TODO: ...WAIT: we forgot to incorporate `Attention` everywhere...
    #   Where exactly, and how?...
    SkipConnection(
        nn.Linear(hidden_sz, hidden_sz),
        nn.ReLU(),
        nn.LayerNorm(hidden_sz),
    ),
    SkipConnection(
        nn.Linear(hidden_sz, hidden_sz),
        nn.ReLU(),
        nn.LayerNorm(hidden_sz),
    ),
)
state_future = nn.Sequential( # state → future; BYOL projector.
    SkipConnection(
        nn.Linear(hidden_sz, hidden_sz),
        nn.ReLU(),
        nn.LayerNorm(hidden_sz),
    ),
    SkipConnection(
        nn.Linear(hidden_sz, hidden_sz),
        nn.ReLU(),
        nn.LayerNorm(hidden_sz),
    ),
)
slow_state_future = MomentumCopy(state_future)
future_transition = nn.Sequential( # future → future; BYOL predictor.
    SkipConnection(
        nn.Linear(hidden_sz, hidden_sz),
        nn.ReLU(),
        nn.LayerNorm(hidden_sz),
    ),
    SkipConnection(
        nn.Linear(hidden_sz, hidden_sz),
        nn.ReLU(),
        nn.LayerNorm(hidden_sz),
    ),
)
# TODO: Wrap in RNN...
#   Do we want a separate class that puts all of the above together, not putting it in the main loop?
#   Or is all the future-stuff a part of the loss, and the main RNN just `embed_tokens` + `state_transition`?



# TODO: An infinite loop where we update state, and display minienv.explored().
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