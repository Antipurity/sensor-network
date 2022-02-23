"""
Testing how far we can push "compression = exploration", where we try to make the learned/abstract futures maximally sensitive to initial states. Probably not very far.
"""



import torch
import torch.nn as nn

import sensornet as sn
import minienv
from model.rnn import RNN
from model.attention import Attention
# TODO: Also have `model.momentum_copy`, copied from the earlier test.



device = 'cuda' if torch.cuda.is_available() else 'cpu'
# TODO: Create the model...
#   What parts do we want, exactly?
#     embed_tokens: input → state
#     state_transition: state → state
#     state_future: state → future # BYOL projection.
#       (With a momentum-copy.)
#     future_transition: future → future # BYOL predictor.
# TODO: Wrap in RNN...
#   Do we want a separate class that puts all of the above together, not putting it in the main loop?
#   Or is all the future-stuff a part of the loss, and the main RNN just `embed_tokens` + `state_transition`?



# TODO: An infinite loop where we update state, and display minienv.explored().
# TODO: Run & test. Try to make it work, at least via trying different normalization schemes.