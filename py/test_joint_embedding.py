"""
A test of whether embedding-prediction (AKA joint embedding) can blot out randomness, unlike direct prediction. Result: yes.

Say, we have a mapping, such as an image augmentation (such as crop or blur), which is supposed to preserve the meaning of data.

To learn it, we can just predict output from input with a neural net. But unpredictable parts cause variance to go up, meaning that the loss can never go below a certain level; for example, simple video predictions are usually quite blurry.

Alternatively, we can embed input & output into another space, and do prediction there; the embedder would then be able to learn to ignore randomness. [BYOL](https://arxiv.org/abs/2006.07733) showed that a way to make this work (prevent embedders from collapsing into a uniform representations) is to make the output-embedder into the slowly-updated copy of the input-embedder.

In this file is a comparision. Uncomment one of the branches near the end.
"""



import torch
import torch.nn as nn

from model.momentum_copy import MomentumCopy
from model.log import log, clear, finish



def f(x):
    """A transformation that has both a fully-predictable part and an inherently-random part."""
    a, b = torch.split(x, 16, -1)
    return torch.cat((a+1, b-1 + torch.randn_like(b)/3), -1) # (If no /3, results can be much harder to distinguish, though BYOL is still a lot better than prediction.)



next = nn.Sequential(
    nn.Linear(32, 32), nn.ReLU(), nn.LayerNorm(32),
    nn.Linear(32, 32), nn.ReLU(), nn.LayerNorm(32),
    nn.Linear(32, 32), nn.ReLU(), nn.LayerNorm(32),
    nn.Linear(32, 32), nn.LayerNorm(32),
)
embed = nn.Sequential(
    nn.Linear(32, 32), nn.ReLU(), nn.LayerNorm(32),
    nn.Linear(32, 32), nn.ReLU(), nn.LayerNorm(32),
    nn.Linear(32, 32), nn.ReLU(), nn.LayerNorm(32),
    nn.Linear(32, 32), nn.LayerNorm(32),
)
embed_copy = MomentumCopy(embed, .999)
opt = torch.optim.Adam([*next.parameters(), *embed.parameters()], lr=1e-3) # 1e-2 is too high for BYOL.

batch_size = 1000



# Pure prediction: loss stays at ≈2.8.
# for iter in range(5000):
#     x = torch.randn(batch_size, 32)
#     y = f(x)
#     y = (y - y.mean(-1, keepdim=True)) / (y.std(-1, keepdim=True) + 1e-5)
#     y_pred = next(x)
#     loss = (y_pred - y).square().sum()
#     loss.backward()
#     opt.step();  opt.zero_grad()
#     if iter == 1000: clear()
#     log(l2 = (loss / batch_size).detach().cpu().numpy())


# A joint-embedding architecture, BYOL: loss is at ≈0.2, so randomness is blotted out where possible.
#   (If run for a long time, especially with lower `embed_copy` momentum, loss rises to ≈1…1.75 and falls in waves. Still better than prediction.)
for iter in range(500000):
    x = torch.randn(batch_size, 32)
    with torch.no_grad():
        y = embed_copy(f(x))
    y_pred = next(embed(x))
    loss = (y_pred - y).square().sum()
    loss.backward()
    opt.step();  opt.zero_grad();  embed_copy.update()
    if iter == 1000: clear()
    log(l2 = (loss / batch_size).detach().cpu().numpy())



finish()