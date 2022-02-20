"""
Contains `RNN`: a simple wrapper for training `fn(state, *args) → state` functions.

Run this file to test it.

Usage example:

```python
import torch
import torch.nn as nn
class OnlyFirstArg(nn.Module):
    def __init__(self, fn):
        super(OnlyFirstArg, self).__init__()
        self.fn = fn
    def forward(self, x, *_): return self.fn(x)

model = RNN(
    OnlyFirstArg(nn.Sequential(
        nn.Linear(96, 128),
        nn.ReLU(),
        nn.LayerNorm(128),
        nn.Linear(128, 96),
    )),
    lambda state, predicts: (state - predicts).square().sum(),
    lambda p: torch.optim.SGD(p, lr=3e-4),
    backprop_length=lambda: random.randint(1, 10),
)
state = torch.randn(96)
for _ in range(50000):
    state = model(state, (state*.9).detach())
```
"""



import torch
import torch.utils.checkpoint



def RNN(transition, loss, optimizer, backprop_length=64, checkpoint=True, trace=True):
    """
    Wraps a state→state differentiable dynamical system (commonly a recurrent neural network, RNN) written in PyTorch.

    The result is a function from `state, *args` to `state`, both `state`s are PyTorch tensors; call it in a loop.

    Arguments:

    - `transition: fn(state, *args) → state`: the system dynamics.
        - (If you don't need to train it, you could just call this function function instead of `RNN`.)
    - `loss: fn(state, *args) → number`: what to minimize via `.backward()`.
        - Note that `state` here is post-transition, so for something like next-state prediction, delay `RNN` steps by one so that next-state is always available.
    - `optimizer: torch.optim.Optimizer`: updates the system. Could be wrapped in `lambda p: torch.optim.SGD(p, lr=1e-2)`.
    - `backprop_length = 64`: how many steps to backpropagate gradient through, capped off by `sum(loss).backward()`. Could be wrapped in a function such as `lambda: random.randint(1, 1024)`.
    - `checkpoint = True`: if `False`, no [checkpointing](https://pytorch.org/docs/stable/checkpoint.html): computation is fast, but used memory grows quickly because all intermediate activations are stored. If `True`, needs less memory, but the forward pass is done twice (so, about 30% slowdown).
    - `trace = True`: if `transition` has no CPU-side control flow, `True` to [precompile](https://pytorch.org/docs/stable/generated/torch.jit.trace.html) for a bit of speed.
    """
    optimizer = optimizer(transition.parameters()) if callable(optimizer) else optimizer
    assert isinstance(optimizer, torch.optim.Optimizer)
    n, n_max = 0, 0
    total_loss = 0.
    def new_bptt(state):
        nonlocal total_loss, n, n_max
        # Backprop, and set up for the next backprop.
        if isinstance(total_loss, torch.Tensor) and total_loss.requires_grad:
            total_loss.backward()
        total_loss = 0.
        state = state.detach().requires_grad_(True)
        # Update.
        optimizer.step()
        optimizer.zero_grad(True)
        # Read params for the next BPTT.
        n, n_max = 0, backprop_length() if callable(backprop_length) else backprop_length
        assert isinstance(n_max, int)
        return state
    def step(state, *args):
        nonlocal n, total_loss, trace, transition
        if n_max == 0: state = new_bptt(state)
        n += 1
        if trace:
            with torch.no_grad():
                transition = torch.jit.trace(transition, (state, *args))
                trace = False
        if not checkpoint: # pragma: no cover
            state = transition(state, *args)
        else:
            state = torch.utils.checkpoint.checkpoint(transition, state, *args)
        total_loss += loss(state, *args)
        if n >= n_max: state = new_bptt(state)
        return state
    return step



if __name__ == '__main__': # pragma: no cover
    """
    A test: next-vector prediction.

    ```bash
    coverage run --branch sensor-network/py/model/rnn.py
    coverage report
    coverage html
    ```
    """
    dev = 'cpu' # 'cuda' if torch.cuda.is_available() else 'cpu'
    data = torch.randn(200, 96, device=dev)
    n, iter = 0, 0
    losses = []
    import torch.nn as nn
    import random
    import matplotlib.pyplot as plt
    import time
    start = time.monotonic()
    def loss(state, predicts):
        L = (state - predicts).square().sum()
        cL = L.cpu().detach().numpy()
        print(''+str(iter), 'L2:', cL, '' if iter%5000 else ('  time: '+str(time.monotonic() - start)+'s'), '       ', end = '\r' if iter%5000 else '\n')
        losses.append(cL)
        return L
    class OnlyFirstArg(nn.Module):
        def __init__(self, fn):
            super(OnlyFirstArg, self).__init__()
            self.fn = fn
        def forward(self, x, *_):
            return self.fn(x)
    model = RNN(
        OnlyFirstArg(nn.Sequential(
            nn.Linear(96+96, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 96),
        )).to(device=dev),
        loss,
        lambda p: torch.optim.SGD(p, lr=1e-3),
        backprop_length=lambda: random.randint(1, 10),
    )
    state = torch.randn(96, device=dev)
    for i in range(50000):
        # Train.
        iter = i
        state = model(torch.cat((state, data[n])), data[(n+1) % data.shape[0]])
        n = (n + 1) % data.shape[0]
    plt.plot(losses)
    plt.show()