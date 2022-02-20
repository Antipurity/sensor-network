"""
TODO:
"""



import torch



def RNN(transition, optimizer, train=True, backprop_length=64, grad_checkpointing=32, trace=True, async_updates=True):
    """
    Wraps a state→state differentiable dynamical system (commonly a recurrent neural network, RNN) written in PyTorch.

    The result is a function from `state, *args` to `state`, both `state`s are PyTorch tensors; call it in a loop. Call `.loss(L)` to minimize `L`.

    - `transition: fn(state, *args) → state`: the system dynamics, and what number they should minimize via `.backward()`.
    - `optimizer: torch.optim.Optimizer`: updates the system. Could be wrapped in `lambda p: torch.optim.SGD(p, lr=1e-2)`.
    - `train = True`: if `False`, it's inference-only. Could be wrapped in `lambda: ...`, called on each iteration.
    - `backprop_length = 64`: how many steps to backpropagate gradient through, capped off by `sum(loss).backward()`. Could be wrapped in a function such as `lambda: random.randint(1, 1024)`.
    - TODO: `grad_checkpointing = 32`: if `None`, no [checkpointing](https://pytorch.org/docs/stable/checkpoint.html): computation is as-quick-as-possible, but used memory grows quickly because all intermediate activations are stored. If an integer `n`, we only store `backprop_length/n` `state`s plus intermediate activations for `n` steps, but do the forward pass twice (so, about 30% slowdown).
        - TODO: `torch.get_rng_state()→state`, `torch.set_rng_state(state)`
    - TODO: `trace = True`: if `transition` has no CPU-side control flow, `True` can be used here to (potentially) speed up execution at the cost of slow startup.
    - TODO: `async_updates=True`: ...Also: if we implement checkpointing manually, then we'd be able to train online by intelligently scheduling backward updates, and copying updates...
        - WHY CAN WE SUDDENLY THINK OF SO MANY GOOD FEATURES SO LATE
        - THEY ARE HARD TO IMPLEMENT
        - WHY NOW
    """
    optimizer = optimizer(transition.parameters()) if callable(optimizer) else optimizer
    assert isinstance(optimizer, torch.optim.Optimizer)
    n, n_max = 0, 0
    loss = 0.
    # TODO: First implement the most basic skeleton. Which we kinda did. Untested though.
    # TODO: Then add tracing.
    # TODO: Then add checkpointing.
    # TODO: Then add async updates (even though we decided to use replay buffers for learning…).
    def new_bptt(state):
        nonlocal loss, n, n_max
        # Backprop, and set up for the next backprop.
        if isinstance(loss, torch.Tensor):
            loss.backward()
            loss = 0.
        state = state.detach().requires_grad_(True)
        # Update.
        optimizer.step()
        optimizer.zero_grad(True)
        # Read params for the next BPTT.
        n, n_max = 0, backprop_length() if callable(backprop_length) else backprop_length
        assert isinstance(n_max, int)
        return state
    def step(state, *args):
        nonlocal n
        if n_max == 0: state = new_bptt(state)
        n += 1
        training = train() if callable(train) else train
        with torch.enable_grad() if training else torch.no_grad():
            state = transition(state, *args)
            # TODO: What about the loss...
            #   Should we have `step.loss(L)` which just adds to the loss anytime?…
            #     Would it really be alright for it to not be traced... Eh, who cares.
        if n >= n_max: state = new_bptt(state)
    def add_loss(L): nonlocal loss;  loss += L
    step.loss = add_loss
    return step



if __name__ == '__main__':
    # Test: next-vector prediction.
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = torch.randn(200, 96, device=dev)
    n = 0
    import torch.nn as nn
    import random
    model = RNN(
        nn.Sequential(
            nn.Linear(96, 128),
            nn.LayerNorm(96),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(96),
            nn.ReLU(),
            nn.Linear(128, 96),
        ).to(device=dev), # TODO: ...Wait: this was supposed to output the loss...
        #   TODO: HOW TO DO A ONE-LINER WITH THE LOSS?! ...Should we make `loss` a separate function after all...
        lambda p: torch.optim.SGD(p, lr=1e-2),
        backprop_length=lambda: random.randint(1, 1024),
    )
    state = torch.randn(96, device=dev)
    for _ in range(50000):
        # Train.
        state = model(torch.concat((state, data[n])))
        n = (n + 1) % data.shape[0]
        L = (state - data[n]).sqr()
        model.loss(L) # TODO: ...But what if that iteration has already passed... ...Maybe `new_bptt` should be called in `model.loss`, not `model`?... But what if users just forget to call this func...
        #   Maybe we really should make `transition` return the `loss`...
        print(L)
    # TODO: Have a sequence of random vectors, and repeatedly expose an RNN to them, printing the next-vector-prediction loss (or the first vector if at the last).
# TODO: Rename this file to make it the __init__.py of the `torch_rnn` (or smth) package, and publish it, because, why not. And, async updates are kinda dope.