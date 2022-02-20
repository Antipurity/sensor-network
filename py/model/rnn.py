"""
TODO:
"""



import torch



def RNN(transition, optimizer, train=True, bptt_length=64, grad_checkpointing=32, trace=True, async_updates=True):
    """
    Wraps a state→state differentiable dynamical system (commonly a recurrent neural network, RNN) written in PyTorch.

    The result is a function from `state, *args` to `state`, both `state`s PyTorch tensors.

    - `transition: fn(state, *args) → (state, loss)`: the system dynamics, and what number they should minimize via `.backward()`.
    - `optimizer: torch.optim.Optimizer`: updates the system. Could be wrapped in `lambda p: torch.optim.SGD(p, lr=1e-2)`.
    - `train = True`: if `False`, it's inference-only. Could be wrapped in `lambda: ...`, called on each iteration.
    - `bptt_length = 64`: how many steps to backpropagate gradient through, capped off by `sum(loss).backward()`. Could be wrapped in a function such as `lambda: random.randint(1, 1024)`.
    - TODO: `grad_checkpointing = 32`: if `None`, no [checkpointing](https://pytorch.org/docs/stable/checkpoint.html): computation is as-quick-as-possible, but used memory grows quickly because all intermediate activations are stored. If an integer `n`, we only store `bptt_length/n` `state`s plus intermediate activations for `n` steps, but do the forward pass twice (so, about 30% slowdown).
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
    loss = 0
    # TODO: First implement the most basic skeleton. Which we kinda did.
    # TODO: Then add tracing.
    # TODO: Then add checkpointing.
    # TODO: Then add async updates (even though we decided to use replay buffers for learning…).
    def new_bptt(state):
        nonlocal loss, n, n_max
        # Backprop, and set up for the next backprop.
        if isinstance(loss, torch.Tensor):
            loss.backward()
        state = state.detach().requires_grad_(True)
        # Update.
        optimizer.step()
        optimizer.zero_grad(True)
        # Read params for the next BPTT.
        n, n_max = 0, bptt_length() if callable(bptt_length) else bptt_length
        assert isinstance(n_max, int)
        return state
    def step(state, *args):
        nonlocal n
        # TODO: First, how to wrap everything in `with torch.no_grad():`, but only if `not (train() if callable(train) else train)`?
        if n_max == 0: state = new_bptt(state)
        n += 1
        state = transition(state, *args)
        if n >= n_max: state = new_bptt(state)
    return step



# TODO: A function for sanity-checking this, right? Like, have a sequence of random vectors, and expose an RNN to them, which should learn to predict the next vector (or the first if at the last vector).
# TODO: Rename this file to make it the __init__.py of the `torch_rnn` (or smth) package, and publish it, because, why not.