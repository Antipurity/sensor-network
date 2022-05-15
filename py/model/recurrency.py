"""
TODO:
"""



import torch
import torch.nn as nn



class State(nn.Module):
    """`State(shape, device=None)`

    Manages RNN state: `s(s() * 2)` doubles the currently-stored value.

    RNN state could be a simple variable, but separating it into a class allows easily managing arbitrarily-many variables at once.

    In addition:
    - `State.loss(…)`: handles an episode's loss in one place.
    - TODO:"""
    _setters = []
    _losses = []
    # TODO: Also need the array of episodes…
    def __init__(self, shape, device=None):
        super().__init__()
        x = shape if isinstance(shape, torch.Tensor) else torch.randn(shape, device=device)
        self.initial = nn.parameter.Parameter(x, requires_grad=True)
        self.current = None
    def __call__(self, to=None):
        """Either:
        - Get. Either the initial value, or the previously-set value.
        - Set, with `x` as an arg. Remembers & returns `x`."""
        if to is not None:
            self.current = to
        assert self.current is not None, "Use `with State.Episode():`"
        return self.current

    @staticmethod
    def loss(reset=False):
        """Gets/sets an episode's loss.

        Either:
        - `State.loss(L)`: adds `L.mean()` to the current loss.
        - `State.loss()`: reports the current loss. Useful for logging.
        - `State.loss(True)`: reports the current loss and resets it to 0. Useful for mid-episode `State.loss(True).backward()`."""
        assert len(State._losses) > 0, "Use `with State.Episode():`"
        if isinstance(reset, bool): # We were asked to report the loss.
            L = State._losses[-1]
            if reset: State._losses[-1] = 0.
            return L
        else: # We were given a tensor to minimize the values of.
            State._losses[-1] = State._losses[-1] + reset.mean()

    class Episode:
        """TODO:"""
# TODO: Have the `State.Episode()` context manager, which resets every entered `State` to its initial version when entered, and restores state when left. (Internally, have a dict from object ID to old current-tensor.)
#   TODO: `State.Episode()` should push `0` to `State._losses` on entry.
#   TODO: `State.Episode()`, when leaving, should do `L.backward()` if it's a tensor, after popping `L`.
#   TODO: `State.Episode()`, on exit, should still contain all current values (so, on entry, 'swap' with new values, and on exit, swap with old values), and when re-entered, should restore those values. (Save+load could then store the actual episodes if needed.)
# TODO: Have the `State.Setter(fn(prev,next)→next)` context manager, which adds `fn` to the list of funcs that we call on `.set`.

x=State((5,6))
x(x() + 1)
x(x() * 2)
print(len([*x.parameters()]))
x().sum().backward()
print(x.initial.grad)