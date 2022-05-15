"""
TODO:
"""



import torch
import torch.nn as nn



class State(nn.Module):
    """`State(shape, device=None)`

    Manages RNN state: `s(s() * 2)` doubles the currently-stored value.

    TODO:"""
    _setters = []
    def __init__(self, shape, device=None):
        super().__init__()
        x = shape if isinstance(shape, torch.Tensor) else torch.randn(shape, device=device, requires_grad=True)
        self.initial = nn.parameter.Parameter(x)
        self.current = self.initial+0
        #   (The current value is not reported as a parameter, and thus not saved with other state.)
        #     (Save+load would thus mess up the current values.)
        #     (Is it even possible to do this well in PyTorch?)
        #   TODO: …Maybe we should set this to None initially, so that users are forced to use `State.Episode()`?… (And maybe that ctx manager, on exit, still contains its current values, which could be restored by re-entering that ctx manager? …Isn't *this* a good solution to save+load: just store the episode if you care that much?)
    def __call__(self, to=None):
        """Either:
        - Get. Either the initial value, or the previously-`.set(×)` value.
        - Set, with `x` as an arg. Remembers & returns the argument."""
        if to is not None:
            self.current = to
        return self.current
# TODO: Have the `State.Episode()` context manager, which resets every entered `State` to its initial version when entered, and restores state when left. (Internally, have a dict from object ID to old current-tensor.)
#   TODO: Have `State.loss(L)`. `State.loss(reset=False)` returns the sum-of-`L.sum()`.
#   TODO: `State.Episode()`, when leaving, should do `L.backward()` if it's a tensor.
# TODO: Have the `State.Setter(fn(prev,next)→next)` context manager, which adds `fn` to the list of funcs that we call on `.set`.

x=State((5,6))
x(x() + 1)
x(x() * 2)
print(len([*x.parameters()]))
x().sum().backward()
print(x.initial.grad)