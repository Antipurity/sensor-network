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
    - `with State.Episode():` for starting a new unroll, with the initial `State`. (Which is differentiable; what else would you reasonably reset to? Either a frozen tensor or a frozen past state, none of which allow for efficient learning.)
    - TODO:"""
    __slots__ = ('id', 'initial', 'current')
    _setters = []
    _losses = []
    _episodes = []
    def __init__(self, shape, device=None):
        super().__init__()
        x = shape if isinstance(shape, torch.Tensor) else torch.randn(shape, device=device)
        self.id = id(self) # Might expose OS-level details. And might collide between processes.
        self.initial = nn.parameter.Parameter(x, requires_grad=True)
        self.current = None
    def __call__(self, to=None):
        """Either:
        - Get. Either the initial value, or the previously-set value.
        - Set, with `x` as an arg. Remembers & returns `x`."""
        assert len(State._episodes) > 0, "Use `with State.Episode():`"
        State._episodes[-1]._register(self)
        if to is not None:
            self.current = to
        return self.current

    @staticmethod
    def loss(reset=False):
        """Gets/sets an episode's loss.

        Either:
        - `State.loss(L)`: adds `L.sum()` to the current loss.
        - `State.loss()`: reports the current loss. Useful for logging.
        - `State.loss(True)`: reports the current loss and resets it to 0. Useful for mid-episode `State.loss(True).backward()`."""
        assert len(State._losses) > 0, "Use `with State.Episode():`"
        if isinstance(reset, bool): # We were asked to report the loss.
            L = State._losses[-1]
            if reset: State._losses[-1] = 0.
            return L
        else: # We were given a tensor to minimize the values of.
            State._losses[-1] = State._losses[-1] + reset.sum()

    class Episode:
        """
        TODO:
        """
        __slots__ = ('state_obj', 'state_old', 'restorable', 'active')
        def __init__(self):
            self.state_obj = {} # id → State
            self.state_old = {} # id → current
            self.restorable = set()
            self.active = False
        def __enter__(self):
            assert not self.active, ""
            self.active = True # No recursive re-entry.
            State._losses.append(0.) # Allow losses.
            State._episodes.append(self) # Make ourselves known.
            return self # For save/load.
        def __exit__(self, type, value, traceback):
            State._episodes.pop() # Make us forgotten.
            # Handle losses.
            L = State._losses.pop()
            if isinstance(L, torch.Tensor): L.backward()
            self.active = False # Allow non-recursive re-entry.
            # Reset all values to pre-entry states.
            objs, olds = self.state_obj, self.state_old
            for id, s in self.state_obj.items():
                s.current, olds[id] = olds[id], s.current
            self.restorable.update(objs.keys())

        def remove(self, s):
            """Provided instead of any garbage collection: if any `State` object will not be used anymore, then just remove it."""
            del self.state_obj[s.id]
            del self.state_old[s.id]
            if s.id in self.restorable: self.restorable.remove(s.id)
        def _register(self, s):
            """Registers a `State` object in this episode, updating its `.current` value if needed."""
            id, objs = s.id, self.state_obj
            if id in objs: return None
            olds, rest = self.state_old, self.restorable
            if id not in rest:
                objs[id] = s
                olds[id] = s.current
                s.current = s.initial+0 # `+0` to not treat this as an assignable param.
            else:
                rest.remove(id)
                olds[id], s.current = s.current, olds[id]

    # TODO: Have the `State.Setter(fn(prev,next)→next)` context manager, which adds `fn` to the list of funcs that we call on `.set`.



# TODO: …Also SWRM…



if __name__ == '__main__': # pragma: no cover
    def run(f): f() # TODO: Use this as a decorator for tests.
    x=State((5,6))
    with State.Episode():
        x(x() + 1)
        x(x() * 2)
        print(len([*x.parameters()]))
        x().sum().backward()
        print(x.initial.grad)
    # TODO: Test basic operation.
    # TODO: Test re-entering the same episode.
    # TODO: Test nested episodes.
    # TODO: Test recursive re-entering.
    # TODO: Test overridable setters.
    # TODO: (Also want a test for saving/loading, with an episode, making sure that it works correctly.)
    #   TODO: …How to support save+load?… What was PyTorch's way of doing that, again?