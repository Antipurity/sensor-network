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
    - `.initial`: the value for new episodes to start from.
    - `State.loss(…)`: handles an episode's loss in one place.
    - `with State.Episode():` for starting a new unroll, with the `.initial` value. (Which is differentiable; what else would you reasonably reset to? Either a frozen tensor or a frozen past state, none of which allow for efficient learning.)
    - `with State.Setter(lambda state, to: to):` modifies state modifications; see the class for usage examples.
    - To save/load: `torch.save`/`torch.load`, on models that contain `State`s, and/or episodes. Saving models within an episode would make their current values top-level (episode-less).
      - Advanced: to save/load via `model.state_dict()`/`model.load_state_dict(d)` (episodes are not supported), have to manually reset `.current` values. Hard-resetting like `with State.Setter(lambda s,_: s.initial+0): model()` should do the trick. (Not handling this was a design choice made for efficiency.)
    """
    # __slots__ = ('id', 'initial', 'current')
    #   `torch.load` complains if we follow these best practices, since it gives us the props of `nn.Module`s.
    _setters = []
    _losses = [0.]
    _episodes = []
    def __init__(self, shape, device=None):
        super().__init__()
        x = shape if isinstance(shape, torch.Tensor) else torch.randn(shape, device=device)
        self.id = id(self) # Might expose OS-level details. And might collide between processes.
        self.initial = nn.parameter.Parameter(x, requires_grad=True)
        self.current = self.initial+0 # `+0`: no duplicate `Parameter` here.
    def __getstate__(self):
        """Custom pickling, for about ≈400-byte savings."""
        return self.id, self.initial, self.current.detach()
    def __setstate__(self, state):
        super().__init__()
        self.id, self.initial, self.current = state
    def __call__(self, to=None):
        """Either:
        - Get. Either the initial value, or the previously-set value.
        - Set, with `x` as an arg. Remembers & returns `x`."""
        if len(State._episodes) > 0:
            State._episodes[-1]._register(self)
        if to is not None:
            for fn in State._setters:
                r = fn(self, to)
                if r is not None: to = r
            self.current = to
        return self.current

    @staticmethod
    def loss(reset=False):
        """Gets/sets an episode's loss.

        Either:
        - `State.loss(L)`: adds `L.sum()` to the current loss.
        - `State.loss()`: reports the current loss. Useful for logging.
        - `State.loss(True)`: reports the current loss and resets it to 0. Useful for mid-episode `State.loss(True).backward()`."""
        assert len(State._losses) > 0
        if isinstance(reset, bool): # We were asked to report the loss.
            L = State._losses[-1]
            if reset: State._losses[-1] = 0.
            return L
        else: # We were given a tensor to minimize the values of.
            State._losses[-1] = State._losses[-1] + reset.sum()

    class Episode:
        """`with State.Episode(): ...`

        Represents one trajectory, unrolled so that learning can happen on it. Each `State` starts at its initial value within it.

        Can be pre-created and re-entered many times. (But not recursively. For that, use normal re-entrance: leave all intermediate episodes, return to this one, then re-enter intermediates.)

        Encountered `State`s are not garbage-collected automatically. Use `ep.remove(state)` if that is really required.
        """
        __slots__ = ('state_obj', 'state_old', 'restorable', 'active')
        def __init__(self):
            self.state_obj = {} # id → State
            self.state_old = {} # id → current
            self.restorable = set()
            self.active = False
        def __enter__(self):
            assert not self.active, "Recursive re-entry is not allowed, so express it with normal re-entry"
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
            id, objs, olds, rest = s.id, self.state_obj, self.state_old, self.restorable
            if id not in rest:
                if id not in objs:
                    objs[id] = s
                    olds[id] = s.current
                    s.current = s.initial+0 # `+0` to not treat this as an assignable param.
                else:
                    return
            else:
                rest.remove(id)
                olds[id], s.current = s.current, olds[id]

    class Setter:
        """`with State.Setter(lambda state, to: to): ...`

        Within this context, `State` updates will go through the specified function first. To not change `to`, return `to` or `None`.

        Example uses:
        - `lambda state, to: state.initial+0 if random.randint(1,100)==1 else to`: hard-reset.
        - `lambda state, to: state.initial*.001 + .999*to`: soft-reset, for very-long episodes.
        - `lambda state, to: State.loss((state() - to.detach()).square())`: predict updates and maximize [mutual information](https://en.wikipedia.org/wiki/Directed_information), for extracting good features and skipping updates.
        """
        __slots__ = ('fn',)
        def __init__(self, fn): self.fn = fn
        def __enter__(self): State._setters.append(self.fn)
        def __exit__(self, type, value, traceback): State._setters.pop()



class SRWM(nn.Module):
    """`SRWM(ins, outs, batch_size=None, device=None)`

    [Self-referential weight matrix](https://arxiv.org/pdf/2202.05780.pdf): a linear-time RNN-like alternative to attention, with meta-learning built-in.
    - `ins`, `outs`: sizes of input & output vectors.
    - `batch_size`: if `None`, inputs should be 1D vectors, else `(batch_size, 1, ?)`-shaped tensors.

    Use this in [Transformer](https://arxiv.org/abs/1706.03762) layers.
    """
    def __init__(self, ins, outs, batch_size=None, device=None):
        super().__init__()
        self.ins, self.outs = ins, outs
        O = outs + ins + ins + 1
        self.W = State((ins, O) if batch_size is None else (batch_size, ins, O), device=device)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(-1)
    def __call__(self, x):
        W = self.W()
        ins, outs = self.ins, self.outs
        si, sm = self.sigmoid, self.softmax
        assert x.shape[-1] == ins
        y, k, q, lr = torch.split(sm(x) @ W, (ins, outs, outs, 1), -1)
        vk, vq = sm(k) @ W, sm(q) @ W
        self.W(W + (si(lr) * (vq - vk)).unsqueeze(-2) * sm(k).unsqueeze(-1))
        return y
    # TODO: …How do we implement the multihead case…



if __name__ == '__main__': # pragma: no cover
    def run(f): f()
    @run
    def test0():
        """Basic `State` operation."""
        s = State((1,1))
        s(s() + 1)
        s(s() * 2)
        s().sum().backward()
        assert (s.initial.grad == 2*torch.ones((1,1))).all()
    @run
    def test1():
        """`State.Episode` operation."""
        s = State((1,1))
        with State.Episode():
            s(s() + 1)
            s(s() * 2)
            s().sum().backward()
            assert (s.initial.grad == 2*torch.ones((1,1))).all()
    @run
    def test2():
        """Re-entering a `State.Episode`."""
        s = State(torch.zeros((1,1)))
        ep = State.Episode()
        with ep:
            s(s() + 1)
            s(s() * 2)
        assert (s.initial == s.current).all()
        with ep:
            s().sum().backward()
            assert (s.initial.grad == 2*torch.ones((1,1))).all()
    @run
    def test3():
        """Nested `State.Episode`s."""
        s = State(torch.zeros((1,1)))
        with State.Episode():
            s(s() + 1)
            with State.Episode():
                s(s() * 2)
                assert (s() == torch.zeros((1,1))).all()
            s().sum().backward()
            assert (s.initial.grad == torch.ones((1,1))).all()
    @run
    def test4():
        """No recursive `State.Episodes`."""
        try:
            ep = State.Episode()
            with ep:
                with ep:
                    ...
            raise RuntimeError()
        except AssertionError: pass
    @run
    def test5():
        """`State.loss` applications."""
        s = State(torch.zeros((1,1)))
        with State.Episode():
            s(s() + 1)
            State.loss(s().square())
            s(s() * 2)
            State.loss(s().square())
            assert (State.loss() == torch.tensor(5.)).all()
        assert (s.initial.grad == torch.tensor([[10.]])).all()
    @run
    def test6():
        """`State.Setter` applications."""
        s = State(torch.zeros((1,1)))
        with State.Setter(lambda state, to: (state() + to) / 2):
            with State.Episode():
                s(s() + 1)
                assert (s() == torch.tensor([[.5]])).all()
                s(s() + 1)
                assert (s() == torch.tensor([[1.]])).all()
    @run
    def test7():
        """Saving and loading `State`s."""
        s = State(torch.zeros((1,1)))
        ep = State.Episode()
        import io
        file = io.BytesIO()
        with ep:
            s(s() + 1)
            s(s() + 2)
            torch.save(s, file)
        # print(file.getbuffer().nbytes, 'bytes')
        file.seek(0)
        s = torch.load(file)
        s(s() + 3)
        assert (s() == torch.tensor([[6.]])).all()
    # TODO: Also a test that creates a few SRWM-Transformer layers and trains them.
    print('Tests OK')