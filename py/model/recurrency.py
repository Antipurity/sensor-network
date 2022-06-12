"""
Code that simplifies working with recurrency (`state → f(state)`).

In particular, this module contains the `State` framework, and the PyTorch NN modules `RNN` and `DeltaNet` and `SRWM` that use that framework.
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

    (What's the difference between RNN states and NN weights, after all? Only a pre-programmed optimizer for the latter, and a learned one (if any) for the former. And PyTorch already makes NN weights hidden parameters.)
    """
    # __slots__ = ('id', 'initial', 'current')
    #   `torch.load` complains if we follow these best practices, since it gives us the props of `nn.Module`s.
    _setters = []
    _losses = [0.]
    _episodes = []
    def __init__(self, shape, device=None):
        super().__init__()
        scale = 1. if isinstance(shape, torch.Tensor) else shape[-2]**-.5 if len(shape)>1 else 1.
        x = shape if isinstance(shape, torch.Tensor) else torch.randn(shape, device=device) * scale
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
        """`with State.Episode(start_from_initial = True, remember_on_exit = True) as ep: ...`

        Represents one trajectory, unrolled so that learning can happen on it. Each `State` starts at its initial value within it.

        `optimizer.step()` should go *after* the episode, if using `State.loss(…)`.    
        Can be pre-created and re-entered many times. (But not recursively. For that, use normal re-entrance: leave all intermediate episodes, return to this one, then re-enter the intermediates.)

        Params:
        - `start_from_initial`: can be `False` to make the episode act as a simple checkpoint within another episode, restoring `State`s when done; or can be `lambda initial, current: current.detach()` to act like `False` but also modify the initial states.
        - `remember_on_exit`: if `False`, re-entering the episode will act like it was never entered, instead of continuing from where it left off.

        Methods:
        - `ep.update(lambda initial, current: current.detach())`: for TBPTT.
        - `ep.clone(start_from_initial = True)`: for preserving the current RNN state.
        - `ep.remove(state)`: encountered `State`s are not garbage-collected automatically, so use this if required.
        """
        __slots__ = ('state_obj', 'state_old', 'restorable', 'active', 'start_from_initial', 'remember_on_exit')
        def __init__(self, start_from_initial=True, remember_on_exit=True):
            assert isinstance(start_from_initial, bool) or callable(start_from_initial)
            assert isinstance(remember_on_exit, bool)
            self.state_obj = {} # id → State
            self.state_old = {} # id → current
            self.restorable = set()
            self.active = False
            self.start_from_initial = start_from_initial
            self.remember_on_exit = remember_on_exit
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
            # Reset all values to pre-entry states
            objs, olds = self.state_obj, self.state_old
            for id, s in self.state_obj.items():
                if self.remember_on_exit:
                    s.current, olds[id] = olds[id], s.current
                else:
                    s.current = olds[id]
            if self.remember_on_exit:
                self.restorable.update(objs.keys())

        def __getstate__(self):
            if self.active:
                return self.state_obj, {id:s() for id,s in self.state_obj}, self.start_from_initial is True
            else:
                return self.state_obj, self.state_old, self.start_from_initial is True
        def __setstate__(self, state):
            self.state_obj, self.state_old, self.start_from_initial = state
            self.restorable = set(self.state_obj.keys())
            self.active = False

        def update(self, fn):
            """Applies `fn(initial, current) → current` to every RNN-state tensor.

            [TBPTT](https://www.gabormelli.com/RKB/Truncated_Back-Propagation_Through_Time_(TBPTT)_Algorithm) can be implemented by sometimes doing `State.loss(True).backward();  optimizer.step();  optimizer.zero_grad(); episode.update(lambda _,x: x.detach())`.

            (If the episode is currently active, this calls `State.Setter`s.)"""
            for id, s in self.state_obj.items():
                if self.active:
                    s(fn(s.initial+0, s()))
                else:
                    self.state_old[id] = fn(self.state_old[id])
            return self
        def clone(self, start_from_initial=True, remember_on_exit=True):
            """Copies the current RNN state."""
            ep = State.Episode(start_from_initial, remember_on_exit)
            for id, s in self.state_obj.items():
                if self.active: self._register(s)
                current = s.current if self.active else self.state_old[id]
                ep.state_obj[id] = s
                ep.state_old[id] = current
            ep.restorable.update(self.state_obj.keys())
            return ep
        def remove(self, s):
            """Provided instead of any garbage collection: if any `State` object will not be used anymore, then just remove it."""
            del self.state_obj[s.id]
            del self.state_old[s.id]
            if s.id in self.restorable: self.restorable.remove(s.id)
            return self

        def _register(self, s):
            """Registers a `State` object in this episode, updating its `.current` value if needed."""
            assert self.active
            id, objs, olds, rest = s.id, self.state_obj, self.state_old, self.restorable
            if id not in rest:
                if id not in objs:
                    objs[id] = s
                    olds[id] = s.current
                    if callable(self.start_from_initial):
                        s.current = self.start_from_initial(s.initial+0, s.current)
                    elif self.start_from_initial:
                        s.current = s.initial+0 # `+0` to not treat this as an assignable Parameter.
                else:
                    return # (Yes, this branch does nothing.)
            else:
                rest.remove(id)
                olds[id], s.current = s.current, olds[id]

    class Setter:
        """`with State.Setter(lambda state, to: to): ...`

        Within this context, `State` updates will go through the specified function first, outermost-setter first.    
        To not change `to`, return `to` or `None`.

        Example uses:
        - `lambda state, to: state.initial+0 if random.randint(1,100)==1 else to`: hard-reset.
        - `lambda state, to: state.initial*.001 + .999*to`: soft-reset, for very-long episodes.
        - `lambda state, to: State.loss((state() - to.detach()).square())`: predict updates and maximize [mutual information](https://en.wikipedia.org/wiki/Directed_information), for extracting good features and skipping updates.
        """
        __slots__ = ('fn',)
        def __init__(self, fn): self.fn = fn
        def __enter__(self): State._setters.append(self.fn)
        def __exit__(self, type, value, traceback): State._setters.pop()



class RNN(nn.Module):
    """RNN: a neural net, with a state vector appended at the front of input & output.

    Has `State`-based state, so don't manage state variables explicitly, and do train it inside `State.Episode`s."""
    __slots__ = ('state', 'sz', 'fn')
    def __init__(self, state_size, *fns, device=None):
        super().__init__()
        assert isinstance(state_size, int)
        self.state = State((state_size,), device)
        self.sz = state_size
        self.fn = fns[0] if len(fns) == 1 else nn.Sequential(*fns)
    def forward(self, *ins):
        """Concatenates RNN-state and inputs, and executes functions."""
        x = torch.cat([self.state().expand(ins[0].shape), *ins], -1)
        y = self.fn(x)
        self.state(y[..., :self.sz])
        return y[..., self.sz:]



class DeltaNet(nn.Module):
    """`DeltaNet(ins, outs=ins, heads=1, device=None)`

    [DeltaNet](https://arxiv.org/abs/2102.11174): a linear-time RNN-like alternative to self-attention, with some meta-learning built-in.
    - `ins`, `outs`: sizes of input & output vectors.
    - `heads`: splits inputs into this many sub-tensors, operates on each independently, then reassembles.
    - `device`: important to pass in; `model.to(device)` won't work due to `State`.

    Use this in [Transformer](https://arxiv.org/abs/1706.03762) layers.    
    (The sequence of input vectors has to be presented not in parallel, but one-by-one. Inputs should be tensors shaped either as `(1, ins)` or `(batch_size, update_count, ins)`; `update_count` can be used to perform many updates at once at the cost of correctness.)

    Use `State.Episode` to train this."""
    def __init__(self, ins, outs=..., heads=1, device=None, Softmax=nn.Softmax):
        assert ins % heads == 0, "Head-count must divide input-size; zero-pad the input or something"
        assert outs % heads == 0, "Head-count must divide output-size; slice the output or something"
        if outs is ...: outs = ins
        h = heads
        super().__init__()
        self.ins, self.outs, self.heads = ins, outs, h
        self.split_sz = (ins//h, outs//h, ins//h, 1)
        from itertools import accumulate
        offs = tuple(accumulate(self.split_sz))
        self.slices = tuple((..., slice(offs[i-1] if i>0 else 0, offs[i])) for i in range(4))
        self.slow = nn.parameter.Parameter(torch.randn(h, ins//h, sum(self.split_sz), device=device))
        self.fast = State((h, ins//h, outs//h), device=device)
        self.sigmoid = nn.Sigmoid()
        self.softmax = Softmax(-1)
    def forward(self, x):
        assert len(x.shape) >= 2
        h, si, sm = self.heads, self.sigmoid, self.softmax

        W = self.fast()
        x = x.reshape(*x.shape[:-1], h, x.shape[-1]//h).transpose(-2,-3) # Per-head.
        full, sl = sm(x) @ self.slow, self.slices
        k, v1, q, lr = full[sl[0]], full[sl[1]], full[sl[2]], full[sl[3]]
        k, q = sm(k), sm(q)
        v2 = k @ W
        self.fast(W + si(lr) * k.transpose(-2,-1) @ (v1 - v2))
        y = q @ W
        return y.transpose(-2,-3).reshape(*y.shape[:-3], y.shape[-2], h * y.shape[-1]) # Concat per-head results.



class SRWM(nn.Module):
    """`SRWM(ins, outs=ins, heads=1, device=None)`

    [Self-Referential Weight Matrix](https://arxiv.org/abs/2202.05780): a drop-in replacement for `DeltaNet`, with DeltaNet only having 1 "meta" layer (fast & slow weight matrices), whereas this has ∞, while being just as fast to run.

    See `DeltaNet` for params & docs.
    """
    def __init__(self, ins, outs=..., heads=1, device=None, Softmax=nn.Softmax):
        assert ins % heads == 0, "Head-count must divide input-size; zero-pad the input or something"
        assert outs % heads == 0, "Head-count must divide output-size; slice the output or something"
        if outs is ...: outs = ins
        super().__init__()
        self.ins, self.outs, self.heads = ins, outs, heads
        self.split_sz = (outs//heads, ins//heads, ins//heads, 1)
        from itertools import accumulate
        offs = tuple(accumulate(self.split_sz))
        self.slices = tuple((..., slice(offs[i-1] if i>0 else 0, offs[i])) for i in range(4))
        self.W = State((heads, ins//heads, sum(self.split_sz)), device=device)
        #   (Here, just 1 global learning rate, unlike in the paper.)
        self.sigmoid = nn.Sigmoid()
        self.softmax = Softmax(-1)
    def __call__(self, x):
        # This is less efficient than https://github.com/IDSIA/modern-srwm/blob/main/reinforcement_learning/torchbeast/self_ref_v1/self_ref_v1.cu
        #   But are these 10 lines more comprehensible than those 1671?
        assert len(x.shape) >= 2
        h, si, sm = self.heads, self.sigmoid, self.softmax

        W = self.W()
        x = x.reshape(*x.shape[:-1], h, x.shape[-1]//h).transpose(-2,-3) # Per-head.
        full, sl = sm(x) @ W, self.slices
        y, k, q, lr = full[sl[0]], full[sl[1]], full[sl[2]], full[sl[3]]
        k, q, lr = sm(k), sm(q), si(lr)
        kt = k.transpose(-2,-1)
        update = kt @ ((lr * (q - k)) @ W) if x.shape[-2] < self.ins else kt @ (lr * (q - k)) @ W
        self.W(W + update)
        #   (Note: for more speed, `torch.baddbmm(W, m1, m2)` *could* be used, but reshaping around that is harder to implement, unless `len(x.shape)==2`. Might want to consider & measure in the future.)
        return y.transpose(-2,-3).reshape(*y.shape[:-3], y.shape[-2], h * y.shape[-1]) # Concat per-head results.



if __name__ == '__main__': # pragma: no cover
    def run(f): f()
    class SkipConnection(nn.Module):
        def __init__(self, *fns): super().__init__();  self.fn = nn.Sequential(*fns)
        def forward(self, x): return x + self.fn(x)
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
    @run
    def test8():
        """`SRWM` is literally 50 times faster with parallel queries. (At least partially because it's CPU-bound.)"""
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        def env(name, net, ins):
            import time;  start = time.monotonic()
            groups = 100
            with torch.no_grad():
                for _ in range(groups):
                    with State.Episode():
                        for x in ins:
                            net(x)
            if torch.cuda.is_available():
                torch.cuda.synchronize(dev)
            end = time.monotonic()
            print(name, (end - start) / groups, 'sec/call')
        N = 256
        net = nn.Sequential(
            nn.Linear(N, N),
            *[nn.Sequential(
                SkipConnection(nn.ReLU(), nn.LayerNorm(N), SRWM(N, N, heads=2, device=dev)),
                SkipConnection(nn.ReLU(), nn.LayerNorm(N), nn.Linear(N, N)),
            ) for _ in range(20)],
        ).to(dev)
        env('64-in-one', net, [torch.randn(64, N, device=dev)])
        env('One-after-another', net, [torch.randn(1, N, device=dev) for _ in range(64)])
    @run
    def test9():
        """`RNN`, `DeltaNet`, `SRWM` can all learn to remember state."""
        N, batch_sz = 32, 16
        def example():
            """The bit to remember, zero-padded to `N`."""
            cond = torch.rand(batch_sz,1,1) < .5
            bit = torch.ones(batch_sz,1,1)
            bit = torch.where(cond, bit, -bit)
            return torch.cat((bit, torch.zeros(batch_sz,1,N-1)), -1)
        def env(name, net):
            """Bit-remembering: just learn to output the prev-iteration bit."""
            opt = torch.optim.Adam(net.parameters(), lr=1e-3)
            for minibatch in range(3000):
                with State.Episode():
                    next = example()
                    net(next)
                    for _ in range(12):
                        prev, next = next, example()
                        pred = net(next)
                        State.loss((pred - prev).square())
                    loss = State.loss().detach() / batch_sz
                    if minibatch > 500 and (loss < .5).all():
                        return minibatch
                opt.step();  opt.zero_grad()
            assert False, name+" does not converge"
        import time;  start = time.monotonic()
        n = env('RNN', RNN(N,
            nn.Linear(N+N, N+N),
            SkipConnection(nn.ReLU(), nn.LayerNorm(N+N), nn.Linear(N+N, N+N)),
        ))
        print('RNN', (time.monotonic() - start) / n, 'sec/minibatch');  start = time.monotonic()
        n = env('DeltaNet', nn.Sequential(
            nn.Linear(N, N),
            SkipConnection(nn.ReLU(), nn.LayerNorm(N), DeltaNet(N, N, heads=1)),
            SkipConnection(nn.ReLU(), nn.LayerNorm(N), nn.Linear(N, N)),
        ))
        print('DeltaNet', (time.monotonic() - start) / n, 'sec/minibatch');  start = time.monotonic()
        n = env('SRWM', nn.Sequential(
            nn.Linear(N, N),
            SkipConnection(nn.ReLU(), nn.LayerNorm(N), SRWM(N, N, heads=1)),
            SkipConnection(nn.ReLU(), nn.LayerNorm(N), nn.Linear(N, N)),
        ))
        print('SRWM', (time.monotonic() - start) / n, 'sec/minibatch');  start = time.monotonic()
    print('Tests OK')