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
        x = shape if isinstance(shape, torch.Tensor) else torch.randn(shape, device=device, requires_grad=True)
        self.id = id(self) # Might expose OS-level details. And might collide between processes.
        self.initial = nn.parameter.Parameter(x)
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

        Optimizer steps should go *after* the episode, if using `State.loss(…)`.    
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
    """`SRWM(ins, outs=ins, device=None)`

    [Self-Referential Weight Matrix](https://arxiv.org/abs/2202.05780): a linear-time RNN-like alternative to self-attention, with meta-learning built-in.

    Use this in [Transformer](https://arxiv.org/abs/1706.03762) layers. (The sequence of input vectors has to be presented not in parallel, but one-by-one.)

    Use `State.Episode` to train this. Inputs should be tensors shaped either as `(1, ins)` or `(batch_size, ins)`.
    """
    def __init__(self, ins, outs=..., device=None):
        if outs is ...: outs = ins
        super().__init__()
        self.ins, self.outs = ins, outs
        self.split_sz = (outs, ins, ins, 1)
        self.W = State((ins, sum(self.split_sz)), device=device)
        #   (Here, just 1 global learning rate.)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(-1)
    def __call__(self, x):
        W = self.W()
        ins, outs = self.ins, self.outs
        si, sm = self.sigmoid, self.softmax
        assert x.shape[-1] == ins
        x = x.unsqueeze(-2) # TODO: …If we still do this, then there's no point to NOT already be implementing multihead SRWM, right…
        y, k, q, lr = torch.split(sm(x) @ W, self.split_sz, -1)
        vk, vq = sm(k) @ W, sm(q) @ W
        self.W(W + ((si(lr) * (vq - vk)).unsqueeze(-2) * sm(k).unsqueeze(-1)).squeeze(-3))
        return y.squeeze(-1)
class SRWM(nn.Module): # TODO:
    """`SRWM(ins, outs=ins, heads=1, device=None)`

    [Self-Referential Weight Matrix](https://arxiv.org/abs/2202.05780): a linear-time RNN-like alternative to self-attention, with meta-learning built-in.
    - `ins`, `outs`: sizes of input & output vectors.
    - `heads`: splits inputs into this many sub-tensors, operates on each independently, then reassembles.

    Use this in [Transformer](https://arxiv.org/abs/1706.03762) layers. (The sequence of input vectors has to be presented not in parallel, but one-by-one.)

    Use `State.Episode` to train this. Inputs should be tensors shaped either as `(1, ins)` or `(batch_size, ins)`.
    """
    def __init__(self, ins, outs=..., heads=1, device=None):
        assert ins % heads == 0, "Head-count must divide input-size; zero-pad the input or something"
        assert outs % heads == 0, "Head-count must divide output-size; slice the output or something"
        if outs is ...: outs = ins
        super().__init__()
        self.ins, self.outs, self.heads = ins, outs, heads
        self.split_sz = (outs//heads, ins//heads, ins//heads, 1)
        self.W = State((heads, ins//heads, sum(self.split_sz)), device=device)
        #   (Here, just 1 global learning rate.)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(-1)
    def __call__(self, x):
        W = self.W()
        ins, outs, h = self.ins, self.outs, self.heads
        si, sm = self.sigmoid, self.softmax
        assert x.shape[-1] == ins
        y, k, q, lr = torch.split(sm(_head(x,h)) @ W, self.split_sz, -1)
        vk, vq = sm(k) @ W, sm(q) @ W
        self.W(W + ((si(lr) * (vq - vk)).unsqueeze(-2) * sm(k).unsqueeze(-1)).squeeze(-3))
        #   TODO: …No way to compute many updates in parallel (but not in batch), right?… Like, maybe replace the outer product by matmul, where the currently-1 dimension is the update count…?
        #     (…Maybe we SHOULD do this after all, even if not covered in the paper, since computational efficiency may just take priority…)
        return _unhead(y, h)
def _head(x, h):
    return x.reshape(*x.shape[:-1], h, 1, x.shape[-1]//h)
def _unhead(y, h):
    assert y.shape[-3] == h
    return y.reshape(*y.shape[:-3], h * y.shape[-1])



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
    @run
    def test8():
        """Using `SRWM` for actual NN training.

        We learn to denoise samples (which is pretty-much equivalent to classification in realistic NNs, [due to neural collapse](https://arxiv.org/abs/2112.15121))."""
        import random
        class SkipConnection(nn.Module):
            def __init__(self, *fns): super().__init__();  self.fn = nn.Sequential(*fns)
            def forward(self, x): return x + self.fn(x)
        N = 32
        net = nn.Sequential(
            SRWM(N, N, heads=1), # TODO: …Why does this fail to learn anything… Even at 500, loss is still 800… Even though there *is* gradient, and `opt` DOES know…
            #   …Learning is at least *non-zero*… But still: why so non-existent?…
            #   …Even an autoencoder is too much to ask for…
            # nn.Linear(N, N),
            # SkipConnection(nn.Linear(N, N)),
            # # TODO: …Are we underperforming?… Absolutely:
            # #   1 linear layer reaches 20 at 5k…
            # #   2 linear layers reach 20 at 5k…
            # #   1 SRWM reaches 350 at 5k… (Loss at 0 is 500.)
            # #   SRWM+linear reach 150…120 at 2k…5k…
            # #   TODO: How can we find the bug?…
            # # TODO: …Compare the speeds of SRWM and Linear… …Like 3…5 times slower, with like 3…5 times more CPU utilization… …Can we optimize it… (…Multi-update formulation, maybe?…)
            # #   …Wait, was `torch.autograd.set_detect_anomaly(True)` the cause of slowness… …Not at all.
            # # SkipConnection(nn.ReLU(), nn.LayerNorm(N), SRWM(N, N, heads=2)),
            # SkipConnection(nn.ReLU(), nn.LayerNorm(N), nn.Linear(N, N)),
            # # SkipConnection(nn.ReLU(), nn.LayerNorm(N), nn.Linear(N, N)),
            # SkipConnection(nn.ReLU(), nn.LayerNorm(N), SRWM(N, N, heads=1)),
            # # SkipConnection(nn.ReLU(), nn.LayerNorm(N), nn.Linear(N, N)),
            # SkipConnection(nn.ReLU(), nn.LayerNorm(N), nn.Linear(N, N)),
        )
        S = State((1,32)) # TODO:
        optS = torch.optim.Adam(S.parameters(), lr=1e-3) # TODO:
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        classes, examples_per_class = 2, 2
        batch_sz = 16 # TODO: 16
        noise_magnitude = .2
        for minibatch in range(5000):
            # First give a few training examples, then the test examples.
            with State.Episode():
                cls = torch.randn(classes, batch_sz, N//2)
                def example(of_class=None):
                    if of_class is None: of_class = random.randrange(classes)
                    noise = noise_magnitude * torch.randn(batch_sz, N//2)
                    return cls[of_class] + noise, cls[of_class]
                for train in range(classes * examples_per_class):
                    net(torch.cat(example(), -1))
                for test in range(classes):
                    ex, cl = example(test)
                    cl_pred = net(torch.cat((ex, torch.zeros_like(cl)), -1))
                    State.loss((cl_pred[..., :N//2] - cl).square())
                print(minibatch, 'L2', State.loss().detach().cpu().numpy()) # TODO:
            # print('               ', [*net.modules()][2].initial.grad.abs().sum()) # TODO:
            # print('               ', S.initial.grad.abs().sum()) # TODO:
            opt.step();  opt.zero_grad(True)
            optS.step();  optS.zero_grad(True) # TODO:
    print('Tests OK')