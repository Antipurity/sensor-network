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
        """`with State.Episode(start_from_initial = True): ...`

        Represents one trajectory, unrolled so that learning can happen on it. Each `State` starts at its initial value within it.

        Optimizer steps should go *after* the episode, if using `State.loss(…)`.    
        Can be pre-created and re-entered many times. (But not recursively. For that, use normal re-entrance: leave all intermediate episodes, return to this one, then re-enter intermediates.)    
        `start_from_initial` can be `False` to make the episode act as a simple checkpoint within another episode, restoring `State`s when done.


        Encountered `State`s are not garbage-collected automatically. Use `ep.remove(state)` if that is really required.
        """
        __slots__ = ('state_obj', 'state_old', 'restorable', 'active', 'start_from_initial')
        def __init__(self, start_from_initial=True): # TODO: Replace `start_with_initial` bool with `initial(initial+0, current)` func. (So that we can pre-emptively detach.)
            assert isinstance(start_from_initial, bool)
            self.state_obj = {} # id → State
            self.state_old = {} # id → current
            self.restorable = set()
            self.active = False
            self.start_from_initial = start_from_initial
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

        # TODO: For convenience of implementing tBPTT, should have `.update(fn)` which goes through .state_obj and sets the value to fn(value). (`fn` could be `lambda x: x.detach()` for tBPTT.)
        # TODO: …For replay buffers, maybe have `.clone()`?…
        # TODO: …For save/load, maybe return just the `(.state_obj, .state_old)` tuple, assuming that `.restorable` is "all keys in .state_old"?…
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
                    if self.start_from_initial:
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

    Use this in [Transformer](https://arxiv.org/abs/1706.03762) layers.    
    (The sequence of input vectors has to be presented not in parallel, but one-by-one. Inputs should be tensors shaped either as `(1, ins)` or `(batch_size, update_count, ins)`; `update_count` can be used to perform many updates at once at the cost of correctness.)

    Use `State.Episode` to train this."""
    def __init__(self, ins, outs=..., heads=1, device=None):
        assert ins % heads == 0, "Head-count must divide input-size; zero-pad the input or something"
        assert outs % heads == 0, "Head-count must divide output-size; slice the output or something"
        if outs is ...: outs = ins
        h = heads
        super().__init__()
        self.ins, self.outs, self.heads = ins, outs, h
        self.split_sz = (ins//h, outs//h, ins//h, 1)
        self.slow = nn.parameter.Parameter(torch.randn(h, ins//h, sum(self.split_sz), device=device))
        self.fast = State((h, ins//h, outs//h), device=device)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(-1)
    def forward(self, x):
        ins, outs, h = self.ins, self.outs, self.heads
        si, sm = self.sigmoid, self.softmax
        assert x.shape[-1] == ins

        x = x.reshape(*x.shape[:-1], h, x.shape[-1]//h).transpose(-2,-3) # Per-head.
        k, v1, q, lr = torch.split(sm(x) @ self.slow, self.split_sz, -1)
        k, q = sm(k), sm(q)
        v2 = k @ self.fast()
        self.fast(self.fast() + si(lr) * (k.transpose(-2,-1) @ (v1 - v2)))
        # self.fast(self.fast() + si(lr) * v1 - si(lr) * v2) # TODO: What does this mess get us at 2k? .3 AT 600, .005 AT 2k?! And if we don't even use `k` at all, we get .02 at 600 and .0005 at 2k…
        #    What the hell is going on here?
        # TODO: Fuck around in Python REPL, to make sure we understand the update mechanism correctly: a random weight-matrix, and a key vector, and extract query, and subtract the mm-outer-product, and re-extract query (which should be 0).
        #   …`q @ (W - (q @ W).t() @ q)` doesn't give 0s (`q` is post-softmax, of course)… In fact, it hardly changes anything in the matrix… Was our impl wrong all along?…
        #     Has to be… But what's the correct version? Can't get actual outer products to work either… (The only ins!=outs shape-correct matmul-based version is `W - q.t() @ (q@W)`. But does its operation really make any sense? With matmul, can't we open the brackets — and then why are we multiplying q.t() with q?)
        #       Nothing's working.
        #       `q @ (W - q.t() @ q @ W)`, which FWP equations work out to, don't add up to 0.
        #       `q @ (W - q@W)` produces 0s. But how is this related to FWP equations? What is this? Aren't we just subtracting the value from *all* keys, not just one?
        # TODO: …But with FWP-equations, we're reaching .7 at 2k, which is kinda a win… Did we change the behavior? We did, didn't we: fixed the shapes… So which is correct: the FWP-equations, or our discovered-by-accident bullshit?…
        #   Our key-less value-only approach really wouldn't work if there are many keys. We really have to weigh by key. It's slower to learn, but it does still reach really low loss, so it's more general, not more wrong.
        y = q @ self.fast()
        return y.transpose(-2,-3).reshape(*y.shape[:-3], y.shape[-2], h * y.shape[-1]) # Concat per-head results.



class SRWM(nn.Module):
    """`SRWM(ins, outs=ins, heads=1, device=None)`

    A drop-in replacement for `DeltaNet`, with more "meta".

    [Self-Referential Weight Matrix](https://arxiv.org/abs/2202.05780): a linear-time RNN-like alternative to self-attention, with meta-learning built-in.
    - `ins`, `outs`: sizes of input & output vectors.
    - `heads`: splits inputs into this many sub-tensors, operates on each independently, then reassembles.

    Use this in [Transformer](https://arxiv.org/abs/1706.03762) layers.    
    (The sequence of input vectors has to be presented not in parallel, but one-by-one. Inputs should be tensors shaped either as `(1, ins)` or `(batch_size, update_count, ins)`; `update_count` can be used to perform many updates at once at the cost of correctness.)

    Use `State.Episode` to train this.
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
        # This is less efficient than https://github.com/IDSIA/modern-srwm/blob/main/reinforcement_learning/torchbeast/self_ref_v1/self_ref_v1.cu
        #   But are 10 lines more comprehensible than 1671?
        W = self.W()
        ins, outs, h = self.ins, self.outs, self.heads
        si, sm = self.sigmoid, self.softmax
        assert x.shape[-1] == ins

        x = x.reshape(*x.shape[:-1], h, x.shape[-1]//h).transpose(-2,-3) # Per-head.
        y, k, q, lr = torch.split(sm(x) @ W, self.split_sz, -1)
        vk, vq = sm(k) @ W, sm(q) @ W
        update = si(lr) * (sm(k).transpose(-2,-1) @ (vq - vk))
        self.W(W + update)
        # print('                          lr', si(lr).mean().detach().cpu().numpy()) # TODO: …LR does go up, to near-1…
        # print('                          W', W.abs().mean().detach().cpu().numpy(), '+', update.abs().mean().detach().cpu().numpy()) # TODO: …W also increases each parameter, and its stdev…
        #   …Also, why *are* updates about 7e-4 initially, 7e-3 at 5k?
        return y.transpose(-2,-3).reshape(*y.shape[:-3], y.shape[-2], h * y.shape[-1]) # Concat per-head results.



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
    # TODO: Separate the RNN-with-bits thing into a separate test.
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
            # #     `RNN` works flawlessly. So, `SRWM` is the bug here.
            # #     …Maybe state initialization is at fault — stdev=1 is too much for weights?…
            # # TODO: …Compare the speeds of SRWM and Linear… …Like 3…5 times slower, with like 3…5 times more CPU utilization… …Can we optimize it… (…Multi-update formulation, maybe?…)
            # #   TODO: Try `torch.baddbmm` for updating the weight matrix.
            # # SkipConnection(nn.ReLU(), nn.LayerNorm(N), SRWM(N, N, heads=2)),
            SkipConnection(nn.ReLU(), nn.LayerNorm(N), nn.Linear(N, N)),
            # # SkipConnection(nn.ReLU(), nn.LayerNorm(N), nn.Linear(N, N)),
            # SkipConnection(nn.ReLU(), nn.LayerNorm(N), SRWM(N, N, heads=1)),
            # # SkipConnection(nn.ReLU(), nn.LayerNorm(N), nn.Linear(N, N)),
            # SkipConnection(nn.ReLU(), nn.LayerNorm(N), nn.Linear(N, N)),
        )
        net = RNN(N,
            nn.Linear(N+N, N+N),
            SkipConnection(nn.ReLU(), nn.LayerNorm(N+N), nn.Linear(N+N, N+N)),
        )
        net = nn.Sequential( # TODO: …Why is even DeltaNet unable to learn bits, reaching L2 of 4.4 per batch (VERY slowly, 7.5 at 100) whereas `RNN` reaches 1e-5?…
            # …But, can't deny that it *is* making *some* progress on the task with time…
            # Maybe the additive nature of the matrix-update is working against us in this task. (`RNN` *did* perform worse with skip-connections all the way through, after all.)
            #   More or less (batch_sz=256):
            #     Without addition in `DeltaNet`: 6 at 600, 6 at 2k.
            #     With addition: 7 at 2k.
            #   Removing LR gating too:
            #     Without addition: 5.2 at 600, 0.6 at 1k, 0.1 at 2k. (Solved.)
            #     With addition: 9 at 600, 7 at 1k, 5 at 2k.
            nn.Linear(N, N),
            SkipConnection(nn.ReLU(), nn.LayerNorm(N), DeltaNet(N, N, heads=2)), # TODO: heads=2… Well, it runs, so not much to say, right? …Except, we're severely underperforming, though if we enable weight-addition, we're underperforming less…
            SkipConnection(nn.ReLU(), nn.LayerNorm(N), nn.Linear(N, N)),
        )
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        classes, examples_per_class = 2, 2
        batch_sz = 16
        noise_magnitude = .2
        for minibatch in range(5000):
            with State.Episode():
                # Bit-env: just recall the previous one-bit input.
                def example():
                    cond = torch.rand(batch_sz,1,1) < .5
                    bit = torch.ones(batch_sz,1,1)
                    bit = torch.where(cond, bit, -bit)
                    return torch.cat((bit, torch.zeros(batch_sz,1,N-1)), -1)
                prev = example()
                net(prev)
                for _ in range(12):
                    next = example()
                    pred = net(next)
                    if minibatch>4000: print(prev[0,0,0].numpy(), pred[0,0,0].detach().numpy()) # TODO: …When seeing sequences of bits, prediction only *gradually* goes in the correct direction: a bit receptive, but not very…
                    State.loss((pred[..., 0] - prev[..., 0]).square())
                    prev = next
                print(minibatch, 'L2', (State.loss()/batch_sz).detach().cpu().numpy()) # TODO:
            # First give a few training examples, then the test examples.
            # with State.Episode():
            #     cls = torch.randn(classes, batch_sz, 1, N//2)
            #     def example(of_class=None):
            #         if of_class is None: of_class = random.randrange(classes)
            #         noise = noise_magnitude * torch.randn(batch_sz, 1, N//2)
            #         return cls[of_class] + noise, cls[of_class]
            #     for train in range(classes * examples_per_class):
            #         net(torch.cat(example(), -1))
            #     for test in range(classes):
            #         ex, cl = example(test)
            #         cl_pred = net(torch.cat((ex, torch.zeros_like(cl)), -1))
            #         State.loss((cl_pred[..., :N//2] - cl).square())
            #     print(minibatch, 'L2', State.loss().detach().cpu().numpy()) # TODO:
            # print('               ', [*net.modules()][2].initial.grad.abs().sum()) # TODO:
            # print('               ', S.initial.grad.abs().sum()) # TODO:
            opt.step();  opt.zero_grad(True)
    print('Tests OK')