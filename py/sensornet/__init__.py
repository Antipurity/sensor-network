"""
Bidirectional-sensor networks for ML: each gathers named numeric data and queries from anywhere, and in a loop, handles that.

Perfectly flexible, [efficient](https://github.com/Antipurity/sensor-network/tree/master/py/sensornet/test.py), [tiny](https://github.com/Antipurity/sensor-network/tree/master/py/sensornet/__init__.py), and [with 100% test coverage](https://github.com/Antipurity/sensor-network/tree/master/py/sensornet/test.py).

Position-invariant and numeric: these constraints free AI models from all concerns about data formats. With only a couple lines of code, you can: learn in completely arbitrary environments, combine digital and analog inputs/outputs, combine tasks at runtime, and perform any machine-learning tasks with a single model (multimodal learning and reinforcement learning and meta-learning and lifelong learning and model distillation; though need a *flawless* implementation that easily handles everything `sn` can bring to bear).

Needs Python 3.5 or newer (mainly for `asyncio` with `await`).

## Installation

```bash
pip install sensornet (TODO: Publish to PyPi.)
```

Or, like, copy this directory, and import it.

Or `pip install -ve .` here, see where your `easy-install.pth` lives, and edit that file to remove the last directory in the path to here.

## Usage

Simple.

First, initialize the handler:

```python
import sensornet as sn
import numpy as np

sn = sn.Handler(8,8,8,8, 64) # Cell-shape: first name-parts, then data.
# OR, simply use the global `sn` as if it's a handler:
sn.cell_shape = (8,8,8,8, 64)
```

Then send/receive data:

```python
async def get():
    sn.set('sensor', 13, 32)
    assert (await sn.get('action', 32)) in range(32)
```

(Simple integer sending/receiving is shown here, but other datatypes are also available.)

And handle it:

```python
@sn.run
async def main():
    fb = None
    while True:
        data, query, error = await sn.handle(fb)
        fb = np.random.rand(query.shape[0], data.shape[1])*2-1
```

This module implements this basic discrete+analog protocol, and does not include anything [else](https://github.com/Antipurity/sensor-network/tree/master/docs/ROADMAP.md) by default, such as string handling or file storage or Internet communication or multiprocessing or integration with ML-libraries apart from PyTorch.

(Implementing a controllable programming language with forking and/or listenable-to data is left as an exercise to the reader.)

---

TODO: A small tour of the datatypes (`sn` just doesn't sound attractive otherwise), hopefully including `@sn.func   async def fn(img: sn.Float(28, 28, dims=2)) -> 10: ...` and/or `sn.Goal()`. And hopefully using an actual ML model to learn.
"""



import numpy
import hashlib
import asyncio
import functools



class Handler:
    """
    ```python
    Handler(8,8,8, 64)
    Handler(*cell_shape, info=None, sensors=None, on_feedback=None, backend=numpy, name_cache_size=1024)
    ```

    A bidirectional sensor network: gathers numeric data from anywhere, and in a loop, handles it, responding to queries with feedback.

    All data is split into fixed-size cells, each of which has numeric name parts (`cell_shape[:-1]`) and data (`cell_shape[-1]`). Handlers (AI models) should be position-invariant.

    Inputs:
    - `cell_shape`: a tuple, where the last number is how many data-numbers there are per cell, and the rest splits the name into parts; see `.name`. Can read/write `.cell_shape` as a property, whenever.
    - `info`: JSON-serializable immutable human-readable info about the used AI model and how to use it properly, and whatever else. To set global info on the `sn` module, do `sn.default.info = ...`.
    - `sensors`: `set` of functions that take this handler, to prepare data to handle via `sn.set(???)` and/or `sn.query(???)`.
    - `on_feedback`: allows easily transforming `.handle`'s feedback. In particular, to integrate with PyTorch, pass in `on_feedback = lambda x: sn.torch(torch, x)`.
    - `backend`: the NumPy object.
    - `name_cache_size`: info that is associated with names is cached for speed.

    If needed, wherever the handler object is available, read `.cell_shape` or `.backend`, or read/[modify](https://docs.python.org/3/library/stdtypes.html#set) `.sensors`. These values might change between sending and receiving feedback.
    """
    __slots__ = ('_s', '_data', '_query', '_error', '_prev_fb', '_next_fb', 'info', 'sensors', 'backend', 'name_cache_size', 'modify_name', '_str_to_floats', '_default', '_wait_until_submit', 'on_feedback')
    def __init__(self, *cell_shape, info=None, sensors=None, on_feedback=None, backend=numpy, name_cache_size=1024, _default=False):
        from builtins import set
        import json;  json.dumps(info) # Just for error-checking.
        sensors = set(sensors) if sensors is not None else set()
        s = self._s = _S()
        s.cell_shape, s.cell_size, s.n = (), 0, 0
        s.wait_for_requests = None # asyncio.Future()
        s.name = None
        s.shaped_names = None
        s.query_cell = 0
        self._data, self._query, self._error = [], [], []
        self._prev_fb = [] # [???, [prev_feedback, _next_fb, cell_count, cell_size], ???]
        self._next_fb = [] # [???, (on_feedback, start_cell, end_cell), ???]
        self.info = info
        self.sensors = sensors
        self.backend = backend
        self.name_cache_size = name_cache_size
        self.modify_name = ()
        self._str_to_floats = functools.lru_cache(name_cache_size)(functools.partial(_str_to_floats, backend))
        self._default = _default
        self._wait_until_submit = []
        self.on_feedback = on_feedback
        self.cell_shape = cell_shape
    def fork(self, modify_name):
        """`sn.fork(lambda name: tuple(['sub-env-name', *name]))`

        Creates an interface, usable for communicating data in a separate sub-'thread' denoted by different names. Data must still be handled by the non-forked version. Data is shared among all forked versions, so use `sn.Handler(???)` to not share data.

        `name` is a tuple, and does not include the extra name-part that `sn.Int` and `sn.Float` prepend."""
        assert callable(modify_name)
        h = self.Handler.__new__(self.Handler)
        for k in h.__slots__:
            setattr(h, k, getattr(self, k))
        h.modify_name = tuple([*self.modify_name, modify_name])
        return h

    def name(self, name):
        """`sn.name(name)`

        Converts a Python name such as `('image', (.25, .5))` to a NumPy template, of shape `(sum(cell_shape[:-1]),)`, with `nan`s wherever the value doesn't matter.

        Names are tuples of either strings (which are MD5-hashed with the 16 bytes converted to -1???1 numbers), `None`s, and/or tuples of either numbers or `None`s.

        All of these top-level name-parts take up 1 part each, so `cell_shape=(8,8,8,8,64)` can have up to 4 name-parts (or 3, since datatypes monopolize 1 name part), 8 numbers each."""
        assert isinstance(name, tuple)
        return self._s.name(name)
    @property
    def cell_size(self): return self._s.cell_size
    @property
    def cell_shape(self): return self._s.cell_shape
    @cell_shape.setter
    def cell_shape(self, cell_shape):
        _shape_ok(cell_shape)
        s = self._s
        if s.cell_shape == cell_shape:
            return self
        self.discard()
        s.cell_shape = cell_shape
        s.cell_size = sum(cell_shape)
        s.n = 0 # For `.wait(???)`, to ensure that the handling loop yields at least sometimes, without the overhead of doing it every time.
        # Also create the cached Python-name-to-NumPy-name method.
        s.name = functools.lru_cache(self.name_cache_size)(functools.partial(_name_template, self.backend, self._str_to_floats, cell_shape))
        @functools.lru_cache(self.name_cache_size)
        def shaped_names(cells, got_shape, max_shape, prefix, name):
            # Prepend `(*prefix, *shape_progress)` numbers to `name`s.
            assert len(got_shape) == len(max_shape)
            np, s = self.backend, self._s
            name_sz = (s.cell_size - s.cell_shape[-1]) if len(s.cell_shape) else 0
            if cells == 0 or not s.cell_size: # pragma: no cover
                return np.zeros((cells, name_sz), dtype=np.float32)
            full_name = np.resize(np.nan_to_num(self.name(tuple([None, *name]))), (cells, name_sz))
            # Progress in `cells`/`got_shape`-space is linearly-dependent on progress in `max_shape`-space.
            sz = np.prod(got_shape) if len(got_shape) else 1
            n, progress = np.linspace(0, sz, cells, dtype=np.float32), []
            for i in reversed(range(len(got_shape))):
                got, max = got_shape[i], max_shape[i]
                progress.append((n % got) / max * 2 - 1)
                n = np.floor_divide(n, got)
            progress = np.stack([np.full((cells,), 1. if p is True else -1. if p is False else p, dtype=np.float32) for p in prefix] + [*reversed(progress)], 1)
            full_name[:, :s.cell_shape[0]] = _fill(np, progress, s.cell_shape[0])
            return full_name
        s.shaped_names = shaped_names

    def set(self, name=None, data=None, type=None, error=None):
        """
        `sn.set(name, data, type=None, error=None)`

        Sends named data to the handler. Receives no feedback; see `.query` for that.

        Args:
        - `name`: see `.name`.
        - `data`: what to send.
        - `type`: how the data is interpreted, such as `sn.Int(8)`. Some types have implicit forms here for brevity, such as `8` which becomes `sn.Int(8)`; see docs of types.
        - `error = None`: `data` transmission error: `None` or numbers with the meaning `max abs(true_data - data)`. `sn.Int` cannot have error; `sn.Float` can.
        """
        self._maybe_default()
        np, s = self.backend, self._s
        if isinstance(name, str): name = (name,)

        if name is not None: type = _default_typing(type)
        if type is not None:
            assert hasattr(type, 'set')
            return type.set(self, name, data, error)
        assert name is None, "Either forgot the type, or meant to pass in unnamed 2D data"

        assert isinstance(data, np.ndarray)
        assert len(data.shape) == 2 and data.shape[-1] == s.cell_size
        if isinstance(error, float):
            error = np.full((data.shape[0], s.cell_shape[-1]), error, dtype=np.float32)
        assert error is None or isinstance(error, np.ndarray) and data.shape[0] == error.shape[0]

        if not s.cell_size: return
        assert error is None or error.shape[1] == s.cell_shape[-1]

        self._data.append(data)
        self._error.append(error)
        if s.wait_for_requests is not None:
            s.wait_for_requests.set_result(None)
            s.wait_for_requests = None
    def query(self, name=None, type=None, callback=None):
        """
        ```python
        await sn.query(name, type)
        sn.query(name, type, *, callback: lambda feedback: ...)
        ```

        From the handler, asks for feedback, or `None` (on transmission errors). To guarantee feedback, use `.get`.

        Args:
        - `name`: see `.set`.
        - `type`: what to receive, such as `sn.Int(8)`.
        - `callback = None`: if `await` has too much overhead, this could be a function that is given the feedback.
            - `.query` calls impose a global ordering, and feedback only arrives in that order, delayed. So to reduce memory allocations, could reuse the same function and use queues.
        """
        self._maybe_default()
        np, s = self.backend, self._s
        assert callback is None or isinstance(callback, asyncio.Future) or callable(callback)
        if isinstance(name, str): name = (name,)

        if name is not None: type = _default_typing(type)
        if type is not None and not isinstance(type, np.ndarray):
            assert hasattr(type, 'query')
            assert callback is None
            return type.query(self, name)
        query = type
        assert name is None, "Either forgot the type, or meant to pass in unnamed 2D data"

        if callback is None: callback = asyncio.Future()

        assert isinstance(query, np.ndarray) and len(query.shape) == 2

        if not s.cell_size:
            if callable(callback):
                return callback(None)
            else:
                callback.set_result(None)
                return callback
        assert query.shape[-1] == s.cell_size-s.cell_shape[-1]

        self._query.append(query)
        cells = query.shape[0]
        self._next_fb.append((callback, s.query_cell, s.query_cell + cells))
        s.query_cell += cells
        if s.wait_for_requests is not None:
            s.wait_for_requests.set_result(None)
            s.wait_for_requests = None
        if isinstance(callback, asyncio.Future):
            return callback
    def get(self, name, type):
        """
        `await sn.get(name, type)`

        Gets feedback, guaranteed. Never returns `None`, instead re-querying until a result is available.
        """
        self._maybe_default()
        if isinstance(name, str): name = (name,)
        if name is not None: type = _default_typing(type)
        if hasattr(type, 'get'):
            return type.get(self, name)
        goal = Handler.Goal.goal
        async def query_until_not_none(fb):
            while True: # Minimize re-requests at the cost of latency.
                fb = await fb
                if fb is not None: return fb
                prev, Handler.Goal.goal = Handler.Goal.goal, goal
                try: fb = self.query(name, type)
                finally: Handler.Goal.goal = prev
        return query_until_not_none(self.query(name, type))
    def pipe(self, data, query, error, callback=None):
        """
        `prev_feedback = await other_handler.pipe(*sn.handle(prev_feedback))`

        Makes another handler handle this packet. Useful if NumPy arrays have to be transferred manually, such as over the Internet.

        A note on efficiency: don't `await h.pipe(???)` but do `await sn.submit()` after piping; to send back the feedback without stalling, do `sn.run(async_func, h.pipe(???))`.
        """
        self._maybe_default()
        if data is not None: self.set(None, data, None, error)
        if query is not None: return self.query(None, query, callback)
    def submit(self):
        """`await sn.submit()`: waits until the current data packet is submitted (which happens when `sn.handle` returns). Use this to set/query streams of data without stalling."""
        ft = asyncio.Future()
        self._wait_until_submit.append(ft)
        return ft
    def handle(self, prev_feedback=None, max_simultaneous_steps=16):
        """
        ```python
        await sn.handle(prev_feedback=None)
        await sn.handle(prev_feedback=None, max_simultaneous_steps=16)
        sn.handle(prev_feedback, None)
        ```

        Handles collected data.

        Pass it the previous handling's feedback: as a NumPy array sized `M??cell_size` or `None` or an `await`able future of that, or a low-level function (takes nothing, returns `False` to wait, `None` to drop, an array to respond).

        If `max_simultaneous_steps` is `None`, there is no waiting, only immediate data/query, possibly empty. Otherwise, this waits until there is some data/query, and until we aren't handle too many steps at once.

        This returns `(data, query, error)`, or an `await`able promise of that.
        - `data`: float32 arrays of already-named cells of data, sized `N??cell_size`.
        - `query`: same, but sized `M??name_size` (only the name).
        - `error`: data transmission error: `None` or a `(cells, cell_shape[-1])`-sized float32 array of `abs(true_data - data)`.

        What can be done with the result:
        - Be ready for data from untrusted sources: `data = numpy.clip(numpy.nan_to_num(data), -1., 1.)`.
        - Blur out the precision that was lost during transmission: `if error is not None: data[:, -sn.cell_shape[-1]:] += error * (numpy.random.rand(*error.shape)*2-1)`
        - Extract goal-cells: `data[data[:, 0] > 0]`
        - Extract analog (`sn.Float`) cells: `data[data[:, 1] > 0]`
        - Extract digital (`sn.Int`) cells: `data[~(data[:, 1] > 0)]`
        """
        assert not len(self.modify_name), "Only non-forked handlers can handle data"
        self._maybe_default()
        np = self.backend
        prev_feedback = self.on_feedback(prev_feedback) if callable(self.on_feedback) else prev_feedback
        if asyncio.iscoroutine(prev_feedback) and not isinstance(prev_feedback, asyncio.Future):
            prev_feedback = asyncio.create_task(prev_feedback) if hasattr(asyncio, 'create_task') else asyncio.ensure_future(prev_feedback)
        assert prev_feedback is None or isinstance(prev_feedback, np.ndarray) or isinstance(prev_feedback, asyncio.Future) or callable(prev_feedback)
        assert max_simultaneous_steps is None or isinstance(max_simultaneous_steps, int) and max_simultaneous_steps > 0
        # Collect sensor data.
        for s in self.sensors: s(self)
        # Remember to respond to the previous step with prev_feedback.
        if len(self._prev_fb):
            if isinstance(prev_feedback, np.ndarray):
                assert prev_feedback.shape[0] == self._prev_fb[-1][2], "Cell-count is wrong"
            self._prev_fb[-1][0] = prev_feedback
        else:
            assert prev_feedback is None, 'The first step cannot give feedback to its previous step'
        # Respond to what we can.
        try:
            while len(self._prev_fb):
                feedback, callbacks, cell_count, cell_size = self._prev_fb[0]
                if isinstance(feedback, asyncio.Future):
                    if not feedback.done(): break
                    feedback = feedback.result()
                else:
                    if callable(feedback):
                        feedback = feedback()
                    if feedback is False: break # Respond in-order, waiting if `False`.
                assert feedback is None or isinstance(feedback, np.ndarray)
                if feedback is not None:
                    assert len(feedback.shape) == 2, "Feedback shape is wrong"
                    assert feedback.shape[0] == cell_count, "Cell-count is wrong"
                    assert feedback.shape[1] == cell_size, "Cell-size is wrong"
                self._prev_fb.pop(0)
                _feedback(callbacks, feedback)
        except:
            self.discard()
            raise
        if max_simultaneous_steps is None:
            return self._take_data()
        else:
            return self._wait_then_take_data(max_simultaneous_steps)
    def commit(self):
        """`sn.commit()`: actually copies the provided data, allowing raw NumPy arrays to be written-to. Irrelevant if such overwriting never happens, or if raw-mode (`sn.set(data=rand(cells,sum(cell_shape)))` or `sn.query(type=rand(cells,cell_shape[-1]))` or `sn.pipe`) is not used."""
        assert not len(self.modify_name), "Only non-forked handlers can handle data"
        np, s = self.backend, self._s
        if len(self._data) == 1 and len(self._query) == 1: return
        values = s.cell_shape[-1] if len(s.cell_shape) else 0
        L1, L2 = s.cell_size, s.cell_size - values
        data = np.concatenate(self._data, 0) if len(self._data) else np.zeros((0, L1), dtype=np.float32)
        query = np.concatenate(self._query, 0) if len(self._query) else np.zeros((0, L2), dtype=np.float32)
        error = _concat_error(self._data, self._error, values, np)
        self._data.clear();  self._data.append(data)
        self._query.clear();  self._query.append(query)
        self._error.clear();  self._error.append(error)
    def discard(self):
        """Clears all scheduled-to-be-sent data."""
        assert not len(self.modify_name), "Only non-forked handlers can handle data"
        try:
            _feedback(self._next_fb, None)
        finally:
            self._s.query_cell = 0
            self._data.clear()
            self._query.clear()
            self._error.clear()
            self._next_fb.clear()
    def _take_data(self):
        """Gather data, queries, and their errors, and return them."""
        self.commit()
        data = self._data[0];  self._data.clear()
        query = self._query[0];  self._query.clear()
        error = self._error[0];  self._error.clear()
        self._prev_fb.append([False, self._next_fb, query.shape[0], self._s.cell_size])
        self._next_fb = []
        self.discard()
        for ft in self._wait_until_submit: ft.set_result(None)
        self._wait_until_submit.clear()
        return data, query, error
    async def _wait_then_take_data(self, max_simultaneous_steps = 16):
        """Limits how many steps can be done at once, yielding if necessary (if no data and no queries, or if it's been too long since the last yield). Particularly important for async feedback: if the handling loop never yields to other tasks, then they cannot proceed, and a deadlock occurs (and memory eventually runs out)."""
        assert isinstance(max_simultaneous_steps, int) and max_simultaneous_steps > 0
        s = self._s
        if not len(self._data) and not len(self._query):
            s.wait_for_requests = asyncio.Future()
            await s.wait_for_requests
        s.n += 1
        if s.n >= max_simultaneous_steps: s.n = 0
        if s.n == 0:
            await asyncio.sleep(0)
        if len(self._prev_fb) > max_simultaneous_steps:
            fb = self._prev_fb[0][0] # The oldest feedback, must be done.
            if isinstance(fb, asyncio.Future) and not fb.done():
                await fb
            elif callable(fb): # pragma: no cover
                while True:
                    r = fb()
                    if r is not False:
                        self._prev_fb[0][0] = r
                        break
                    await asyncio.sleep(.003)
        return self._take_data()
    def _maybe_default(self):
        """If we're the top-level sensor, we have to react to changes in the global `cell_shape`."""
        global cell_shape, cell_size, info
        if self._default:
            self.cell_shape = cell_shape # global ??? us
            cell_shape, cell_size = self._s.cell_shape, self._s.cell_size # us ??? global
            self.info = info # global ??? us

    class Event:
        """
        `None` or `sn.Event()`: a datatype with no data, which can be set at important points in a program, either as a notification like `sn.set(name)` or as a goal like `sn.set(name, type=sn.Goal())`.
        """
        __slots__ = ()
        def __repr__(self): return 'sn.Event()'
        def set(self, sn, name, data, error):
            assert data is None and error is None
            has_floats = sn.info is None or 'analog' in sn.info and sn.info['analog']
            return sn.set(name, [0.], sn.Float(1)) if has_floats else sn.set(name, 0, sn.Int(2))
    class Int:
        """
        ```py
        8
        [8] * 15
        sn.Int(15, 8)
        sn.Int(*[*shape, options])
        ```

        Datatype: a sequence of integers, each `0???options-1`. `shape` is the upper limit on each of the data's dimensions.

        To use this, `Handler`s need to specify `info={'choices_per_cell': opts}`. This datatype will take care of conversions.

        For efficiency, where possible, request many `Int`s via `shape`, and request powers-of-2 `options`. (We favor modularity over mathematically-optimal allocation.)

        `sn.Int` and `sn.IntFloat` are less efficient but more precise than `sn.Float` (due to representation and how handlers can model data), similarly to digital vs analog signals.
        """
        __slots__ = ('sz', 'shape', 'opts')
        def __init__(self, *shape):
            s, o = shape[:-1], shape[-1]
            assert isinstance(s, tuple) and all(isinstance(n, int) and n>0 for n in s)
            assert isinstance(o, int) and o>1, "Only makes sense to choose between 2 or more options"
            from operator import mul
            sz = functools.reduce(mul, s, 1)
            self.sz, self.shape, self.opts = sz, s, o
        def __repr__(self):
            return 'sn.Int(' + ','.join(repr(s) for s in [*self.shape, self.opts]) + ')'
        def set(self, sn, name, data, error):
            for fn in sn.modify_name: name = fn(name)
            assert error is None, "Integers are precise"
            assert not isinstance(data, list) and not isinstance(data, tuple) or all(x is not None for x in data), "`sn.List` does not support `None`s, so create an `sn.List` explicitly"
            np = sn.backend
            data = np.array(data, dtype=np.int32, copy=False)
            data = sn.Float.expand_shape(np, data, self.shape)
            cur_shape = data.shape
            data = data.reshape(self.sz)
            assert ((data >= 0) & (data < self.opts)).all(), "Out-of-range ints"

            # Assemble & set cells.
            cpc = sn.info['choices_per_cell']
            from math import frexp;  bpc = frexp(cpc - 1)[1]
            shape = sn._s.cell_shape
            if not len(shape): return
            assert shape[-1] >= bpc
            cells = sn.Int.repack(sn, self.sz, self.opts, cpc)
            prefix = (Handler.Goal.goal, False, *sn.List.extra_prefix)
            names = sn._s.shaped_names(cells, cur_shape, self.shape, prefix, name)
            data = sn.Int.repack(sn, data, self.opts, cpc)
            data = sn.Int.encode_ints(np, np.expand_dims(data, -1), bpc)
            zeros = np.zeros((cells, shape[-1] - bpc), dtype=np.float32)
            cells = np.concatenate((names, data, zeros), -1)
            sn.set(None, cells)
        def query(self, sn, name):
            for fn in sn.modify_name: name = fn(name)
            cpc = sn.info['choices_per_cell']
            from math import frexp;  bpc = frexp(cpc - 1)[1]
            shape = sn._s.cell_shape
            assert not len(shape) or shape[-1] >= bpc
            cells = sn.Int.repack(sn, self.sz, self.opts, cpc) if len(shape) else 0
            prefix = (Handler.Goal.goal, False, *sn.List.extra_prefix)
            names = sn._s.shaped_names(cells, self.shape, self.shape, prefix, name) if cells else None
            async def do_query(fb):
                if not cells: return
                fb = await fb
                if fb is None: return None
                start = -shape[-1]
                fb = fb[:, start : start+bpc] # Returned names are ignored.
                fb = sn.Int.decode_bits(sn.backend, fb) % cpc
                R = sn.Int.repack(sn, fb, cpc, self.opts)[:self.sz].reshape(self.shape)
                return R if len(R.shape)>0 else R.item()
            return do_query(sn.query(None, names) if cells else None)
        @staticmethod
        def encode_ints(np, ints, bitcount):
            """`sn.Int.encode_ints(sn, ints, bitcount)???bits`: turns an `(N, 1)`-shaped int32 array into a `(N, bitcount)`-shaped float32 array of -1|1. Importantly, `choices_per_cell >= 2**bitcount`."""
            assert len(ints.shape) == 2 and ints.shape[-1] == 1
            powers2 = 2 ** np.arange(bitcount-1, -1, -1, dtype=np.int32)
            return np.where((ints & powers2) > 0, np.full_like(powers2, 1., dtype=np.float32), np.full_like(powers2, -1., dtype=np.float32))
        @staticmethod
        def decode_bits(np, bits):
            """`sn.Int.decode_bits(sn, bits)???ints`: from `(N, bitcount)`-shape float32 to `(N,)`-shape int32. Use `ints % options` or `sn.Int.repack(ints, choices_per_cell, options)` afterward."""
            assert len(bits.shape) == 2
            powers2 = 2 ** np.arange(bits.shape[1]-1, -1, -1, dtype=np.int32)
            return np.where(bits > 0, powers2, np.full_like(powers2, 0)).sum(-1)
        @staticmethod
        def repack(sn, ints, from_opts, to_opts):
            """`repack(sn, ints, from_opts, to_opts)???ints`: changes the base of `ints` digits, and thus its size. Importantly, `repack(sn, repack(sn, X, A, B), B, A) = X`."""
            assert isinstance(ints, int) or len(ints.shape) == 1
            np = sn.backend
            from math import log, floor
            if from_opts > to_opts: # More `ints`: unpack.
                mul = floor(log(from_opts, to_opts)) # Allow aliasing if source has too much capacity.
                if isinstance(ints, int): return ints * mul
                powers = to_opts ** np.arange(mul-1, -1, -1, dtype=np.int32)
                ints = np.expand_dims(ints, 1)
                return (np.floor_divide(ints, powers) % to_opts).reshape(ints.shape[0] * mul)
            elif from_opts < to_opts: # Less `ints`: pack.
                div = floor(log(to_opts, from_opts)) # Prefer wasting capacity to not having enough.
                if isinstance(ints, int): return -(-ints // div)
                sz = -(-ints.shape[0] // div) * div
                ints = np.concatenate((ints, np.zeros((sz - ints.shape[0],), dtype=np.int32)))
                ints = ints.reshape(ints.shape[0] // div, div)
                powers = from_opts ** np.arange(div-1, -1, -1, dtype=np.int32)
                return (ints * powers).sum(-1)
            else:
                return ints
    class Float:
        """
        `sn.Float(*shape, dims=1)`

        Datatype: an array of floating-point numbers. For example, `sn.Float(256)` is a simple array of up-to 256 numbers, while `sn.Float(32, 32, dims=2)` is a grayscale image.
        - `shape`: the maximum sizes of data, per-dimension. This allows handlers to learn the relative positions of all cells when `sn.set`ting, even when data is underspecified.
        - `dims`: how many dimensions to create per-cell patches over. In particular, using `dims=2` and image-shapes like `(3,256,256)` recreates the popular machine-learning technique of [splitting images ](https://arxiv.org/abs/2010.11929)[into patches.](https://arxiv.org/abs/2105.01601)
            - For efficiency, `sn.cell_shape[-1] ** (1/dims)` should be an integer. For 2d images, 64 or 256 work well.

        To use this, `Handler`s need `info={'analog':True}`.

        Compared to `sn.Int` or `sn.IntFloat`:
        - This is much more space-efficient, but may be much harder to generate precisely.
        - Models differ. Ints have to be sampled either autoregressively ([GPT](https://arxiv.org/abs/2005.14165)-like, materializing explicit probabilities cell-by-cell; very slow) or deterministically (sample the history-embedding, then simply unroll the model). Floats have to be sampled via i.e. GANs/DDPGs, VAEs, diffusion models.
        """
        __slots__ = ('shape', 'dims')
        def __init__(self, *shape, dims=1):
            assert isinstance(dims, int) and dims >= 1
            assert shape, "Shape cannot be empty"
            assert isinstance(shape, tuple) and all(isinstance(n, int) and n>0 for n in shape) and len(shape) >= dims
            self.shape, self.dims = shape, dims
        def __repr__(self):
            d = '' if self.dims == 1 else 'dims='+str(self.dims)
            d = d if not d or not len(self.shape) else ','+d
            return 'sn.Float(' + ','.join(repr(s) for s in self.shape) + d + ')'
        def set(self, sn, name, data, error):
            for fn in sn.modify_name: name = fn(name)
            assert sn.info is None or sn.info['analog'] is True
            np = sn.backend
            data = np.array(data, dtype=np.float32, copy=False)
            if isinstance(error, float):
                error = np.full(data.shape, error, dtype=np.float32)
            assert error is None or isinstance(error, np.ndarray) and error.shape == data.shape
            data = sn.Float.expand_shape(np, data, self.shape)
            cur_shape = data.shape

            shape = sn._s.cell_shape
            if not len(shape): return
            data = sn.Float.blocks(np, data, sn._s.cell_shape[-1], self.dims)
            error = sn.Float.blocks(np, error, sn._s.cell_shape[-1], self.dims) if error is not None else None

            prefix = (Handler.Goal.goal, True, *sn.List.extra_prefix)
            names = sn._s.shaped_names(data.shape[0], cur_shape, self.shape, prefix, name)
            sn.set(None, np.concatenate((names, data), -1), None, error)
        def query(self, sn, name):
            # Flatten feedback's cells and reshape it to our shape.
            #   The returned names are ignored.
            for fn in sn.modify_name: name = fn(name)
            assert sn.info is None or sn.info['analog'] is True
            np = sn.backend

            shape = sn._s.cell_shape
            fb_shape = sn.Float.blocks_shape(self.shape, shape[-1], self.dims) if len(shape) else (0,0)
            prefix = (Handler.Goal.goal, True, *sn.List.extra_prefix)
            names = sn._s.shaped_names(fb_shape[0], self.shape, self.shape, prefix, name) if fb_shape[0] else np.zeros(fb_shape, dtype=np.float32)
            async def do_query(fb):
                if not len(shape): return
                fb = await fb
                if fb is None: return None
                return sn.Float.unblocks(fb[:, -shape[-1]:], self.shape, self.dims)
            return do_query(sn.query(None, names))
        @staticmethod
        def expand_shape(np, x, shape):
            while len(x.shape) < len(shape): x = np.expand_dims(x, 0)
            assert len(x.shape) == len(shape), "Add max batch dimension sizes to the datatype too"
            assert all(x.shape[i] <= shape[i] for i in range(len(x.shape)))
            return x
        @staticmethod
        def blocks_shape(x, in_cell, dims=1):
            # Shape-only `blocks`. Exists so that `query` can know how much to query.
            assert len(x) >= dims
            d = int(in_cell ** (1/dims))
            cells = 1
            for i in range(len(x)-dims): cells *= x[i]
            for i in range(dims): cells *= -(-x[-i-1] // d)
            return (cells, in_cell)
        @staticmethod
        def blocks(np, x, in_cell, dims=1):
            assert len(x.shape) >= dims
            d = int(in_cell ** (1/dims)) # One square/hypercubic patch per cell.
            assert d**dims <= in_cell <= (d+1)**dims
            if dims == 1: # Special-case this for efficiency; `d == in_cell` here.
                y = np.concatenate((x, np.zeros([*x.shape[:-1], (-x.shape[-1]) % d], dtype=np.float32)), -1) if x.shape[-1] % d else x
                p = 1
                for i in y.shape: p *= i # Quite a lot faster than `np.prod(y.shape)`.
                return y.reshape(p // d, d)
            # Zero-pad to make all of `dims` divisible by `d`.
            y = np.pad(x, list(reversed([(0, (-x.shape[-i-1]) % d if i < dims else 0) for i in range(len(x.shape))])))
            # Reshape all of `dims` to be `N/d ?? d`, then transpose such that all `d`s are at the end.
            y = y.reshape(*x.shape[:-dims], *[d if i%2 else y.shape[-(i//2)-1] // d for i in range(2*dims)])
            tr = list(range(len(y.shape)))
            for i in range(dims): tr[-i-1], tr[-i-i-1] = tr[-i-i-1], tr[-i-1]
            y = y.transpose(*tr)
            # Flatten all patches, and zero-pad.
            p = 1
            for i in y.shape[:-dims]: p *= i
            y = y.reshape(p, d**dims)
            return y if d**dims == in_cell else np.concatenate((y, np.zeros((y.shape[0], in_cell - d**dims), dtype=np.float32)), -1)
        @staticmethod
        def unblocks(y, shape, dims=1):
            # Reverse `blocks`.
            assert len(y.shape) == 2
            in_cell = y.shape[-1]
            d = int(in_cell ** (1/dims))
            if dims == 1: # Special-case this for efficiency.
                return y.reshape(*shape[:-1], shape[-1] + (-shape[-1]) % d)[..., :shape[-1]]
            # Un-zero-pad and un-flatten patches.
            x = y[:, :d**dims].reshape(*shape[:-dims], *reversed([-(-shape[-i-1] // d) for i in range(dims)]), *([d] * dims))
            # Un-transpose patches and reshape them back, and un-zero-pad.
            almost_final_shape = [*shape[:-dims], *[x*d for x in x.shape[-dims-dims : -dims]]]
            tr = list(range(len(x.shape)))
            for i in range(dims): tr[-i-1], tr[-i-i-1] = tr[-i-i-1], tr[-i-1]
            x = x.transpose(*tr)
            return x.reshape(*almost_final_shape)[tuple(slice(0,s) for s in shape)]
    class IntFloat:
        """
        `sn.IntFloat(*shape, opts=..., mu=0, domain=(-1,1))`: datatype, representing discretized floating-point numbers; for low-precision high-throughput analog numbers, use `sn.Float` instead.

        Each number is [??-encoded](https://en.wikipedia.org/wiki/%CE%9C-law_algorithm) if requested, or linearly-encoded by default. By default, each number occupies 1 cell.
        """
        goal = False
        __slots__ = ('shape', 'opts', 'mu', 'domain')
        def __init__(self, *shape, opts=..., mu=0, domain=(-1, 1)):
            self.shape, self.opts, self.mu, self.domain = shape, opts, mu, domain
        def __repr__(self):
            return 'sn.IntFloat(' + ','.join(repr(s) for s in self.shape) + ',opts='+repr(self.opts) + ',mu='+repr(self.mu) + ',domain='+repr(self.domain) + ')'
        def set(self, sn, name, data, error):
            assert error is None, "Unimplemented"
            opts = sn.info['choices_per_cell'] if self.opts is ... else self.opts
            np = sn.backend
            data = np.array(data, dtype=np.float32, copy=False)
            min, max = self.domain
            data = (2*data - min - max) / (max - min)
            return sn.set(name, self.encode(np, data, opts, self.mu), sn.Int(*self.shape, opts))
        def query(self, sn, name):
            opts = sn.info['choices_per_cell'] if self.opts is ... else self.opts
            np = sn.backend
            async def mu_decode(fb):
                fb = np.array(await fb, dtype=np.int32, copy=False)
                fb = self.decode(np, fb, opts, self.mu)
                min, max = self.domain
                return ((fb if len(fb.shape) else fb.item()) * (max - min) + min + max) * .5
            return mu_decode(sn.query(name, sn.Int(*self.shape, opts)))
        @staticmethod
        def encode(np, floats, opts, mu):
            floats = np.clip(floats, -1, 1)
            if mu:
                floats = np.sign(floats) * np.log(1 + mu*np.absolute(floats)) / np.log(1 + mu)
            return ((floats+1)*.5 * (opts-1)).astype(np.int32)
        @staticmethod
        def decode(np, ints, opts, mu):
            floats = np.clip(ints.astype(np.float32) / (opts-1) * 2 - 1, -1, 1)
            if mu:
                floats = np.sign(floats) * ((1+mu) ** np.absolute(floats) - 1) / mu
            return floats
    class Goal:
        """
        `sn.Goal(type)`: datatype, where the wrapped `type` will mark all its cells as to-be-sought-out. See `sn.handle` for how to detect those.

        There are generally two ways to specify goals of optimizers: as one number (the "return" in reinforcement learning) or as a tensor (the "prediction target" in supervised learning, or the "goal" in goal-conditioned RL). `sn.Goal` uses the latter, though in general, each way can be implemented using the other so there's no difference.
        """
        goal = False
        __slots__ = ('type',)
        def __init__(self, type=None): self.type = _default_typing(type)
        def __repr__(self):
            return 'sn.Goal(' + repr(self.type) + ')'
        def set(self, sn, name, data, error):
            prev, Handler.Goal.goal = Handler.Goal.goal, True
            try: return sn.set(name, data, self.type, error)
            finally: Handler.Goal.goal = prev
        def query(self, sn, name):
            prev, Handler.Goal.goal = Handler.Goal.goal, True
            try: return sn.query(name, self.type)
            finally: Handler.Goal.goal = prev
        def get(self, sn, name):
            prev, Handler.Goal.goal = Handler.Goal.goal, True
            try: return sn.get(name, self.type)
            finally: Handler.Goal.goal = prev
    class List:
        """
        ```py
        [*types]
        sn.List(*types)
        ```

        Datatype: a simple list of other datatypes. Data and error (if specified) must be per-type tuples/lists.

        List items share the name, but differ in the index. Can `sn.set` subsets: `None`s in data are ignored.
        """
        extra_prefix = []
        __slots__ = ('types',)
        def __init__(self, *types): self.types = [_default_typing(t) for t in types]
        def __repr__(self):
            return 'sn.List(' + ','.join(repr(t) for t in self.types) + ')'
        def set(self, sn, name, data, error):
            T = self.types
            assert isinstance(data, list) and len(data) <= len(T)
            assert error is None or len(data) == len(error)
            with List._ExtraPrefix() as extra:
                for i in range(len(data)):
                    if data[i] is not None:
                        sn.set(extra(name, i, T), data[i], T[i], None if error is None else error[i])
        def query(self, sn, name):
            T = self.types
            with List._ExtraPrefix() as extra:
                return asyncio.gather(*[sn.query(extra(name, i, T), T[i]) for i in range(len(T))])
        def get(self, sn, name):
            T = self.types
            with List._ExtraPrefix() as extra:
                return asyncio.gather(*[sn.get(extra(name, i, T), T[i]) for i in range(len(T))])
        class _ExtraPrefix: # For adding one number to the extra name-part that `Int`|`Float` prepend.
            def __enter__(self):
                List.extra_prefix.append(0.)
                return self.extra_prefix
            def __exit__(self, type, value, traceback):
                List.extra_prefix.pop()
            @staticmethod
            def extra_prefix(name, i, T):
                List.extra_prefix[-1] = i / (len(T)-1) * 2 - 1 if len(T)>1 else -1.
                return name
    class Dict:
        """
        ```py
        {**types}
        sn.Dict(**types)
        ```

        Datatype: a simple dict of other datatypes.

        Each entry's key gets appended to the dict's name. Can `sn.set` subsets: `None`s and missing `types` are ignored.
        """
        extra_prefix = []
        __slots__ = ('types',)
        def __init__(self, **types):
            assert all(isinstance(s, str) for s in types.keys())
            self.types = {k: _default_typing(v) for k,v in types.items()}
        def __repr__(self):
            return 'sn.Dict(' + ','.join(k+'='+repr(v) for k,v in self.types.items()) + ')'
        def set(self, sn, name, data, error):
            assert isinstance(data, dict)
            assert error is None or isinstance(error, dict)
            for k, v in data.items():
                if v is not None:
                    sn.set(tuple([*name, k]), v, self.types[k], None if error is None else error[k])
        def query(self, sn, name):
            async def finish(fb):
                return {k:v for k,v in zip(self.types.keys(), await fb)}
            return finish(asyncio.gather(*[sn.query(tuple([*name, k]), v) for k, v in self.types.items()]))
        def get(self, sn, name):
            async def finish(fb):
                return {k:v for k,v in zip(self.types.keys(), await fb)}
            return finish(asyncio.gather(*[sn.get(tuple([*name, k]), v) for k, v in self.types.items()]))

    def func(self, fn):
        """
        ```py
        @sn.func
        async def async_fn(img: sn.Float(28, 28, dims=2)) -> sn.Int(10): return 4
        ```

        Modular, seamless [behavior ](https://arxiv.org/abs/1805.01954)[clon](https://stable-baselines.readthedocs.io/en/master/guide/pretrain.html)[ing.](https://en.wikipedia.org/wiki/Mind_uploading)

        Note that the `sn` in `sn.func` is another argument, and could be any `Handler`. `sn.func` doesn't support sync execution, because `sn` was built to support async execution.

        To stop calling `async_fn` and start requesting results from `sn`, do `result.query = True`. To disable `sn`, do `result.set = False`. To do either of those globally, do `sn.Handler.func.query = True` and/or `sn.Handler.func.set = False` at program-start.
        """
        from inspect import signature, iscoroutinefunction
        assert callable(fn) and iscoroutinefunction(fn)
        name, sgn = fn.__name__, signature(fn)
        in_type = self.Dict(**{k: p.annotation if p.annotation is not sgn.empty else None for k,p in sgn.parameters.items()})
        out_type = self.Dict(**{'return': sgn.return_annotation if sgn.return_annotation is not sgn.empty else None})

        def on_call(*args, **kwargs):
            a, s, q = sgn.bind(*args, **kwargs), on_call.set, on_call.query
            a.apply_defaults()
            if s: self.set(name, a.arguments, in_type)
            async def on_result(R):
                R = await R
                if not q:
                    if s: self.set(name, {'return': R}, out_type)
                else: # `R` is never `None`, because `sn.Dict` prefers to return `{'return': None}`.
                    R = R['return']
                return R
            return on_result(fn(*args, **kwargs) if not q else self.query(name, out_type))

        if not hasattr(self.func, 'set'): default.Handler.func.set = True
        if not hasattr(self.func, 'query'): default.Handler.func.query = False
        on_call.set = self.func.set
        on_call.query = self.func.query
        return on_call



    @staticmethod
    def run(fn, *args, **kwargs):
        """A convenient potentially-async function decorator, simply doing a call.
        
        If not async, simply calls the function, else lets sync code seamlessly call async code: either creates & runs an `asyncio` loop or schedules a new task in an existing one.

        Note: `Handler` queries, while returning `await`able objects, do not follow Python's convention of "wait before running any `async`-function code", and register the queries immediately. To follow this convention in your code, when calling `async`-functions from sync code (such as `.sensors` callbacks), pass `sn.query(???)` as an arg to `sn.run(???)` and `await` it inside, rather than doing `await sn.query(???)` inside."""
        R = fn(*args, **kwargs) if callable(fn) else fn
        if not asyncio.iscoroutinefunction(fn): return R
        if hasattr(asyncio, 'get_running_loop'): # pragma: no cover
            try: asyncio.get_running_loop();  return asyncio.create_task(R)
            except: return asyncio.run(R)
        else: # pragma: no cover
            try: b = asyncio.get_event_loop().is_running()
            except RuntimeError: b = False # Throws *after* the first global `sn.run`.
            return asyncio.ensure_future(R) if b else asyncio.run(R)



    @staticmethod
    def torch(torch, tensor, awaitable=False): # pragma: no cover
        """PyTorch integration, providing GPU???CPU async transfer, usable as `await sn.torch(torch, x, True)` or `sn.handle(sn.torch(torch, x))`. (Since PyTorch doesn't make this easy.)"""
        if not isinstance(tensor, torch.Tensor) or not tensor.is_cuda:
            tensor = tensor.detach().numpy() if isinstance(tensor, torch.Tensor) else tensor
            if not awaitable:
                return tensor
            else:
                f = asyncio.Future()
                f.set_result(tensor)
                return f
        with torch.no_grad():
            # https://discuss.pytorch.org/t/non-blocking-device-to-host-transfer/42353/2
            result = torch.zeros_like(tensor, dtype=torch.float32, layout=torch.strided, device='cpu', memory_format=torch.contiguous_format)
            result.copy_(tensor, non_blocking=True)
            event = torch.cuda.Event()
            event.record()
            if not awaitable:
                return lambda: event.query() and result.numpy()
            else:
                async def busyquery():
                    while True:
                        if event.query(): return result.numpy()
                        await asyncio.sleep(.003)
                return busyquery()



def _shape_ok(cell_shape: tuple):
    assert isinstance(cell_shape, tuple)
    assert all(isinstance(s, int) and s>=0 for s in cell_shape)
    assert not cell_shape or cell_shape[-1] > 0
def _str_to_floats(np, string: str):
    hash = hashlib.md5(string.encode('utf-8')).digest()
    return np.array(np.frombuffer(hash, dtype=np.uint8), np.float32)/255.*2. - 1.
def _fill(np, x, size): # ??? y
    """
    Ensures that the last axis of a NumPy array `x` has the appropriate `size`, returning `y`.

    If it's too small, fractally folds `x` via repeated `x ??? 1 - 2*abs(x)` to increase AI-model sensitivity where we can.

    >>> import numpy as np
    >>> _fill(np, np.zeros((2,), dtype=np.float32), 6)
    np.array([ 0.,  0.,  1.,  1., -1., -1.])
    """
    sz = x.shape[-1]
    if sz == size: return x
    if sz > size: return x[..., :size]
    folds = [x]
    for _ in range(1, -(-size // sz)):
        folds.append(1 - 2 * np.abs(folds[-1]))
    x = np.concatenate(folds, -1)
    if x.shape[-1] == size: return x
    return x[..., :size]
def _feedback(callbacks, feedback):
    fb = None
    got_err = None
    for callback, start_cell, end_cell in callbacks:
        if feedback is not None:
            assert feedback.shape[0] >= end_cell
            fb = feedback[start_cell:end_cell, :]
        try:
            callback(fb) if callable(callback) else callback.set_result(fb) if not callback.cancelled() else None
        except KeyboardInterrupt as err:
            got_err = err
        except Exception as err:
            got_err = err
    if got_err is not None: raise got_err
def _concat_error(main, error, length, np):
    if any(e is not None for e in error):
        assert len(main)
        return np.concatenate([e if e is not None else np.zeros((d.shape[0], length), dtype=np.float32) for d,e in zip(main, error)], 0)
    else:
        return None
def _name_template(np, str_to_floats, cell_shape, name):
    assert isinstance(name, tuple)
    if len(name) > len(cell_shape)-1:
        import warnings
        warnings.warn("The name is longer than the cell-shape allows for: " + str(name))
    template = np.full((sum(cell_shape[:-1]),), np.nan, dtype=np.float32)
    at = 0
    for i, sz in enumerate(cell_shape[:-1]):
        in_name = i < len(name)
        part = name[i] if in_name else None
        if isinstance(part, str):
            template[at : at+sz] = _fill(np, str_to_floats(part), sz)
        elif isinstance(part, tuple):
            if len(part) > sz:
                import warnings
                warnings.warn("A tuple is longer than fits in a name part: " + str(part) + "; simplify types or names")
            template[at : at + sz] = _fill(np, np.array(part, dtype=np.float32), sz)
        elif isinstance(part, float) or isinstance(part, int):
            raise TypeError("A part of the name is a number; wrap it in a tuple")
        elif part is not None:
            raise TypeError("Names must consist of strings, `None`, and tuples of either numbers or `None`s")
        at += sz
    return template
def _default_typing(type):
    """Implicit types, computed when `name is not None`. Convenience."""
    if type is None: return Handler.Event()
    if isinstance(type, int): return Handler.Int(type)
    if isinstance(type, dict): return Handler.Dict(**type)
    if isinstance(type, tuple) or isinstance(type, list):
        assert len(type), "Can't be non-empty"
        if isinstance(type[0], int) and all(t == type[0] for t in type):
            return Handler.Int(len(type), type[0])
        return Handler.List(*type)
    return type



class _S:
    """Holds dynamically-changing-pointer `Handler` state, so that these changes can be shared among `Handler` instances."""
    __slots__ = ('cell_shape', 'cell_size', 'n', 'wait_for_requests', 'name', 'shaped_names', 'query_cell')



# Make the module itself act exactly like an instance of `Handler`, and the other way around.
Handler.Handler = Handler
default = Handler(_default=True)
backend = default.backend
sensors = default.sensors
modify_name = default.modify_name
cell_shape, cell_size, info = default._s.cell_shape, default._s.cell_size, default.info
fork = default.fork
name = default.name
set = default.set
query = default.query
get = default.get
pipe = default.pipe
submit = default.submit
handle = default.handle
commit = default.commit
discard = default.discard
Event = Handler.Event
Int = Handler.Int
Float = Handler.Float
IntFloat = Handler.IntFloat
Goal = Handler.Goal
List = Handler.List
Dict = Handler.Dict
func = Handler.func
run = Handler.run
torch = Handler.torch