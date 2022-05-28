"""
Bidirectional-sensor networks for ML: each gathers named numeric data and queries from anywhere, and in a loop, handles that.

This is the core protocol: as flexible as you can imagine, [as fast as you need](https://github.com/Antipurity/sensor-network/tree/master/py/sensornet/test.py), [as small as is reasonable](https://github.com/Antipurity/sensor-network/tree/master/py/sensornet/__init__.py), and [as well-tested as you can measure](https://github.com/Antipurity/sensor-network/tree/master/py/sensornet/test.py).

Position-invariant and numeric, nothing is fixed: these constraints free AI models from all concerns about data formats. With only a couple lines of code, you can: learn in completely arbitrary environments, combine tasks at runtime, and perform cross-dataset meta-learning and multimodal learning and multi-objective reinforcement learning and lifelong learning and model distillation.

Python 3.5 or newer (for `asyncio` with `await`).

---

Usage is simple.

First, initialize the handler:

```python
import sensornet as sn
import numpy as np

h = sn.Handler(8,8,8,8, 64) # First name-parts, then data, in a cell.
# OR, simply use the global `sn` as if it's a handler:
sn.shape(8,8,8,8, 64)
h = sn
```

Then send/receive data:

```python
def set():
    h.data('name', np.random.rand(32)*2-1)

async def get():
    nums = await h.get('name', 32)
    assert nums.shape == (32,)
```

And handle it:

```python
@sn.run
async def main():
    fb = None
    while True:
        data, query, data_error, query_error = await h.handle(fb)
        fb = np.random.rand(query.shape[0], data.shape[1])*2-1
```

This module implements this basic protocol, and does not include anything [else](https://github.com/Antipurity/sensor-network/tree/master/docs/ROADMAP.md) by default, such as string/image handling or file storage or multiprocessing or Internet communication or integration with ML-libraries.

(Implementing a controllable language with forking and/or listenable-to data, and training an AI model that does something useful there, is left as an exercise to the reader.)

---

## Integrations

- PyTorch: given `import torch`, `sn.torch(torch, tensor)` should be used as the argument to `h.handle(...)`.
"""



import numpy as np
import hashlib
import asyncio



class Handler:
    """
    ```python
    Handler().shape(8,8,8, 64)
    Handler(8,8,8, 64)
    Handler(*cell_shape, sensors=None, listeners=None, backend=numpy, namer_cache_size=1024, modify_name=None)
    ```

    A bidirectional sensor network: gathers numeric data from anywhere, and in a loop, handles it, responding to queries with feedback.

    All data is split into fixed-size cells, each of which has numeric name parts (`cell_shape[:-1]`) and data (`cell_shape[-1]`). Handlers (AI models) should be position-invariant.

    Inputs:
    - `cell_shape`: a tuple, where the last number is how many data-numbers there are per cell, and the rest splits the name into parts. In particular, each string in a name would take up 1 part.
    - `sensors`: function/s that take this handler, to prepare data to handle.
    - `listeners`: function/s that take data & error & cell-shape, when data is ready to handle. See `Filter`.
    - `backend`: the NumPy object.
    - `namer_cache_size`: info that is associated with names is cached for speed.
    - `modify_name`: a func from name to name, with singular strings already wrapped in a one-item tuple.

    If needed, read `.cell_shape` or `.cell_size` or `.backend`, or read/`.append(fn)` `.sensors` or `.listeners`, wherever the handler object is available. These values might change between sending and receiving feedback.
    """
    def __init__(self, *cell_shape, sensors=None, listeners=None, backend=np, namer_cache_size=1024, modify_name=None):
        assert modify_name is None or callable(modify_name)
        self._query_cell = 0
        self._data = []
        self._query = []
        self._data_error = []
        self._query_error = []
        self._prev_fb = [] # […, [prev_feedback, _next_fb, cell_shape, cell_count, cell_size], …]
        self._next_fb = [] # […, (on_feedback, shape, start_cell, end_cell, namer, length), …]
        self._wait_for_requests = None # asyncio.Future()
        self._namers_cache = {}
        x = sensors if sensors is not None else []
        self.sensors = [x] if callable(x) else list(x) if isinstance(x, tuple) else x
        x = listeners if listeners is not None else []
        self.listeners = [x] if callable(x) else list(x) if isinstance(x, tuple) else x
        self.cell_shape = ()
        self.cell_size = 0
        self.backend = backend
        self.namer_cache_size = namer_cache_size
        self.modify_name = modify_name
        if len(cell_shape):
            self.shape(*cell_shape)
    def shape(self, *cell_shape):
        """`sn.shape(*cell_shape)`

        Changes the current shape, where the last number is data, the rest split the name into parts. Safe to call whenever."""
        _shape_ok(cell_shape)
        if self.cell_shape == cell_shape:
            return self
        self.discard()
        self.cell_shape = cell_shape
        self.cell_size = sum(cell_shape)
        self.n = 0 # For `.wait(…)`, to ensure that the handling loop yields at least sometimes, without the overhead of doing it every time.
        return self
    def data(self, name=None, data=None, error=None):
        """
        `sn.data(name, data, error=None)`

        Sends named data to the handler. Receives nothing; see `.query`.

        Args:
        - `name`, such as `('image', .25, .5)`:
            - If a string, or a tuple/list of strings or per-part tuples, then it's used to turn the 1D data array into a 2D array.
            - If `None`, `data` & `error` must already incorporate the name and be sized `cells×sum(cell_shape)`. Either don't modify them in-place afterwards, or do `sn.commit()` right after this.
        - `data`: a list, or a NumPy 1D array of numbers, preferably -1…1.
        - `error = None`: data transmission error: `None` or a `data`-sized float32 array of `abs(true_data - data)`.

        More details on names:
        - A string: MD5-hashed, and the resulting 16 bytes are put into one part, sliced, and shifted and rescaled to -1…1.
        - `None` or `...`: anything goes here (zeros when naming).
        - A tuple:
            - A number, -1…1: put into a part directly.
            - A function `f(start, end, total)` for dynamic cell naming. The arguments refer to indices in the data; the result is a number.
                - For example, `lambda start, end, total: start/total*2-1` puts a number, from -1 (inclusive) to 1 (exclusive).
                - For example, `lambda start, end, total: end / total*2-1` puts a number, from -1 (exclusive) to 1 (inclusive).
                - (Good idea to always include at least something dynamic, unless data only occupies one cell.)
        """
        np = self.backend
        if isinstance(name, tuple) or isinstance(name, list): name = self._namer(name)
        elif isinstance(name, str): name = self._namer((name,))
        else: assert name is None
        if isinstance(error, float): error = np.full_like(data, error)
        if isinstance(data, list): data = np.array(data, dtype=np.float32)

        assert isinstance(data, np.ndarray)
        assert error is None or isinstance(error, np.ndarray) and data.shape == error.shape

        if not self.cell_size: return

        if name is not None:
            if len(data.shape) != 1: data = data.flatten()
            data = name.name(data, data.shape[0], self.cell_shape, None)
            if error is not None:
                if len(error.shape) != 1: error = error.flatten()
                error = name.name(error, error.shape[0], self.cell_shape, 0.)
        assert len(data.shape) == 2 and data.shape[-1] == self.cell_size
        self._data.append(data)
        self._data_error.append(error)
        if self._wait_for_requests is not None:
            self._wait_for_requests.set_result(None)
            self._wait_for_requests = None
    def query(self, name=None, query=None, error=None, callback=None):
        """
        ```python
        await sn.query(name, query: int|tuple)
        sn.query(name, query: int|tuple, *, callback = lambda feedback, sn: ...)
        ```

        From the handler, asks for a NumPy array, or `None` (usually on transmission errors).

        Args:
        - `name`: see `.data`.
        - `query`: the shape of the feedback that you want to receive.
        - `callback = None`: if `await` has too much overhead, this could be a function that is given the feedback.
            - `.query` calls impose a global ordering, and feedback only arrives in that order, delayed. So to reduce memory allocations, could reuse the same function and use queues.
        """
        np = self.backend
        if isinstance(name, tuple) or isinstance(name, list): name = self._namer(name)
        elif isinstance(name, str): name = self._namer((name,))
        else: assert name is None
        if callback is None: callback = asyncio.Future()

        assert isinstance(query, np.ndarray) if name is None else (isinstance(query, int) or isinstance(query, tuple))
        assert isinstance(callback, asyncio.Future) or callable(callback)

        if not self.cell_size:
            if callable(callback):
                callback(None)
                return
            else:
                f = asyncio.Future()
                f.set_result(None)
                return f

        length = None
        shape = query
        if name is not None:
            length = 1 if isinstance(query, tuple) else query
            if isinstance(query, tuple):
                for n in query: length *= n
            assert isinstance(length, int) and length > 0
            query = name.name(None, length, self.cell_shape, None)
        assert len(query.shape) == 2 and query.shape[-1] == self.cell_size-self.cell_shape[-1]
        assert error is None or isinstance(error, np.ndarray) and query.shape == error.shape
        self._query.append(query)
        self._query_error.append(error)
        cells = query.shape[0]
        self._next_fb.append((callback, shape, self._query_cell, self._query_cell + cells, name, length))
        self._query_cell += cells
        if self._wait_for_requests is not None:
            self._wait_for_requests.set_result(None)
            self._wait_for_requests = None
        if isinstance(callback, asyncio.Future):
            return callback
    def pipe(self, data, query, data_error, query_error, callback=None):
        """
        `prev_feedback = await other_handler.pipe(*sn.handle(prev_feedback))`

        Makes another handler handle this packet.

        Useful if NumPy arrays have to be transferred manually, such as over the Internet.
        """
        self.data(None, data, data_error)
        return self.query(None, query, query_error, callback)
    async def get(self, name, query, error=None):
        """
        `await sn.get(name, query)`

        Gets feedback, guaranteed.

        Never returns `None`, instead re-querying until a numeric result is available.
        """
        while True:
            fb = await self.query(name, query, error)
            if fb is not None: return fb
    def handle(self, prev_feedback=None, max_simultaneous_steps=16):
        """
        ```python
        await sn.handle(prev_feedback=None)
        await sn.handle(prev_feedback=None, max_simultaneous_steps=16)
        sn.handle(prev_feedback, None)
        ```

        Handles collected data.

        Pass it the previous handling's feedback: as a NumPy array sized `M×cell_size` or `None` or an `await`able future of that, or a low-level function (takes nothing, returns `False` to wait, `None` to drop, an array to respond).

        This returns `(data, query, data_error, query_error)`, or an `await`able promise of that.
        - `data`: float32 arrays of already-named cells of data, sized `N×cell_size`.
        - `query`: same, but sized `M×name_size` (only the name).
        - `data_error`, `query_error`: data transmission error: `None` or a `data`-sized float32 array of `abs(true_data - data)`.
            - A usage example: `if data_error is not None: data = numpy.clip(data + data_error * (numpy.random.rand(*data.shape)*2-1), -1, 1)`.
        """
        np = self.backend
        if asyncio.iscoroutine(prev_feedback) and not isinstance(prev_feedback, asyncio.Future):
            prev_feedback = asyncio.ensure_future(prev_feedback)
        assert prev_feedback is None or isinstance(prev_feedback, np.ndarray) or isinstance(prev_feedback, asyncio.Future) or callable(prev_feedback)
        assert max_simultaneous_steps is None or isinstance(max_simultaneous_steps, int) and max_simultaneous_steps > 0
        # Collect sensor data.
        for s in self.sensors: s(self)
        # Remember to respond to the previous step with prev_feedback.
        if len(self._prev_fb):
            self._prev_fb[-1][0] = prev_feedback
        else:
            assert prev_feedback is None, 'The first step cannot give feedback to its previous step'
        # Respond to what we can.
        try:
            while len(self._prev_fb):
                feedback, callbacks, cell_shape, cell_count, cell_size = self._prev_fb[0]
                if isinstance(feedback, asyncio.Future):
                    if not feedback.done(): break
                    feedback = feedback.result()
                else:
                    if callable(feedback):
                        feedback = feedback()
                    if feedback is False: break # Respond in-order, waiting if `False`.
                assert feedback is None or isinstance(feedback, np.ndarray)
                if feedback is not None:
                    assert len(feedback.shape) == 2 and feedback.shape[0] == cell_count and feedback.shape[1] == cell_size
                self._prev_fb.pop(0)
                _feedback(callbacks, feedback, cell_shape)
        except:
            self.discard()
            raise
        if max_simultaneous_steps is None:
            return self._take_data()
        else:
            return self._wait_then_take_data(max_simultaneous_steps)
    def commit(self):
        """`sn.commit()`: actually copies the provided data/queries, allowing their NumPy arrays to be written-to elsewhere."""
        np = self.backend
        if len(self._data) == 1 and len(self._query) == 1: return
        L1, L2 = self.cell_size, self.cell_size - (self.cell_shape[-1] if len(self.cell_shape) else 0)
        data = np.concatenate(self._data, 0) if len(self._data) else np.zeros((0, L1), dtype=np.float32)
        query = np.concatenate(self._query, 0) if len(self._query) else np.zeros((0, L2), dtype=np.float32)
        data_error = _concat_error(self._data, self._data_error, L1, np)
        query_error = _concat_error(self._query, self._query_error, L2, np)
        self._data.clear();  self._data.append(data)
        self._query.clear();  self._query.append(query)
        self._data_error.clear();  self._data_error.append(data_error)
        self._query_error.clear();  self._query_error.append(query_error)
    def discard(self):
        """Clears all scheduled-to-be-sent data."""
        try:
            _feedback(self._next_fb, None, self.cell_shape)
        finally:
            self._query_cell = 0
            self._data.clear()
            self._query.clear()
            self._data_error.clear()
            self._query_error.clear()
            self._next_fb.clear()
    def _take_data(self):
        """Gather data, queries, and their errors, and return them."""
        self.commit()
        data = self._data[0];  self._data.clear()
        query = self._query[0];  self._query.clear()
        data_error = self._data_error[0];  self._data_error.clear()
        query_error = self._query_error[0];  self._query_error.clear()
        self._prev_fb.append([False, self._next_fb, self.cell_shape, query.shape[0], self.cell_size])
        self._next_fb = []
        self.discard()
        for l in self.listeners: l(data, data_error, self.cell_shape)
        return data, query, data_error, query_error
    async def _wait_then_take_data(self, max_simultaneous_steps = 16):
        """
        Limits how many steps can be done at once, yielding if necessary (if no data and no queries, or if it's been too long since the last yield).

        Particularly important for async feedback: if the handling loop never yields to other tasks, then they cannot proceed, and a deadlock occurs (and memory eventually runs out).
        """
        assert isinstance(max_simultaneous_steps, int) and max_simultaneous_steps > 0
        if not len(self._data) and not len(self._query):
            self._wait_for_requests = asyncio.Future()
            await self._wait_for_requests
        self.n += 1
        if self.n >= max_simultaneous_steps: self.n = 0
        if self.n == 0:
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
    def _namer(self, name):
        if self.modify_name is not None:
            name = self.modify_name(name)
        hash, cache = _name_hash(name), self._namers_cache
        if hash not in cache:
            cache[hash] = _Namer(*name, backend=self.backend)
        if len(cache) > self.namer_cache_size: # pragma: no cover
            cache.popitem()
        return cache[hash]



class _Namer:
    """
    `_Namer(*name, backend=numpy)`

    The allocator of 1D arrays onto 2D named cells.

    ---

    Neural networks are good at classification but are poor at regression. So numbers are repeatedly folded via `x → 1 - 2*abs(x)` until a part has no free space, to increase AI-model sensitivity. Handlers can also un/fold or apply I/FFT if needed.
    """
    def __init__(self, *name, backend=np):
        self.named = name
        self.templ = None
        self.cell_shape = None
        self.cell_size = None
        self.last_cells, self.last_name = None, None
        self.backend = backend
    def name(self, data, length, cell_shape, fill=None):
        """
        1D to 2D.
        """
        np = self.backend
        assert data is None or len(data.shape) == 1
        if cell_shape != self.cell_shape: # Update what needs to be done.
            self.templ = _name_template(self.named, cell_shape, np)
            self.templ = np.nan_to_num(self.templ[0]), self.templ[1], self.templ[2]
            self.cell_shape = cell_shape
            self.cell_size = sum(cell_shape)
            self.last_cells, self.last_name = None, None
        # Pad & reshape `data`.
        data_size = cell_shape[-1]
        name_size = self.cell_size - data_size
        cells = -(-length // data_size)
        total = cells * data_size
        if data is not None:
            data = _pad(data, total, np=np)
            data = np.reshape(data, (cells, data_size))
        # Finalize the name, then concat it before `data`.
        if fill is not None:
            name = np.full((cells, name_size), fill)
            return np.concatenate((name, data), 1)
        if self.last_cells != cells: # Cache.
            # Do what `_name_template` implies: use template, fill func-indices, and `_fill` parts.
            template, func_indices, part_sizes = self.templ
            name = np.repeat(np.expand_dims(template, 0), cells, 0)
            #   (Not using NumPy's x.repeat(…) is a mild slowdown, but we care about compatibility with PyTorch more.)
            start = np.expand_dims(np.arange(0, total, data_size, dtype=np.float32), -1)
            end = np.minimum(start + data_size, length)
            #   (`np.clip(↑, length, None)` is much slower.)
            if len(func_indices):
                for at, fn in func_indices:
                    name[:, at:at+1] = fn(start, end, total)
                at = 0
                for i, part in enumerate(part_sizes):
                    sz = cell_shape[i]
                    if part != sz:
                        name[:, at : at+sz] = _fill(name[:, at : at+part], sz, np=np)
                    at += sz
            self.last_name, self.last_cells = name, cells
        else:
            name = self.last_name
        name = _fill(name, name_size, np=np)
        return np.concatenate((name, data), 1) if data is not None else name
    def unname(self, feedback, length, cell_shape):
        """
        Reverses `.name`.
        """
        # Ignore the name, only heed data.
        np = self.backend
        assert len(feedback.shape) == 2 and feedback.shape[-1] == sum(cell_shape)
        feedback = feedback[:, -cell_shape[-1]:]
        return _pad(feedback.flatten(), length, np=np)



class Filter:
    """`Filter(name, func, backend=numpy)`

    Wraps a `func(data, error=None, cell_shape)` such that it only sees the cells with numeric-names matching the `name`. The recommended way to specify `Handler().listeners`.

    Example uses: getting a global reward from the env; getting [CLIP](https://cliport.github.io/)-embedding goals from the env; debugging/reversing sensors with known code (i.e. showing the env's images).

    `func`:
    - `None`: a call will simply return a per-cell bit-mask of whether the name fits.
    - A function: not called if there are no matches, but otherwise, defers to `func` with `data` and `error` 2D arrays already lexicographically-sorted. But, they must be split/flattened/batched/gathered manually, for example via `data[:, -cell_shape[-1]:].flatten()[:your_max_size]`.

    If needed for manually naming cells, `fltr.template(cell_shape)` is a 1D NumPy array with `nan`s for `None`s and numbers for name-parts."""
    def __init__(self, name, func = None, backend=np):
        assert func is None or callable(func)
        self.name = name
        self.func = func
        self.cell_shape = None
        self.templ = None
        self.backend = backend
    def __call__(self, data, error=None, cell_shape=()):
        assert len(cell_shape), 'Specify the cell-shape too'
        np = self.backend
        # Match.
        template = self.template(cell_shape)
        name_sz = sum(cell_shape) - cell_shape[-1]
        matches = (template != template) | (np.abs(data[:, :name_sz] - template) <= (error if error is not None else 0.) + 1e-5)
        matches = matches.all(-1)
        if self.func is None:
            return matches
        data = data[matches]
        inds = np.lexsort(data.T[::-1])
        data = data[inds]
        if error is not None: error = error[matches][inds]
        # Call.
        if data.size > 0:
            return self.func(data, error, cell_shape)
    def template(self, cell_shape):
        """Returns a 1D template that's responsible for matching the name."""
        # Reconstruct the template if needed.
        if cell_shape != self.cell_shape:
            self.templ = _name_template(self.name, cell_shape, np)
            self.cell_shape = cell_shape
        template, func_indices, part_sizes = self.templ
        return template



class Torch: # pragma: no cover
    """A [PyTorch](https://pytorch.org/) backend. Often [slower](https://github.com/pytorch/pytorch/issues/9873) than NumPy, even on GPU, so, not recommended."""
    __slots__ = ('torch', 'nan', 'uint8', 'ndarray', 'float32')
    def __init__(self, autogpu=False):
        import torch
        if autogpu and torch.cuda.is_available():
            # (This is much easier than specifying `device=…`-but-only-if-not-NumPy.)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        self.torch = torch
        self.nan = float('nan')
        self.uint8 = torch.uint8
        self.ndarray = torch.Tensor
        self.float32 = torch.float32
    def repeat(self, x, repeats, axis):
        # Even though `torch.repeat_interleave`'s interface exactly matches, this is more efficient.
        assert axis == 0
        return x.expand(repeats, *x.shape[1:])
    def array(self, x, dtype):
        return self.torch.tensor(x, dtype=dtype) if not isinstance(x, self.torch.Tensor) else x.to(dtype)
    def expand_dims(self, x, axis):
        return self.torch.unsqueeze(x, axis)
    def concatenate(self, tensors, axis):
        return self.torch.cat(tensors, axis)
    def frombuffer(self, x, dtype):
        if hasattr(self.torch, 'frombuffer'): return self.torch.frombuffer(x, dtype)
        assert dtype is self.torch.uint8
        return self.torch.tensor(np.frombuffer(x, np.uint8), dtype=dtype)
    def lexsort(self, x):
        # https://discuss.pytorch.org/t/numpy-lexsort-equivalent-in-pytorch/47850/3
        assert len(x.shape) == 2
        inv = self.torch.unique(x.flip(0), dim=-1, sorted=True, return_inverse=True)[1]
        return self.torch.argsort(inv)
    def minimum(self, x1, x2):
        # `x2` is always a number here, and `torch.minimum` doesn't support that.
        return self.torch.clamp(x1, x2)
    def nan_to_num(self, x):
        return self.torch.where(self.torch.isnan(x), self.torch.zeros_like(x), x)
    def __getattr__(self, name):
        return object.__getattribute__(self.torch, name)



def _shape_ok(cell_shape: tuple):
    assert isinstance(cell_shape, tuple)
    assert all(isinstance(s, int) and s>=0 for s in cell_shape)
    assert cell_shape[-1] > 0
def _str_to_floats(string: str, np):
    hash = hashlib.md5(string.encode('utf-8')).digest()
    return np.array(np.frombuffer(hash, dtype=np.uint8), np.float32)/255.*2. - 1.
def _pad(x, size, np=...): # → y
    """Ensures that the 1D NumPy array `x` has the appropriate `size` by slicing or zero-padding, returning `y`. Can undo itself."""
    assert len(x.shape) == 1
    if x.shape[0] == size: return x
    if x.shape[0] > size: return x[:size]
    z = np.zeros((size - x.shape[0],), dtype=np.float32)
    return np.concatenate((x, z), 0)
def _fill(x, size, np=...): # → y
    """
    Ensures that the last axis of a NumPy array `x` has the appropriate `size`, returning `y`.

    If it's too small, fractally folds `x` via repeated `x → 1 - 2*abs(x)` to increase AI-model sensitivity where we can.

    >>> _fill(np.zeros((2,), dtype=np.float32), 6)
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
def _feedback(callbacks, feedback, cell_shape):
    fb = None
    got_err = None
    assert feedback is None or feedback.shape[-1] == sum(cell_shape)
    for callback, shape, start_cell, end_cell, namer, length in callbacks:
        if feedback is not None:
            fb = feedback[start_cell:end_cell, :]
            assert fb.shape[0] == end_cell - start_cell
            if namer is not None:
                fb = namer.unname(fb, length, cell_shape)
                fb = fb.reshape(shape)
        try:
            callback(fb) if callable(callback) else callback.set_result(fb) if not callback.cancelled() else None
        except KeyboardInterrupt as err:
            got_err = err
        except Exception as err:
            got_err = err
    if got_err is not None: raise got_err
def _concat_error(main, error, length, np):
    if any(e is not None for e in error):
        return np.concatenate([e if e is not None else np.zeros_like(d) for d,e in zip(main, error)], 0) if len(main) else np.zeros((0, length), dtype=np.float32)
    else:
        return None
def _name_template(name, cell_shape, np):
    """Converts a name into `(template, func_indices, part_sizes)`.

    - `template`: a NumPy array of shape `(sum(cell_shape[:-1]),)`, with `nan`s wherever the value doesn't matter.
    - `func_indices`: a list of `(index_to_write_at, func_to_write_the_result_of)`.
    - `part_sizes`: a list of name-part sizes, always equal to `cell_shape[p]` unless the part has a func, for `_fill`ing."""
    _shape_ok(cell_shape)
    template = np.full((sum(cell_shape[:-1]),), np.nan, dtype=np.float32)
    func_indices = []
    part_sizes = list(cell_shape[:-1])
    at, nums = 0, []
    for i, sz in enumerate(cell_shape[:-1]):
        in_name = i < len(name)
        part = name[i] if in_name else None
        if isinstance(part, str):
            template[at : at+sz] = _fill(_str_to_floats(part, np), sz, np=np)
        elif isinstance(part, tuple):
            for j, num in enumerate(part):
                if callable(num):
                    func_indices.append((at + j, num))
                else:
                    assert isinstance(num, float) or isinstance(num, int)
            nums = [None if callable(num) else num for num in part]
            template[at : at + sz] = _fill(np.array(nums, dtype=np.float32), sz, np=np)
            if any(callable(num) for num in part):
                part_sizes[i] = len(part)
        elif callable(part):
            raise TypeError("A part of the name is a func; wrap it in a tuple")
        elif isinstance(part, float) or isinstance(part, int):
            raise TypeError("A part of the name is a number; wrap it in a tuple")
        elif part is not None and part is not ...:
            raise TypeError("Names must consist of strings, `None`/`...`, and tuples of either numbers or number-returning functions")
        at += sz
    return template, func_indices, tuple(part_sizes)
def _name_hash(name):
    if callable(name): return id(name)
    if not isinstance(name, tuple) and not isinstance(name, list): return name
    result = name
    for i in range(len(name)):
        v1, v2 = name[i], _name_hash(name[i])
        if result is name and v1 is not v2: result = list(name)
        if v1 is not v2: result[i] = v2
    return tuple(result)



def torch(torch, tensor, awaitable=False): # pragma: no cover
    """PyTorch integration, providing GPU→CPU async transfer, usable as `await sn.torch(torch, x, True)` or `sn.handle(sn.torch(torch, x))`. (Since PyTorch doesn't make this easy.)"""
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



def run(fn, *args, **kwargs):
    """A convenience async-function decorator: equivalent to `import asyncio;  asyncio.run(fn())`."""
    return asyncio.run(fn(*args, **kwargs)) if asyncio.iscoroutinefunction(fn) else fn(*args, **kwargs)



default = Handler()
sensors = default.sensors
cell_shape, cell_size = default.cell_shape, default.cell_size
def shape(*k, **kw):
    global cell_shape, cell_size
    r = default.shape(*k, **kw)
    cell_shape, cell_size = default.cell_shape, default.cell_size
    return r
def data(*k, **kw):
    return default.data(*k, **kw)
def query(*k, **kw):
    return default.query(*k, **kw)
def pipe(*k, **kw):
    return default.pipe(*k, **kw)
def get(*k, **kw):
    return default.get(*k, **kw)
def handle(*k, **kw):
    return default.handle(*k, **kw)
def commit():
    return default.commit()
def discard():
    return default.discard()
shape.__doc__ = Handler.shape.__doc__
data.__doc__ = Handler.data.__doc__
query.__doc__ = Handler.query.__doc__
pipe.__doc__ = Handler.pipe.__doc__
get.__doc__ = Handler.get.__doc__
handle.__doc__ = Handler.handle.__doc__
commit.__doc__ = Handler.commit.__doc__
discard.__doc__ = Handler.discard.__doc__