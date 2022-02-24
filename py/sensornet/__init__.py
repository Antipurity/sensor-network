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

h = sn.Handler((8, 24, 64), 8) # See `sn.Namer` for discussion of cell shapes.
# OR, simply use the global `sn` as if it's a handler:
sn.shape((8, 24, 64), 8)
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
async def main():
    fb = None
    while True:
        await h.wait() # TODO:
        data, query, data_error, query_error = h.handle(fb)
        fb = np.random.rand(query.shape[0], data.shape[1])*2-1)
```

This module implements this basic protocol, and does not include anything else by default, such as string/image handling or file storage or Internet communication.

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
    `Handler(cell_shape=None, part_size=None)`

    A differentiable sensor network: gathers numeric data from anywhere, and in a loop, handles it (and sends feedback back if requested).

    All data is split into fixed-size cells, each of which has a numeric name and a part of data. Handlers (AI models) should be position-invariant.

    Either pass in `cell_shape` (such as `(8, 24, 64)`: the last one is how many data-numbers there are per cell, the rest are the name's sizes) and `part_size` (such as `8`; sent name is split into parts; for example, each string in a name would take up 1 part), or call `.shape(cell_shape, part_size)`.

    To gather data automatically, do `handler.sensors.append(lambda handler: ...)`.

    If needed, read `.cell_shape` or `.part_size` or `.cell_size` wherever the object is available. These values might change between sending and receiving feedback.
    """
    def __init__(self, cell_shape=None, part_size=None):
        self._query_cell = 0
        self._data = []
        self._query = []
        self._data_error = []
        self._query_error = []
        self._prev_fb = [] # […, [prev_feedback, _next_fb, cell_shape, part_size, cell_count, cell_size], …]
        self._next_fb = [] # […, (on_feedback, shape, start_cell, end_cell, namer, length), …]
        self._wait_for_requests = None # asyncio.Future()
        self.sensors = [] # Called by `.handle(…)`.
        self.cell_shape = ()
        self.part_size = 0
        self.cell_size = 0
        if cell_shape is not None or part_size is not None:
            self.shape(cell_shape, part_size)
    def shape(self, cell_shape, part_size):
        """`sn.shape(cell_shape, part_size)`

        Changes the current shape. `cell_shape` is `(padding, name_size, data_size)`, where `padding % part_size == 0` and `name_size % part_size == 0`.

        Recommendation: leave space for about 3 name parts, meaning, `name_size = part_size*3`. Also leave some space in `padding`, in case some transforms want to leave their mark such as the source IP."""
        _shape_ok(cell_shape, part_size)
        if self.cell_shape == cell_shape and self.part_size == part_size:
            return
        self.discard()
        self.cell_shape = cell_shape
        self.part_size = part_size
        self.cell_size = sum(cell_shape)
        self.n = 0 # For `.wait(…)`, to ensure that the handling loop yields at least sometimes, without the overhead of doing it every time.
    def data(self, name=None, data=None, error=None, reward=0.):
        """
        `sn.data(name, data, error=None, reward=0.)`

        Sends named data to the handler. Receives nothing; see `.query`.

        Args:
        - `name`:
            - If a tuple/list of strings and -1…1 numbers and functions to -1…1 numbers from start-number & end-number & total-numbers NumPy arrays, converted to a `Namer`.
                - Recommendation: try to use at most about 3 parts, or however much the cell-shape allows for.
            - If a `Namer`, it is used.
            - If `None`, `data` & `error` must already incorporate the name and be sized `cells×cell_size`.
        - `data`: a NumPy array of numbers, preferably -1…1.
        - `error = None`: data transmission error: `None` or a `data`-sized float32 array of `abs(true_data - data) - 1`. Preferably -1…1.
        - `reward = 0.`: rates prior performance of these cells with -1…1, for reinforcement learning. In a handler, extract with `data[:, 0]`. Pass in `None` to not overwrite the first number.
        """
        if isinstance(name, tuple) or isinstance(name, list): name = Namer(*name)
        elif isinstance(name, str): name = Namer(name)
        if isinstance(error, float): error = np.full_like(data, error)

        assert name is None or isinstance(name, Namer)
        assert isinstance(data, np.ndarray)
        assert error is None or isinstance(error, np.ndarray) and data.shape == error.shape

        if not self.cell_size: return

        if name is not None:
            if len(data.shape) != 1: data = data.flatten()
            data = name.name(data, data.shape[0], self.cell_shape, self.part_size, None)
            if error is not None:
                if len(error.shape) != 1: error = error.flatten()
                error = name.name(error, error.shape[0], self.cell_shape, self.part_size, -1.)
        assert len(data.shape) == 2 and data.shape[-1] == self.cell_size
        if reward is not None: data[:, 0] = reward
        self._data.append(data)
        self._data_error.append(error)
        if self._wait_for_requests is not None:
            self._wait_for_requests.set_result(None)
            self._wait_for_requests = None
    def query(self, name=None, query=None, error=None, reward=0., callback=None):
        """
        ```python
        await sn.query(name, query: int|tuple, *, reward=0.)
        sn.query(name, query: int|tuple, *, reward=0., callback = lambda feedback, sn: ...)
        ```

        From the handler, asks for a NumPy array, or `None` (usually on transmission errors).

        Args:
        - `query`: the shape of the feedback that you want to receive.
            - Requesting a whole cell (`sn.cell_shape[-1]`) rather than `1` number may be more performant due to implementation details.
        - `callback = None`: if `await` has too much overhead, this could be a function that is given the feedback.
            - `.query` calls impose a global ordering, and feedback only arrives in that order, delayed. So to reduce memory allocations, could reuse the same function and use queues.
        - `name`, `reward`: see `.data`.
        """
        if isinstance(name, tuple) or isinstance(name, list): name = Namer(*name)
        elif isinstance(name, str): name = Namer(name)
        if callback is None: callback = asyncio.Future()

        assert name is None or isinstance(name, Namer)
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
            query = name.name(None, length, self.cell_shape, self.part_size, None)
        assert len(query.shape) == 2 and query.shape[-1] == self.cell_size-self.cell_shape[-1]
        assert error is None or isinstance(error, np.ndarray) and query.shape == error.shape
        if reward is not None: query[:, 0] = reward
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
        self.data(None, data, data_error, None)
        return self.query(None, query, query_error, None, callback)
    async def get(self, name, query, error=None, reward=0.):
        """
        `await sn.get(name, query, *, reward=0.)`

        Gets feedback, guaranteed.

        Never returns `None`, instead re-querying until a numeric result is available.
        """
        while True:
            fb = await self.query(name, query, error, reward)
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
        - `data_error`, `query_error`: data transmission error: `None` or a `data`-sized float32 array of `abs(true_data - data) - 1`.
            - A usage example: `if data_error is not None: data = numpy.clip(data + (data_error+1) * (numpy.random.rand(*data.shape)*2-1), -1, 1)`.
        - (To extract rewards: `data[:, 0]` and/or `query[:, 0]`.)
        """
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
                feedback, callbacks, cell_shape, part_size, cell_count, cell_size = self._prev_fb[0]
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
                _feedback(callbacks, feedback, cell_shape, part_size)
        except:
            self.discard()
            raise
        if max_simultaneous_steps is None:
            return self._take_data()
        else:
            return self._wait_then_take_data(max_simultaneous_steps)
    # TODO: Maybe take this opportunity to have `.commit()`?
    def discard(self):
        """Clears all scheduled-to-be-sent data."""
        try:
            _feedback(self._next_fb, None, self.cell_shape, self.part_size)
        finally:
            self._query_cell = 0
            self._data.clear()
            self._query.clear()
            self._data_error.clear()
            self._query_error.clear()
            self._next_fb.clear()
    def _take_data(self):
        """Gather data, queries, and their errors, and return them."""
        L1, L2 = self.cell_size, self.cell_size - (self.cell_shape[-1] if len(self.cell_shape) else 0)
        data = np.concatenate(self._data, 0) if len(self._data) else np.zeros((0, L1))
        query = np.concatenate(self._query, 0) if len(self._query) else np.zeros((0, L2))
        data_error = _concat_error(self._data, self._data_error, L1)
        query_error = _concat_error(self._query, self._query_error, L2)
        self._prev_fb.append([False, self._next_fb, self.cell_shape, self.part_size, query.shape[0], self.cell_size])
        self._next_fb = []
        self.discard()
        return (data, query, data_error, query_error)
    async def _wait_then_take_data(self, max_simultaneous_steps = 16):
        """
        Limits how many steps can be done at once, yielding if necessary (if no data and no queries, or if it's been too long since the last yield).

        Particularly important for async feedback: if the handling loop never yields to other tasks, then they cannot proceed, and a deadlock occurs (and memory eventually runs out).
        """
        # TODO: ...Re-run all tests while measuring coverage, *again*...
        assert isinstance(max_simultaneous_steps, int) and max_simultaneous_steps > 0
        if not len(self._data) and not len(self._query):
            self._wait_for_requests = asyncio.Future()
            await self._wait_for_requests
        self.n += 1
        if self.n >= max_simultaneous_steps: self.n = 0
        if self.n == 0:
            await asyncio.sleep(0)
        if len(self._prev_fb) <= max_simultaneous_steps: return self._take_data()
        fb = self._prev_fb[0][0] # The oldest feedback, must be done.
        if isinstance(fb, asyncio.Future) and not fb.done():
            await fb
            return self._take_data()
        elif callable(fb): # pragma: no cover
            while True:
                r = fb()
                if r is not False:
                    self._prev_fb[0][0] = r
                    return self._take_data()
                await asyncio.sleep(.003)



class Namer:
    """
    `Namer(*name)`

    An optimization opportunity: wrapping a name in this and reusing this object is faster than passing the name tuple directly to `handler.send(name=...)` (which re-constructs this object each time).

    ---

    To achieve position-invariance of `handler.send`, data cells need names.

    Naming data first flattens then transforms it into a 2D array, sized `cells×sum(cell_shape)`. All in `cell_shape[:-1]` must be divisible by `part_size`.

    Numeric names are split into fixed-size *parts*. `cell_shape[-2]` is where `Namer` puts its parts, so ensure that `cell_shape` has space for a few. Each part can be:
    - A string: MD5-hashed, and the resulting 16 bytes are put into one part, shifted and rescaled to -1…1.
    - A number, -1…1: put into a part directly, potentially shared with non-string pieces.
    - A function `f(start, end, total)` for dynamic cell naming. The arguments refer to indices in the data; the result is a number.
        - For example, `lambda start, end, total: start/total*2-1` puts a number, from -1 (inclusive) to 1 (exclusive).
        - For example, `lambda start, end, total: end / total*2-1` puts a number, from -1 (exclusive) to 1 (inclusive).
        - (Good idea to always include at least something dynamic, unless data only occupies one cell.)

    Neural networks are good at classification but are poor at regression. So numbers are repeatedly folded via `x → 1 - 2*abs(x)` until a part has no free space, to increase AI-model sensitivity. Handlers can also un/fold or apply I/FFT if needed.
    """
    def __init__(self, *name):
        self.named = name
        self.name_parts = None
        self.cell_shape = None
        self.part_size = None
        self.cell_size = None
        self.last_cells, self.last_name = None, None
    def name(self, data, length, cell_shape, part_size, fill=None):
        """
        1D to 2D.
        """
        assert data is None or len(data.shape) == 1
        self._name_parts(cell_shape, part_size)
        # Pad & reshape `data`.
        data_size = cell_shape[-1]
        name_size = self.cell_size - data_size
        cells = -(-length // data_size)
        total = cells * data_size
        if data is not None:
            data = _fill(data, total, 0) # Can't make it smaller, so `_unfill` will never have to make feedback larger.
            data = np.reshape(data, (cells, data_size))
        # Finalize the name, then concat it before `data`.
        if fill is not None:
            name = np.full((cells, name_size), fill)
            return np.concatenate((name, data), 1)
        start = np.expand_dims(np.arange(0, total, data_size), -1)
        end = np.minimum(start + data_size, length)
        if self.last_cells != cells:
            name = self.last_name = np.concatenate([_fill(np.concatenate([x(start, end, total) if callable(x) else np.repeat(x, cells, 0) for x in p], 1), part_size, 1) if isinstance(p, list) else np.repeat(p, cells, 0) for p in self.name_parts], 1)
            self.last_cells = cells
        else:
            name = self.last_name
        name = _fill(name, name_size, 1)
        return np.concatenate((name, data), 1) if data is not None else name
    def unname(self, feedback, length, cell_shape, _):
        """
        Reverses `.name`.
        """
        # Ignore the name, only heed data.
        assert len(feedback.shape) == 2 and feedback.shape[-1] == sum(cell_shape)
        feedback = feedback[:, -cell_shape[-1]:]
        return _unfill(feedback.flatten(), length, 0)
    def _name_parts(self, cell_shape, part_size):
        # Recomputes the name's parts for faster naming. MD5-hashes, and merges consecutive raw numbers.
        if cell_shape == self.cell_shape and part_size == self.part_size:
            return self.name_parts
        _shape_ok(cell_shape, part_size)
        extra_parts = sum(cell_shape[:-2]) // part_size
        name_parts = [np.zeros((1, part_size)) for _ in range(extra_parts)]
        nums = []
        for part in self.named:
            if isinstance(part, str):
                name_parts.append(_fill(np.expand_dims(_str_to_floats(part), 0), part_size, 1))
            elif callable(part) or not isinstance(part, bool) and (isinstance(part, float) or isinstance(part, int)):
                nums.append(np.atleast_2d(part) if not callable(part) else part)
                if len(nums) >= part_size:
                    name_parts.append(nums)
                    nums = []
            else:
                raise TypeError("Names must consist of strings, numbers, and number-returning functions")
        if len(nums): name_parts.append(nums)
        # Concat consecutive numbers in the name for a bit more performance.
        i = 0
        while i < len(name_parts):
            part = name_parts[i]
            if isinstance(part, list):
                start, end = 0, 0
                while end <= len(part):
                    if start<end and (end >= len(part) or not isinstance(part[end], np.ndarray)):
                        L = end-start
                        part[start:end] = [np.concatenate(part[start:end], 1)]
                        end -= L-1
                        start = end
                    if start < len(part) and not isinstance(part[start], np.ndarray): start = end+1
                    end += 1
                if len(part) == 1:
                    part = name_parts[i] = _fill(part[0], part_size, 1)
            if i > 0 and isinstance(name_parts[i-1], np.ndarray) and isinstance(name_parts[i], np.ndarray):
                name_parts[i-1:i+1] = [np.concatenate(name_parts[i-1:i+1], 1)]
            else:
                i += 1
        self.name_parts = name_parts
        self.cell_shape = cell_shape
        self.part_size = part_size
        self.cell_size = sum(cell_shape)
        self.last_cells, self.last_name = None, None
        return self.name_parts



def _shape_ok(cell_shape: tuple, part_size: int):
    assert isinstance(part_size, int) and part_size > 0
    assert isinstance(cell_shape, tuple)
    assert all(_inty(s) for s in cell_shape)
    assert cell_shape[-1] > 0
    assert all(s % part_size == 0 for s in cell_shape[:-1])
def _str_to_floats(string: str):
    hash = hashlib.md5(string.encode('utf-8')).digest()
    return np.frombuffer(hash, dtype=np.uint8).astype(np.float32)/255.*2. - 1.
def _fill(x, size, axis=0): # → y
    """
    Ensures that an `axis` of a NumPy array `x` has the appropriate `size`, returning `y`.

    If it's too small, fractally folds `x` via repeated `x → 1 - 2*abs(x)` to increase AI-model sensitivity where we can.

    >>> _fill(np.zeros((2,)), 6)
    np.array([ 0.,  0.,  1.,  1., -1., -1.])
    """
    if x.shape[axis] == size: return x
    if x.shape[axis] > size:
        return np.take(x, range(0,size), axis)
    folds = [x]
    for _ in range(1, -(-size // x.shape[axis])):
        folds.append(1 - 2 * np.abs(folds[-1]))
    x = np.concatenate(folds, axis)
    if x.shape[axis] == size: return x
    return np.take(x, range(0,size), axis)
def _unfill(y, size, axis=0): # → x
    """Undoes `_fill(x, y.shape[axis], axis)→y` via `_unfill(y, x.shape[axis], axis)→x`.

    `(x,y) → (copysign((1-y)/2, x), y)`"""
    if y.shape[axis] == size: return y
    assert y.shape[axis] > size # Users of `_unfill` never actually request the padding of data.
    folds = np.split(y, range(size, y.shape[axis], size), axis)
    if folds[-1].shape[0] < size:
        folds[-1] = np.concatenate((folds[-1], 1 - 2 * np.abs(np.take(folds[-2], range(folds[-1].shape[0], size), axis))), 0)
    for i in reversed(range(1, -(y.shape[axis] // -size))):
        x, y = folds[i-1], folds[i]
        folds[i-1] = np.copysign(.5 * (1-y), x)
    return folds[0]
def _feedback(callbacks, feedback, cell_shape, part_size):
    fb = None
    got_err = None
    assert feedback is None or feedback.shape[-1] == sum(cell_shape)
    for callback, shape, start_cell, end_cell, namer, length in callbacks:
        if feedback is not None:
            fb = feedback[start_cell:end_cell, :]
            assert fb.shape[0] == end_cell - start_cell
            if namer is not None:
                fb = namer.unname(fb, length, cell_shape, part_size)
                fb = fb.reshape(shape)
        try:
            callback(fb) if callable(callback) else callback.set_result(fb)
        except KeyboardInterrupt as err:
            got_err = err
        except Exception as err:
            got_err = err
    if got_err is not None: raise got_err
def _inty(n):
    return isinstance(n, int) and n>=0
def _concat_error(main, error, length):
    if any(e is not None for e in error):
        return np.concatenate([e if e is not None else -np.ones_like(d) for d,e in zip(main, error)], 0) if len(main) else np.zeros((0, length))
    else:
        return None



def torch(torch, tensor, awaitable=False): # pragma: no cover
    """PyTorch integration, providing GPU→CPU async transfer, usable as `await sn.torch(torch, x, True)` or `sn.handle(sn.torch(torch, x))`. (Since PyTorch doesn't make this easy.)"""
    if not tensor.is_cuda:
        tensor = tensor.detach().numpy()
        if not awaitable:
            return tensor
        else:
            f = asyncio.Future()
            f.set_result(tensor)
            return f
    with torch.no_grad():
        # https://discuss.pytorch.org/t/non-blocking-device-to-host-transfer/42353/2
        result = torch.zeros_like(tensor, layout=torch.strided, device='cpu', memory_format=torch.contiguous_format)
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



default = Handler()
sensors = default.sensors
cell_shape, part_size, cell_size = default.cell_shape, default.part_size, default.cell_size
def shape(*k, **kw):
    global cell_shape, part_size, cell_size
    r = default.shape(*k, **kw)
    cell_shape, part_size, cell_size = default.cell_shape, default.part_size, default.cell_size
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
def discard(*k, **kw):
    return default.discard(*k, **kw)
shape.__doc__ = Handler.shape.__doc__
data.__doc__ = Handler.data.__doc__
query.__doc__ = Handler.query.__doc__
pipe.__doc__ = Handler.pipe.__doc__
get.__doc__ = Handler.get.__doc__
handle.__doc__ = Handler.handle.__doc__
discard.__doc__ = Handler.discard.__doc__