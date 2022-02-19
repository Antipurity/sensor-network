"""
Module for differentiable sensor networks: each gathers named numeric data from anywhere, and in a loop, handles it (and sends feedback back if requested).

Position-invariant and numeric, nothing is fixed: these constraints free AI models from all concerns about data formats. With only a couple lines of code, you can: learn in completely arbitrary environments, combine tasks at runtime, and perform cross-dataset meta-learning and multimodal learning and multi-objective reinforcement learning and lifelong learning and model distillation.

Python 3.4 or newer (for `asyncio`).

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
    h.send(name = ('name',), data = np.random.rand(32)*2-1)

async def get():
    nums = await h.get(('name',), 32)
    assert nums.shape == (32,)
```

And handle it:

```python
async def main():
    fb = None
    while True:
        await h.wait()
        data, error, no_data, no_feedback = h.handle(fb)
        fb = process(...) # See `sn.Handler.handle` for what these mean.
```

This module implements this basic protocol, and does not include anything else by default, such as string/image handling or file storage or Internet communication.

(Implementing a controllable language with forking and/or listenable-to data, and training an AI model that does something useful there, is left as an exercise to the reader.)
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
        self._cell = 0
        self._data = []
        self._error = []
        self._no_data = []
        self._no_feedback = []
        self._prev_fb = [] # [prev_feedback, …, _next_fb, …]
        self._next_fb = [] # […, (on_feedback, shape, start_cell, end_cell, namer, length, original_shape), …]
        self.sensors = [] # Called by `.handle(…)`.
        self.cell_shape = ()
        self.part_size = 0
        self.cell_size = 0
        if cell_shape is not None or part_size is not None:
            self.shape(cell_shape, part_size)
    def shape(self, cell_shape, part_size):
        """`sn.shape(cell_shape, part_size)`

        Changes the current shape. `cell_shape` is `(padding, name_size, data_size)`, where `padding % part_size == 0` and `name_size % part_size == 0`."""
        _shape_ok(cell_shape, part_size)
        if self.cell_shape == cell_shape and self.part_size == part_size:
            return
        self.discard()
        self.cell_shape = cell_shape
        self.part_size = part_size
        self.cell_size = sum(cell_shape)
        self.n = 0 # For `.wait(…)`, to ensure that the handling loop yields at least sometimes, without too much overhead.
    def send(self, name=None, data=None, error=None, no_data=None, no_feedback=None, reward=0., on_feedback=None):
        """
        `sn.send(name=..., data=numpy.random.rand(16)*2-1)`
        `await sn.send(name=..., data=16, on_feedback=True)`
        `sn.send(name=None, data=None, error=None, no_data=None, no_feedback=None, reward=0., on_feedback=None)`

        Sends named data, possibly getting same-size feedback, possibly only getting feedback.

        Returns `None` or `on_feedback` if that is an `asyncio.Future`. If async, do `sn.wait()` before `sn.handle(...)`.

        Use `.get` to retry until feedback is non-`None`.

        Arguments:
        - `name = None`:
            - If a tuple/list of strings and -1…1 numbers and functions to -1…1 numbers from start-number & end-number & total-numbers NumPy arrays, converted to a `Namer`.
            - If a `Namer`, it is used.
            - If `None`, `data` & `error` must already incorporate the name and be sized `cells×cell_size`.
        - `data = None`: `None`, or how many numbers of no-data feedback to return, or a NumPy array of -1…1 numbers.
        - `error = None`: data transmission error: `None` or a `data`-sized float32 array of `abs(true_data - data) - 1`. -1…1.
        - `reward = 0.`: rates prior performance of these cells with -1…1, for reinforcement learning. Replaces the first number of every cell. Pass in `None` to disable this.
        - `on_feedback = None`: could be `True` or `asyncio.Future()` to return and fill, or for efficiency, a function from `feedback` (could be `None`, otherwise a NumPy array shaped the same as `data`), `cell_shape`, `part_size`, `handler`, to nothing.
            - `.send` calls impose a global ordering, and feedback only arrives in that order, delayed. So to reduce memory allocations, could reuse the same function and use queues.
        - `no_data = None` and `no_feedback = None`: for passing through `sn.handle(…)`'s result to another handler, like `h.send(None, *sn.handle(…), reward=None, on_feedback=...)`.
        """
        if data is None and on_feedback is None: return
        if on_feedback is True: on_feedback = asyncio.Future()
        assert name is None or isinstance(name, tuple) or isinstance(name, Namer)
        assert data is None or isinstance(data, np.ndarray) or _inty(data) or (isinstance(data, tuple) or isinstance(data, list)) and all(_inty(n) for n in data)
        assert error is None or isinstance(error, np.ndarray)
        assert on_feedback is None or callable(on_feedback) or isinstance(on_feedback, asyncio.Future)
        assert no_data is None and no_feedback is None or isinstance(no_data, np.ndarray) and isinstance(no_feedback, np.ndarray) and no_data.dtype == no_feedback.dtype == np.dtype('bool')
        if not self.cell_size:
            if on_feedback is not None:
                on_feedback(None, self.cell_shape, self.part_size, self) if callable(on_feedback) else on_feedback.set_result(None)
            return
        if no_feedback is None:
            no_feedback = not on_feedback
        if no_data is None:
            no_data = False if isinstance(data, np.ndarray) else True
            if no_data: data = np.zeros((data,) if _inty(data) else data if data is not None else (0, self.cell_size), dtype=np.float32)
        length = None
        original_shape = None
        if name is not None:
            if len(data.shape) != 1: original_shape, data = data.shape, data.flatten()
            if error is not None and len(error.shape) != 1: error = error.flatten()
            length = data.shape[0]
        # Name.
        if isinstance(name, tuple) or isinstance(name, list):
            name = Namer(*name)
        if isinstance(name, Namer):
            data = name.name(data, self.cell_shape, self.part_size, None)
            error = name.name(error, self.cell_shape, self.part_size, -1.)
        else:
            assert name is None
        assert len(data.shape) == 2
        assert data.shape[-1] == self.cell_size
        assert error is None or data.shape == error.shape
        # Reward.
        if reward is not None:
            data[:, 0] = reward
        # Send.
        cells = data.shape[0]
        shape = (cells,)
        if not isinstance(no_data, np.ndarray):
            no_data = np.full(shape, no_data)
        if not isinstance(no_feedback, np.ndarray):
            no_feedback = np.full(shape, no_feedback)
        assert no_data.shape == shape
        assert no_feedback.shape == shape
        self._data.append(data)
        self._error.append(error)
        self._no_data.append(no_data)
        self._no_feedback.append(no_feedback)
        if on_feedback is not None:
            self._next_fb.append((on_feedback, data.shape, self._cell, self._cell + cells, name, length, original_shape))
        self._cell += cells
        if isinstance(on_feedback, asyncio.Future):
            return on_feedback
    async def get(self, name, len, reward=0.):
        """
        `await sn.get(name, len, reward=0.)`

        Gets feedback, guaranteed. Never returns `None`, instead re-requesting until a numeric result is available.
        """
        while True:
            fb = await self.send(name, len, reward=reward, on_feedback=True)
            if fb is not None: return fb
    def handle(self, prev_feedback=None):
        """
        `sn.handle(prev_feedback=None)`

        Handles collected data.

        Pass it the previous handling's feedback: as a NumPy array or `None` or an `await`able future of that (see `sn.wait`).

        This returns `(data, error, no_data, no_feedback)`.
        - `data`: `None` or a float32 array of already-named cells of data, sized `cells×cell_size`. -1…1.
        - `error`: data transmission error: `None` or a `data`-sized float32 array of `abs(true_data - data) - 1`. -1…1.
            - A usage example: `if error is not None: data = numpy.clip(data + (error+1) * (numpy.random.rand(*data.shape)*2-1), -1, 1)`.
        - `no_data`: bools, sized `cells`.
        - `no_feedback`: bools, sized `cells`.
        - Usage:
            - `data[:, 0]` would return per-cell rewards.
            - `data[:, :-sn.cell_shape[-1]]` would select only names.
            - `data[:, -sn.cell_shape[-1]:]` would select only data.
            - `numpy.compress(~no_data, data)` would select only inputs.
            - `numpy.compress(~no_feedback, data)` would select only queries.
            - `numpy.put(numpy.zeros_like(data), numpy.where(~no_feedback)[0], feedback)` would put back the selected queries in-place, making `data` suitable for `prev_feedback` here.
        """
        if asyncio.iscoroutine(prev_feedback) and not isinstance(prev_feedback, asyncio.Future):
            prev_feedback = asyncio.ensure_future(prev_feedback)
        assert prev_feedback is None or isinstance(prev_feedback, np.ndarray) or isinstance(prev_feedback, asyncio.Future)
        # Collect sensor data.
        for s in self.sensors: s(self)
        # Gather data.
        if len(self._data):
            data = np.concatenate(self._data, 0)
            error = np.concatenate([e if e is not None else -np.ones((0, self.cell_size)) for e in self._error], 0) if any(e is not None for e in self._error) else None
            no_data = np.concatenate(self._no_data, 0)
            no_feedback = np.concatenate(self._no_feedback, 0)
        else:
            data, error, no_data, no_feedback = None, None, None, None
        # Remember to respond to the previous step with prev_feedback.
        if len(self._prev_fb):
            self._prev_fb[-1][0] = prev_feedback
        else:
            assert prev_feedback is None, 'The first step cannot give feedback to its previous step'
        self._prev_fb.append([False, self._next_fb, self.cell_shape, self.part_size])
        self._next_fb = []
        self.discard()
        while True:
            feedback, callbacks, cell_shape, part_size = self._prev_fb[0]
            if isinstance(feedback, asyncio.Future):
                if not feedback.done(): break
                feedback = feedback.result()
            else:
                if feedback is False: break # Respond in-order.
            assert feedback is None or isinstance(feedback, np.ndarray)
            self._prev_fb.pop(0)
            _feedback(callbacks, feedback, cell_shape, part_size, self)
        return (data, error, no_data, no_feedback)
    async def wait(self, max_simultaneous_steps = 16):
        """
        `await sn.wait(max_simultaneous_steps = 16)`

        If called before each `sn.handle(…)`, will limit how many steps can be done at once, yielding if necessary.

        Particularly important for async feedback: if the handling loop never yields to other tasks, then they cannot proceed, and a deadlock occurs (and memory eventually runs out).
        """
        assert isinstance(max_simultaneous_steps, int) and max_simultaneous_steps > 0
        self.n += 1
        if self.n >= max_simultaneous_steps: self.n = 0
        if self.n == 0:
            await asyncio.sleep(0)
        if len(self._prev_fb) <= max_simultaneous_steps: return
        fb = self._prev_fb[0][0] # The oldest feedback, must be done.
        if isinstance(fb, asyncio.Future) and not fb.done():
            await fb
    def discard(self):
        """Clears all scheduled-to-be-sent data."""
        try:
            _feedback(self._next_fb, None, self.cell_shape, self.part_size, self)
        finally:
            self._cell = 0
            self._data.clear()
            self._error.clear()
            self._no_data.clear()
            self._no_feedback.clear()
            self._next_fb.clear()



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
    def name(self, data, cell_shape, part_size, fill=None):
        """
        1D to 2D.
        """
        if data is None: return
        assert len(data.shape) == 1
        self._name_parts(cell_shape, part_size)
        # Pad & reshape `data`.
        data_size = cell_shape[-1]
        name_size = self.cell_size - data_size
        length = data.shape[0]
        cells = -(-length // data_size)
        total = cells * data_size
        data = _fill(data, total, 0) # Can't make it smaller, so `_unfill` will never have to make feedback larger.
        data = np.reshape(data, (cells, data_size))
        # Finalize the name, then concat it before `data`.
        if fill is not None:
            name = np.full((cells, name_size), fill)
            return np.concatenate((name, data), 1)
        start = np.expand_dims(np.arange(0, total, data_size), -1)
        end = np.minimum(start + data_size, length)
        name = np.concatenate([_fill(np.concatenate([x(start, end, total) if callable(x) else np.repeat(x, cells, 0) for x in p], 1), part_size, 1) if isinstance(p, list) else np.repeat(p, cells, 0) for p in self.name_parts], 1) # TODO: Can we also cache this result, based on `cells`, for much better performance?
        name = _fill(name, name_size, 1)
        return np.concatenate((name, data), 1)
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
def _feedback(callbacks, feedback, cell_shape, part_size, handler):
    fb = None
    got_err = None
    for on_feedback, expected_shape, start_cell, end_cell, namer, length, original_shape in callbacks:
        if feedback is not None:
            fb = feedback[start_cell:end_cell, :]
            assert fb.shape == expected_shape
            if namer is not None: fb = namer.unname(fb, length, cell_shape, part_size)
            fb = fb.reshape(original_shape)
        try:
            on_feedback(fb, cell_shape, part_size, handler) if callable(on_feedback) else on_feedback.set_result(fb)
        except KeyboardInterrupt as err:
            got_err = err
        except Exception as err:
            got_err = err
    if got_err is not None: raise got_err
def _inty(n):
    return isinstance(n, int) and n>=0





default = Handler()
sensors = default.sensors
cell_shape, part_size, cell_size = default.cell_shape, default.part_size, default.cell_size
def shape(*k, **kw):
    global cell_shape, part_size, cell_size
    r = default.shape(*k, **kw)
    cell_shape, part_size, cell_size = default.cell_shape, default.part_size, default.cell_size
    return r
def send(*k, **kw):
    return default.send(*k, **kw)
def handle(*k, **kw):
    return default.handle(*k, **kw)
def discard(*k, **kw):
    return default.discard(*k, **kw)
def get(*k, **kw):
    return default.get(*k, **kw)
def wait(*k, **kw):
    return default.wait(*k, **kw)
shape.__doc__ = Handler.shape.__doc__
send.__doc__ = Handler.send.__doc__
handle.__doc__ = Handler.handle.__doc__
discard.__doc__ = Handler.discard.__doc__
get.__doc__ = Handler.get.__doc__
wait.__doc__ = Handler.wait.__doc__