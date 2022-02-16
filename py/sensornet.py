"""
Module for differentiable sensor networks: each gathers numeric data from anywhere, and in a loop, handles it (and sends feedback back if requested).

Position-invariant and numeric: these constraints allow AI models to disregard all concerns about data formats, and with only a couple lines of code on the sensor-side and no changes on the handler-side: perform cross-dataset meta-learning and multimodal learning and multi-objective reinforcement learning and model distillation, combine tasks at runtime, and learn in completely arbitrary environments.

Python 3.4 or newer (for asyncio).

---

TODO: After we have the tests, have a mini-tutorial on the proper use.
    TODO: Mention that users can use the top-level module, or equivalently, create a `Handler` and call methods on that, and/or use many `Handler`s.
"""



import numpy as np
import hashlib
import asyncio



class Handler:
    """
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
        self._next_fb = [] # […, (on_feedback, shape, start_cell, end_cell, namer, length), …]
        self.sensors = [] # Called by `.handle(…)`.
        self.cell_shape = ()
        self.part_size = 0
        self.cell_size = 0
        if cell_shape is not None or part_size is not None:
            self.shape(cell_shape, part_size)
    def shape(self, cell_shape, part_size):
        """Changes the current shape."""
        _shape_ok(cell_shape, part_size)
        self.discard()
        self.cell_shape = cell_shape
        self.part_size = part_size
        self.cell_size = sum(cell_shape)
    def send(self, name=None, data=None, error=None, reward=0., on_feedback=None):
        """
        Sends named data, possibly getting same-size feedback, possibly only getting feedback.

        Arguments:
        - `name = None`:
            - If a tuple/list of strings and -1…1 numbers and functions to -1…1 numbers from start-number & end-number & total-numbers NumPy arrays, converted to a `Namer`.
            - If a `Namer`, it is used.
            - If `None`, `data` & `error` must already incorporate the name and be sized `cells×cell_size`.
        - `data = None`: `None`, or how many numbers of no-data feedback to return, or a 1D NumPy array of -1…1 numbers.
        - `error = None`: data transmission error: `None` or a `data`-sized float32 array of `abs(true_data - data) - 1`. -1…1.
        - `reward = 0.`: rates prior performance of these cells, for reinforcement learning.
        - `on_feedback = None`: a function from `feedback` (could be `None`), `cell_shape`, `part_size`, `handler`, to nothing.
            - Neither `.send` nor `.handle` steps are never re-ordered, so to reduce memory allocations, could reuse the same function and use queues.

        Returns `None`. See `.maybe_get` and `.get` for more convenient `asyncio`-based interfaces.
        """
        if name is None and on_feedback is None: return
        if not self.cell_size:
            on_feedback(None, self.cell_shape, self.part_size, self)
            return
        assert name is None or isinstance(name, tuple)
        assert data is None or isinstance(data, int) and data>=0 or isinstance(data, np.ndarray)
        assert error is None or isinstance(error, np.ndarray)
        assert on_feedback is None or callable(on_feedback)
        no_feedback = not on_feedback
        no_data = False if isinstance(data, np.ndarray) else True
        if no_data: data = np.zeros((data or 0, self.cell_size), dtype=np.float32)
        if name is not None:
            if len(data.shape) != 1: data = data.flatten()
            if error and len(error.shape) != 1: error = error.flatten()
        # Name.
        if isinstance(name, tuple) or isinstance(name, list):
            name = Namer(name, self.cell_shape, self.part_size)
        if isinstance(name, Namer):
            assert name.cell_shape == self.cell_shape
            assert name.part_size == self.part_size
            data = name.name(data, None)
            error = name.name(error, -1.)
        else:
            assert name is None
        assert len(data.shape) == 2
        assert data.shape[-1] == self.cell_size
        assert error is None or data.shape == error.shape
        # Reward.
        data[:, 0] = reward
        # Send.
        cells = data.shape[0]
        shape = (cells,)
        self._data.append(data)
        self._error.append(error)
        self._no_data.append(np.tile(no_data, shape))
        self._no_feedback.append(np.tile(no_feedback, shape))
        self._next_fb.append((on_feedback, data.shape, self._cell, self._cell + cells, name))
        self._cell += cells
    def maybe_get(self, name, len, reward=0.):
        """
        Wraps `.send` to allow `await`ing a 1D tensor from the handler, or `None`.
        """
        # `asyncio.get_running_loop().create_future()` is better for customization-by-the-loop reasons, but imposes Python 3.7.
        fut = asyncio.Future()
        self.send(name, len, None, reward, lambda feedback, cell_shape, part_size, handler: fut.set_result(feedback))
        return fut
    async def get(self, name, len, reward=0.):
        """
        Wraps `.send` to allow `await`ing a 1D tensor from the handler. Never returns `None`, instead re-requesting until a numeric result is available.
        """
        while True:
            fb = self.maybe_get(name, len, reward)
            if fb is not None: return fb
    def handle(self, prev_feedback=None):
        """
        Handles collected data.

        Pass it the previous handling's feedback (as a NumPy array or `None`), or if not immediately available (i.e. needs GPU→CPU transfer), a function with no inputs that will return `None` or the feedback.

        This returns `(data, error, no_data, no_feedback)`.
        - `data`: a float32 array of already-named cells of data, sized `cells×cell_size`. -1…1.
        - `error`: data transmission error: `None` or a `data`-sized float32 array of `abs(true_data - data) - 1`. -1…1.
        - `no_data`: a bit-mask, sized `cells`.
        - `no_feedback`: a bit-mask, sized `cells`.
        - Usage:
            - `data[:, 0]` would return per-cell rewards.
            - `numpy.compress(~no_data, data)` would select only inputs.
            - `numpy.compress(~no_feedback, data)` would select only queries.
            - `numpy.put(numpy.zeros_like(data), numpy.where(~no_feedback)[0], feedback)` would put back the selected queries in-place, making `data` suitable for `prev_feedback` here.
        """
        assert prev_feedback is None or callable(prev_feedback)
        # Collect sensor data.
        for s in self.sensors: s(self)
        if not len(self._data): return
        # Gather data.
        data = np.concatenate(self._data, 0)
        error = np.concatenate([e or -np.ones((0, self.cell_size)) for e in self._error], 0) if any(self._error) else None
        no_data = np.concatenate(self._no_data, 0)
        no_feedback = np.concatenate(self._no_feedback, 0)
        # Forget this step's data, and report feedback.
        if prev_feedback is not None:
            self._prev_fb.append((prev_feedback, self._next_fb, self.cell_shape, self.part_size))
        else:
            for on_feedback, expected_shape, start_cell, end_cell, namer, length in self._next_fb:
                on_feedback(None, self.cell_shape, self.part_size, self)
        self.discard()
        while len(self._prev_fb):
            feedback, callbacks, cell_shape, part_size = self._prev_fb[0]
            if callable(feedback): feedback = feedback()
            if feedback is None: break
            assert isinstance(feedback, np.ndarray)
            self._prev_fb.pop(0)
            for on_feedback, expected_shape, start_cell, end_cell, namer, length in callbacks:
                fb = feedback[start_cell:end_cell, :]
                assert fb.shape == expected_shape
                fb = namer.unname(fb, length)
                on_feedback(fb, cell_shape, part_size, self)
        return (data, error, no_data, no_feedback)
    def discard(self):
        """Clears all scheduled-to-be-sent data."""
        for on_feedback, expected_shape, start_cell, end_cell, namer, length in self._next_fb:
            try:
                on_feedback(None, self)
            except KeyboardInterrupt:
                raise
            except Exception as err:
                print(err)
        self._cell = 0
        self._data.clear()
        self._error.clear()
        self._no_data.clear()
        self._no_feedback.clear()
        self._next_fb.clear()



class Namer:
    """
    A class for augmenting a 1D array with numeric names, into a 2D array, sized cells×cell_size.

    Pre-constructing this instead of passing in names to `handler.send` might be faster.
    """
    def __init__(self, name, cell_shape, part_size):
        assert isinstance(name, list) or isinstance(name, tuple)
        _shape_ok(cell_shape, part_size)
        self.cell_shape = cell_shape
        self.part_size = part_size
        self.cell_size = sum(cell_shape)
        name_parts = []
        nums = []
        for part in name:
            if isinstance(part, str):
                name_parts.append(_str_to_floats(part))
            elif isinstance(part, float) or isinstance(part, int) or callable(part):
                nums.append(part)
                if len(nums) >= part_size:
                    name_parts.append(nums)
                    nums = []
            else:
                raise TypeError("Names must consist of strings, numbers, and number-returning functions")
        if len(nums): name_parts.append(nums)
        self.name_parts = name_parts
    def name(self, data, fill=None):
        """
        1D to 2D.
        """
        assert len(data.shape) == 1
        # Pad & reshape `data`.
        cells = -(-data.shape[0] // self.cell_size)
        total = cells * self.cell_size
        data = np.reshape(_fill(data, total, 0), (cells, self.cell_size))
        # Finalize the name, then concat it before `data`.
        if fill is not None:
            name = np.full((cells, self.cell_size - self.cell_shape[-1]), fill)
            return np.concatenate((), 1)
        start = np.arange(0, total, self.cell_size)
        end = start + self.cell_size
        name = [_fill(np.array([x(start, end, total) if callable(x) else x for x in p]) if isinstance(p, list) else p, self.part_size, 1) for p in self.name_parts]
        return np.concatenate([*name, data], 1)
    def unname(self, feedback, length):
        """
        Reverses `.name`.
        """
        # Ignore the name, only heed data.
        assert len(feedback.shape) == 2 and feedback.shape[-1] == self.cell_size
        feedback = feedback[:, -self.cell_shape[-1]:]
        return _unfill(np.flatten(feedback), length, 0)



def _shape_ok(cell_shape: tuple, part_size: int):
    assert isinstance(part_size, int) and part_size > 0
    assert isinstance(cell_shape, tuple)
    assert all(isinstance(s, int) and s > 0 for s in cell_shape)
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
    if x.shape[axis] == size: return x # TODO: Why is this index non-existent?
    if x.shape[axis] > size: return np.take(x, range(0,size), axis)
    folds = [x]
    for _ in range(1, -(-size // x.shape[axis])):
        folds.append(1 - 2 * np.abs(folds[-1]))
    x = np.concatenate(folds, 0)
    if x.shape[axis] == size: return x
    return np.take(x, range(0,size), axis)
def _unfill(y, size, axis=0): # → x
    """Undoes `_fill(x, y.shape[axis], axis)→y` via `_unfill(y, x.shape[axis], axis)→x`.
    
    `(x,y) → (copysign((1-y)/2, x), y)`"""
    if y.shape[axis] == size: return y
    if y.shape[axis] < size:
        assert axis == 0 # Good enough for us.
        return np.pad(y, (0, size - y.shape[axis]))
    folds = np.split(y, range(0, y.shape[axis], size), axis)
    if folds[-1].shape[0] < size:
        folds[-1] = np.concatenate((folds[-1], 1 - 2 * np.abs(np.take(folds[-2], range(folds[-1].shape[0], size), axis))), 0)
    for i in reversed(range(1, -(y.shape[axis] // -size))):
        x, y = folds[i-1], folds[i]
        folds[i-1] = np.copysign(.5 * (1-y), x)
    return folds[0]
    
    



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
def maybe_get(*k, **kw):
    return default.maybe_get(*k, **kw)
def get(*k, **kw):
    return default.get(*k, **kw)