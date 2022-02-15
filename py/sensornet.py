import numpy as np
import hashlib



class Handler:
    """
    TODO:

    TODO: Mention that the top-level `sensornet` module acts like an instance of this class, but if more processing streams are needed, can construct them.

    To gather data automatically, do `handler.sensors.append(lambda handler: ...)`.

    If needed, read `.cell_shape` or `.part_size` or `.cell_size` wherever the object is available.

    TODO: After we have a test, have a mini-tutorial on the proper use.
    """
    def __init__(self, cell_shape=None, part_size=None):
        self._cell = 0
        self._data = []
        self._error = []
        self._no_data = []
        self._no_feedback = []
        self._prev_fb = [] # [prev_feedback, …, _next_fb, …]
        self._next_fb = [] # […, (on_feedback, shape, start_cell, end_cell, namer), …]
        self.sensors = [] # Called by `.handle(…)`.
        self.cell_shape = ()
        self.part_size = 0
        self.cell_size = 0
        if cell_shape is not None or part_size is not None:
            self.shape(cell_shape, part_size)
    def shape(self, cell_shape, part_size):
        _shape_ok(cell_shape, part_size)
        self.discard()
        self.cell_shape = cell_shape
        self.part_size = part_size
        self.cell_size = sum(cell_shape)
    def send(self, name=None, data=None, error=None, on_feedback=None):
        """
        TODO:
        TODO: on_feedback(feedback, cell_shape, part_size); `feedback` might be `None`.
        TODO: data is None or a number (how many no-data cells to request, as an action) or a NumPy array (flat if `name`, else 2D)
        """
        if name is None and on_feedback is None: return
        if not self.cell_size: return on_feedback(None, self.cell_shape, self.part_size)
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
        # Send.
        cells = data.shape[0]
        shape = (cells,)
        self._data.append(data)
        self._error.append(error)
        self._no_data.append(np.tile(no_data, shape))
        self._no_feedback.append(np.tile(no_feedback, shape))
        self._next_fb.append((on_feedback, data.shape, self._cell, self._cell + cells, name))
        self._cell += cells
    def handle(self, prev_feedback=None):
        """
        TODO:
        TODO: returns (data, error, no_data, no_feedback); describe each of these.
            TODO: Usage: `numpy.compress(~no_data, data)` would select only inputs; `numpy.compress(~no_feedback, data)` would select only queries; `numpy.put(data, numpy.where(~no_feedback)[0], feedback)` would put back the selected queries in-place, making `data` suitable for `prev_feedback` here.
        TODO: if `prev_feedback` is a function, it'll be called with no args on each subsequent handling until it returns a non-`None` NumPy tensor; else it should be a NumPy tensor or `None`
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
            for on_feedback, expected_shape, start_cell, end_cell, namer in self._next_fb:
                on_feedback(None, self.cell_shape, self.part_size)
        self.discard()
        while len(self._prev_fb):
            feedback, callbacks, cell_shape, part_size = self._prev_fb[0]
            if callable(feedback): feedback = feedback()
            if feedback is None: break
            assert isinstance(feedback, np.ndarray)
            self._prev_fb.pop(0)
            for on_feedback, expected_shape, start_cell, end_cell, namer in callbacks:
                fb = feedback[start_cell:end_cell, :]
                assert fb.shape == expected_shape
                fb = namer.unname(fb)
                on_feedback(fb, cell_shape, part_size)
        return (data, error, no_data, no_feedback)
    def discard(self):
        """Clears all scheduled-to-be-sent data."""
        for on_feedback, start_cell, end_cell in self._next_fb:
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
    TODO:
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
        TODO:
        """
        assert len(data.shape) == 1
        # TODO: ...Do we want to compute `cells` from data.shape[0] and self.cell_size (divide rounding up) here, and zero-pad `data`?...
        #   Or rather, fractal-fold `data`: _fill(data, size, 0).
        # TODO: If fill is not None, take np.full((cells, self.cell_size - self.cell_shape[-1]), fill) as the full name tensor, and concat with the reshaped data.
        # TODO: name(data, fill=None)→data2
        #   …Which just consists of putting the header in, right?… …Then again, the header is variable due to functions… And should probably be fractal-folded too…
        #   …Do we iterate over all cells in Python, and copy data manually?…
        #     …Or should we come up with some NumPy solution? Maybe even try to get away with operating on NumPy tensors in calls to funcs? …How to do that…
        #     Kinda want each name part to have its own tensor, then after we have all that (each fractally-folded to be sized self.part_size) and the reshaped data, just concat all.
    # TODO: unname(feedback)→feedback2



def _shape_ok(cell_shape: tuple, part_size: int):
    assert isinstance(part_size, int) and part_size > 0
    assert isinstance(cell_shape, tuple)
    assert all(isinstance(s, int) and s > 0 for s in cell_shape)
    assert len(cell_shape) == 4
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
    # TODO: ...What if y.shape[axis] < size?... Zero-pad? How?
    # TODO: Pad `y` so that y.shape[axis]%size==0, then folds = np.split(y, range(0, y.shape[axis], size), axis)
    # TODO: Pad the last array so that the rest is filled with 1-2*np.abs(prev_to_last), to not lose precision.
    # TODO: Going in reverse over `folds`, where `x` is the prev elem and `y` is its next elem, do `x[...] = np.copysign(.5 * (1-y), x)`
    # TODO: Return folds[0].



default = Handler()
sensors = default.sensors
# TODO: Also expose method-like funcs that defer to `default`'s stuff.
#   (And cell_shape, part_size, cell_size, all updated after `.shape(…)` returns.)



# TODO: Also launch tests if this module is executed directly: correctness, then throughput.
#   (The interface is not hard to use for the "functions that wait for the handler's decision, where some can spawn new functions and such", right? Not quite asyncio levels of simplicity, but still... ...Could we integrate with asyncio too?...)
#     (Its flexibility as an RL interface is kinda amazing. There are just no limitations, at all, there's only writing down ideas.)
#     `'asyncio' in sys.modules` and `sys.modules['asyncio']` with py3.3+ could check whether it has been imported already. Which introduces import-order requirements, which is bad UX.
#     `try: import asyncio; except ImportError: ...` with py3.6+ for robust checking.
#     …What would an asyncio interface look like exactly, though? And, can't users just easily implement it anyway if they need to?