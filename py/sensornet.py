import numpy as np
import hashlib



class Handler:
    """
    TODO:

    To gather data automatically, do `handler.sensors.append(lambda handler: ...)`.

    If needed, read `.cell_shape` or `.part_size` or `.cell_size` wherever the object is available.
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
    # TODO: name(data, fill=None)→data2
    #   …Which just consists of putting the header in, right?… …Then again, the header is variable due to functions… And should probably be fractal-folded too…
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
# TODO: Do we want fractal un/filling too?



default = Handler()
# TODO: Also expose method-like funcs that defer to `default`'s stuff.



# TODO: Also launch tests if this module is executed directly: correctness, then throughput.
#   (The interface is not hard to use for the "functions that wait for the handler's decision, where some can spawn new functions and such", right? Not quite asyncio levels of simplicity, but still... ...Could we integrate with asyncio too?...)