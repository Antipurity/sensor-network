import numpy as np
import hashlib



class Handler:
    """
    TODO:

    If needed, read `.cell_shape` or `.part_size` or `.cell_size` wherever the object is available.
    """
    def __init__(self, cell_shape=None, part_size=None):
        self._data = []
        self._error = []
        self._no_data = []
        self._no_feedback = []
        self._prev_fb = [] # [prev_feedback, …, […, (on_feedback, start_cell, end_cell), …], …]
        self._next_fb = [] # […, (on_feedback, start_cell, end_cell), …]
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
    # TODO: send(self, name=None, data, error=None, on_feedback=None)  (`sn.handle` concatenates these) (NumPy data tensors get flattened) (if `data` is a plain number, that's how many no-data cells are being requested) (`on_feedback(feedback, sn)`, called in-order so that re-using a function doesn't break things)
    # TODO: `handle(self, cell_shape, prev_feedback) → (data, error, no_data, no_feedback)` (if `prev_feedback` is a function, it'll be called with no args on each subsequent handling until it returns a non-`None` NumPy tensor; else it should be a NumPy tensor or `None`)
    #   TODO: Call each of self.sensors with `sn`.
    #   TODO: Concat all the collected data (& error).
    #   TODO: Empty all our memory of data.
    #     TODO: Push self._next_fb to self._prev_fb.
    #     TODO: While len(self._prev_fb) and its first item's first item is either a NumPy array or returns non-None, deconstruct the feedback and call all the `on_feedback` funcs.
    #       TODO: ...Don't we want to do the same at `handle`'s start, until it can no longer be done?...
    #   TODO: Return it.
    def discard(self):
        """Clears all scheduled-to-be-sent data."""
        for on_feedback, start_cell, end_cell in self._next_fb:
            try:
                on_feedback(None, self)
            except KeyboardInterrupt:
                raise
            except Exception as err:
                print(err)
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
    # TODO: name(data)→data2
    #   …Which just consists of putting the header in, right?… …Then again, the header is variable due to functions… And should probably be fractal-folded too…
    # TODO: unname(feedback)→feedback2



def _shape_ok(cell_shape, part_size: int):
    assert isinstance(part_size, int) and part_size > 0
    assert isinstance(cell_shape, list) or isinstance(cell_shape, tuple)
    assert all(isinstance(s, int) and s > 0 for s in cell_shape)
    assert len(cell_shape) == 4
    assert all(s % part_size == 0 for s in cell_shape[:-1])
def _str_to_floats(string: str):
    hash = hashlib.md5(string.encode('utf-8')).digest()
    return np.frombuffer(hash, dtype=np.uint8).astype(np.float32)/255.*2. - 1.
# TODO: Do we want fractal un/filling too?



default = Handler()
# TODO: Also expose method-like funcs that defer to `default`'s stuff.