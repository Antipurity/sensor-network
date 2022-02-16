"""
TODO:

To measure [test coverage](http://www.kaner.com/pdfs/pnsqc00.pdf), use [Coverage](https://coverage.readthedocs.io/en/6.3.1/) or an equivalent.
"""
#   (The interface is not hard to use for the "functions that wait for the handler's decision, where some can spawn new functions and such", right?)
#     (Its flexibility & convenience as an RL interface is kinda amazing. There are just no limitations, at all, there's only writing down ideas.)



import sensornet as sn
import numpy as np
import time



def test0():
    """No shape, so no handling."""
    def no_feedback(fb, *_): assert fb is None
    h = sn.Handler()
    assert h.handle() == (None, None, None, None)
    h.send()
    h.send(data=np.array([1.]))
    h.send(data=np.array([1.]), on_feedback = no_feedback)
def test1():
    """Already-named data, and transmission error."""
    h = sn.Handler((8, 24, 64), 8)
    h.send(data = np.zeros((3, 96)), error = np.full((3, 96), -1.))
    data, error, no_data, no_feedback = h.handle()
    assert (data == np.zeros((3, 96))).all()
    assert (error == np.full((3, 96), -1.)).all()
    assert (no_data == np.array([False, False, False])).all()
    assert (no_feedback == np.array([True, True, True])).all()
def test2():
    """Different kinds of names."""
    h = sn.Handler((8, 24, 64), 8)
    h.send(name=('test',), data=np.array([-.4, -.2, .2, .4]))
    h.send(name=('test', -.2, .2, lambda start,end,total: start/total*2-1), data=np.array([-.4, -.2, .2, .4]))
    data, error, no_data, no_feedback = h.handle()
    assert data.shape == (2, 96)
test0()
test1()
test2()
# TODO: Allow `None` to be a part of the name (zero-filling). …Or, start zero-fill every part of the name except `cell_shape[-2]`, to match JS behavior.
# TODO: And all the other tests, as many as we need to bring the coverage up to 100%.
# TODO: Also try sending None as data, and still have on_feedback.
# TODO: Also send "no-data" as a number requesting a cell-count. Via h.get, and async handling.
# TODO: Also h.sensors.append(lambda h: ...).
# TODO:
#   try:
#     h.send(name=(False,), data=np.array([1.]))
#     raise RuntimeError('Should have raised an error')
#   except AssertionError:
#     pass
print('Tests OK')



def benchmark(N=64*10):
    """Raw number-shuffling performance."""
    sn.shape((8, 24, 64), 8)
    assert sn.cell_shape == (8, 24, 64)
    assert sn.part_size == 8
    assert sn.cell_size == 96
    iterations, feedback = 0, None
    N = 64*10
    def check_feedback(fb, *_):
        assert fb is None or fb.shape == (N,) and fb[0] == .2
    send_data = np.random.randn(N)
    start, duration = time.monotonic(), 10.
    name = sn.Namer(('benchmark',), sn.cell_shape, sn.part_size)
    while time.monotonic() - start < duration:
        sn.send(name, data=send_data, on_feedback=check_feedback)
        data, error, no_data, no_feedback = sn.handle(feedback)
        feedback = np.full_like(data, .2) if data is not None else None
        iterations += 1
    thr = N*4 * iterations / duration
    print('With', N, 'values, throughput:', thr, 'bytes/sec', f'({round(thr/1024/1024*100)/100} MiB/s)', f'({round(thr/N/4*100)/100} it/s)')
benchmark(64*10) # TODO: Mirror the JS basic benchmark.