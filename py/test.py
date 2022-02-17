"""
Tests and benchmarks for this Python implementation of a sensor network.

TODO: Should we include a chart of most-recent benchmark results, or something?

To measure [test coverage](http://www.kaner.com/pdfs/pnsqc00.pdf), use [Coverage](https://coverage.readthedocs.io/en/6.3.1/) or an equivalent. Should be 100% or there's a problem.
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
    h.send(name=(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1), data=np.array([-.4, -.2, .2, .4]))
    data, error, no_data, no_feedback = h.handle()
    assert data.shape == (3, 96)
def test3():
    """Named error."""
    h = sn.Handler((8, 24, 64), 8)
    def yes_feedback(fb, *_): assert fb is not None
    h.send(name=('test',), data=np.zeros((16,)), error=np.full((16,), -.5), on_feedback=yes_feedback)
    h.handle()
    h.handle(np.zeros((1, 96)))
def test4():
    """Name's error."""
    h = sn.Handler((8, 24, 64), 8)
    try:
        h.send(name=(True,), data=np.array([1.])); assert False
    except TypeError:
        pass
def test5():
    """Sensors are auto-called at each step."""
    h = sn.Handler((8, 24, 64), 8)
    def eh_feedback(fb, *_): assert fb is not None
    h.sensors.append(lambda h: h.send(name=('test',), data=np.array([.1, .2, .3]), on_feedback=eh_feedback))
    h.sensors.append(lambda h: h.send(name=('test',), data=np.array([.4, .5, .6]), on_feedback=eh_feedback))
    assert h.handle()[0].shape == (2, 96)
    h.handle(np.zeros((2, 96)))
def test6():
    """Errors thrown by `on_feedback` are re-thrown."""
    h = sn.Handler((8, 24, 64), 8)
    def err1(*_): raise KeyboardInterrupt()
    def err2(*_): raise TypeError('damn')
    h.send(data=None, on_feedback=err1)
    assert h.handle()[0].shape == (0, 96)
    try: h.handle(); assert False
    except KeyboardInterrupt: pass
    h.send(data=np.zeros((5, 96)), on_feedback=err2)
    assert h.handle()[0].shape == (5, 96)
    try: h.handle(); assert False
    except TypeError: pass
def test7():
    """Deferring feedback via a function."""
    h = sn.Handler((8, 24, 64), 8)
    n, got = 0, False
    def yes_feedback(fb, *_): nonlocal got;  assert fb is not None;  got = True
    def fb(): nonlocal n; n+=1; return [None, None, np.zeros((2,96))][n-1]
    h.send(data=np.zeros((2,96)), on_feedback=yes_feedback)
    assert h.handle()[0].shape == (2,96)
    h.handle(fb)
    h.handle()
    assert got is False
    h.handle()
    assert got is True
test0()
test1()
test2()
test3()
test4()
test5()
test6()
test7()
# TODO: Allow `None` to be a part of the name (zero-filling). …Or, start zero-fill every part of the name except `cell_shape[-2]`, to match JS behavior.
# TODO: Also send "no-data" as a number requesting that-many-numbers. Via h.get, and async handling.
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
        assert fb is not None and fb.shape == (N,) and fb[0] == .2
    send_data = np.random.randn(N)
    start, duration = time.monotonic(), 10.
    name = sn.Namer('benchmark')
    while time.monotonic() - start < duration:
        sn.send(name, data=send_data, on_feedback=check_feedback)
        data, error, no_data, no_feedback = sn.handle(feedback)
        feedback = np.full_like(data, .2) if data is not None else None
        iterations += 1
    sn.discard()
    thr = N*4 * iterations / duration
    print('With', N, 'values, throughput:', thr, 'bytes/sec', f'({round(thr/1024/1024*100)/100} MiB/s)', f'({round(thr/N/4*100)/100} it/s)')
benchmark(64*10) # TODO: Mirror the JS basic benchmark.