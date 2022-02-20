"""
Tests and benchmarks for this Python implementation of a sensor network.

Expect around 10000 steps per second or less, which should be more than enough for any AI model.

```bash
python3 sensor-network/py/sensornet/test.py
```

Sample run:

```
Tests OK
With 4800 values, throughput: 289027200.0 bytes/sec (275.64 MiB/s) (15053.5 it/s)
With 9600 values, throughput: 509157120.0 bytes/sec (485.57 MiB/s) (13259.3 it/s)
With 14400 values, throughput: 674772480.0 bytes/sec (643.51 MiB/s) (11714.8 it/s)
With 19200 values, throughput: 835614720.0 bytes/sec (796.9 MiB/s) (10880.4 it/s)
With 24000 values, throughput: 916416000.0 bytes/sec (873.96 MiB/s) (9546.0 it/s)
With 28800 values, throughput: 1017285120.0 bytes/sec (970.16 MiB/s) (8830.6 it/s)
With 33600 values, throughput: 1075200000.0 bytes/sec (1025.39 MiB/s) (8000.0 it/s)
With 38400 values, throughput: 707143680.0 bytes/sec (674.38 MiB/s) (4603.8 it/s)
With 43200 values, throughput: 711987840.0 bytes/sec (679.0 MiB/s) (4120.3 it/s)
With 48000 values, throughput: 820915200.0 bytes/sec (782.89 MiB/s) (4275.6 it/s)
```

To measure [test coverage](http://www.kaner.com/pdfs/pnsqc00.pdf), use [Coverage](https://coverage.readthedocs.io/en/6.3.1/) or an equivalent. Should be 100% or there's a problem.

```bash
coverage run --branch sensor-network/py/sensornet/test.py
coverage report
coverage html
```
"""



import __init__ as sn
import numpy as np
import asyncio
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
    """Non-1D data and feedback."""
    h = sn.Handler((8, 24, 64), 8)
    got = False
    def yes_feedback(fb, *_): nonlocal got;  assert fb.shape == (2,3,4);  got = True
    h.send(name=('test',), data=np.zeros((2,3,4)), on_feedback=yes_feedback)
    assert h.handle()[0].shape == (1,96)
    h.handle(np.zeros((1,96)))
    assert got
def test8():
    """Async operations."""
    sn.shape((0, 32, 64), 8)
    sn.shape((0, 32, 64), 8)
    assert sn.cell_shape == (0, 32, 64)
    assert sn.part_size == 8
    assert sn.cell_size == 96
    name = sn.Namer('test')
    n = 0
    finished = 0
    async def request_data(h, maybe=False):
        nonlocal finished
        fb = await (h.send(name, (3,5), reward=None, on_feedback=True) if maybe else h.get(name, (3,5)))
        finished += 1
        if not maybe: assert fb.shape == (3,5)
    async def give_feedback_later(data, error, no_data, no_feedback):
        nonlocal n
        await asyncio.sleep(.1)
        n += 1
        return data if n==30 or n>60 else None
    async def main():
        for _ in range(5):
            asyncio.ensure_future(request_data(sn))
        asyncio.ensure_future(request_data(sn, True))
        fb = None
        while finished < 6:
            await sn.wait(16)
            fb = give_feedback_later(*sn.handle(fb))
        await asyncio.sleep(.1)
        await fb # To silence a warning.
        await sn.wait(1) # For test coverage.
        sn.send()
        sn.discard()
    asyncio.run(main())
def test9():
    """Pass-through of (synthetic) handler data to another one."""
    h = sn.Handler((8, 24, 64), 8)
    def yes_feedback(fb, *_): assert fb.shape == (13, 96)
    cells, shape = (13,), (13, 96)
    h.send(name=None, reward=None, data=np.zeros(shape), error=np.zeros(shape), no_data=np.full(cells, True), no_feedback=np.full(cells, False), on_feedback=yes_feedback)
    assert h.handle()[0].shape == shape
    h.handle(np.zeros(shape))
def test10():
    """PyTorch tensor GPU→CPU async transfer."""
    try:
        import torch # This statement takes so long. So long. So long. So long.
        async def main():
            a = torch.tensor([1., 2., 3.], device='cpu')
            b = torch.tensor([1., 2., 3.], device='cuda' if torch.cuda.is_available() else 'cpu')
            assert ((await sn.torch(torch, a)) == (await sn.torch(torch, b))).all()
        asyncio.run(main())
    except ImportError: pass # pragma: no cover
test0()
test1()
test2()
test3()
test4()
test5()
test6()
test7()
test8()
test9()
test10()
print('Tests OK')



async def benchmark(N=64*10):
    """Raw number-shuffling performance."""
    h = sn.Handler((8, 24, 64), 8)
    iterations, feedback = 0, None
    def check_feedback(fb, *_):
        assert fb is not None and fb.shape == (N,) and fb[0] == .2
    send_data = np.random.randn(N)
    start, duration = time.monotonic(), 10.
    name = sn.Namer('benchmark')
    while time.monotonic() - start < duration:
        await h.wait()
        # Using Futures is a 30% slowdown.
        h.send(name, data=send_data, on_feedback=check_feedback)
        data, error, no_data, no_feedback = h.handle(feedback)
        feedback = np.full_like(data, .2) if data is not None else None
        iterations += 1
    h.discard()
    thr = N*4 * (96/64) * iterations / duration
    print('With', N*96//64, 'values, throughput:', thr, 'bytes/sec', f'({round(thr/1024/1024*100)/100} MiB/s)', f'({round(iterations/duration*100)/100} it/s)')
for i in range(10):
    asyncio.run(benchmark(64*50 * (i+1)))