"""
Tests and benchmarks for this Python implementation of a sensor network.

Not counting other bottlenecks, expect up to around 10000 steps per second, which should be enough for most AI models.

```bash
python3 sensor-network/py/sensornet/test.py
```

Sample run:

```
Tests OK
With 4800 values, throughput: 215731200.0 bytes/sec (205.74 MiB/s) (11236.0 it/s)
With 9600 values, throughput: 402485760.0 bytes/sec (383.84 MiB/s) (10481.4 it/s)
With 14400 values, throughput: 550494720.0 bytes/sec (524.99 MiB/s) (9557.2 it/s)
With 19200 values, throughput: 696384000.0 bytes/sec (664.12 MiB/s) (9067.5 it/s)
With 24000 values, throughput: 812592000.0 bytes/sec (774.95 MiB/s) (8464.5 it/s)
With 28800 values, throughput: 899758080.0 bytes/sec (858.08 MiB/s) (7810.4 it/s)
With 33600 values, throughput: 991656960.0 bytes/sec (945.72 MiB/s) (7378.4 it/s)
With 38400 values, throughput: 1091082240.0 bytes/sec (1040.54 MiB/s) (7103.4 it/s)
With 43200 values, throughput: 1165933440.0 bytes/sec (1111.92 MiB/s) (6747.3 it/s)
With 48000 values, throughput: 1256160000.0 bytes/sec (1197.97 MiB/s) (6542.5 it/s)
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
    h = sn.Handler()
    assert h.handle()[0].shape == (0,0)
    h.data(data=np.array([1.]))
    assert h.handle()[0].shape == (0,0)
def test1():
    """Already-named data, and transmission error."""
    h = sn.Handler((8, 24, 64), 8)
    h.data(data = np.zeros((3, 96)), error = np.full((3, 96), -.7))
    data, query, data_error, query_error = h.handle()
    assert (data == np.zeros((3, 96))).all()
    assert (query == np.zeros((0, 32))).all()
    assert (data_error == np.full((3, 96), -.7)).all()
    assert query_error == None
def test2():
    """Different kinds of names."""
    h = sn.Handler((8, 24, 64), 8)
    h.data(name=('test',), data=np.array([-.4, -.2, .2, .4]))
    h.data(name=('test', -.2, .2, lambda start,end,total: start/total*2-1), data=np.array([-.4, -.2, .2, .4]))
    h.data(name=(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1), data=np.array([-.4, -.2, .2, .4]))
    data, query, *_ = h.handle()
    assert data.shape == (3, 96)
def test3():
    """Named error."""
    h = sn.Handler((8, 24, 64), 8)
    def yes_feedback(fb, *_): assert fb is not None
    h.query(name=('test',), query=16, callback=yes_feedback)
    h.handle()
    h.handle(np.zeros((1, 96)))
def test4():
    """Name's error."""
    h = sn.Handler((8, 24, 64), 8)
    try:
        h.data(name=(True,), data=np.array([1.])); assert False
    except TypeError:
        pass
def test5():
    """Sensors are auto-called at each step."""
    h = sn.Handler((8, 24, 64), 8)
    def eh_feedback(fb, *_): assert fb is not None
    h.sensors.append(lambda h: h.query(name=('test',), query=3, callback=eh_feedback))
    h.sensors.append(lambda h: h.query(name=('test',), query=3, callback=eh_feedback))
    assert h.handle()[1].shape == (2, 32)
    h.handle(np.zeros((2, 96)))
def test6():
    """Errors thrown by `callback` are re-thrown."""
    h = sn.Handler((8, 24, 64), 8)
    def err1(*_): raise KeyboardInterrupt()
    def err2(*_): raise TypeError('damn')
    h.query(name='death', query=1, callback=err1)
    assert h.handle()[1].shape == (1, 32)
    try: h.handle(); assert False
    except KeyboardInterrupt: pass
    h.query(query=np.zeros((5, 32)), error=np.full((5, 32), -.5), callback=err2)
    assert h.handle()[1].shape == (5, 32)
    try: h.handle(); assert False
    except TypeError: pass
def test7():
    """Non-1D data and feedback."""
    h = sn.Handler((8, 24, 64), 8)
    got = False
    def yes_feedback(fb, *_): nonlocal got;  assert fb.shape == (2,3,4);  got = True
    h.data(name=('test',), data=np.zeros((2,3,4)))
    h.query(name=('test',), query=(2,3,4), callback=yes_feedback)
    data, query, *_ = h.handle()
    assert data.shape == (1,96) and query.shape == (1,32)
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
        fb = await (h.query(name, (3,5)) if maybe else h.get(name, (3,5)))
        finished += 1
        if not maybe: assert fb.shape == (3,5)
    async def give_feedback_later(data, query, data_error, query_error):
        nonlocal n
        await asyncio.sleep(.1)
        n += 1
        return np.zeros((query.shape[0], data.shape[1])) if n==30 or n>60 else None
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
        sn.discard()
    asyncio.run(main())
async def test9():
    """Pass-through of (synthetic) handler data to another one."""
    h = sn.Handler((8, 24, 64), 8)
    shape1, shape2 = (13,96), (13,32)
    fut = h.pipe(np.random.rand(*shape1)*2-1, np.random.rand(*shape2)*2-1, np.zeros(shape1), np.zeros(shape2))
    data, query, *_ = h.handle()
    assert data.shape == shape1 and query.shape == shape2
    h.handle(np.zeros(shape1))
    assert (await fut).shape == shape1
def test10():
    """PyTorch tensor GPU→CPU async transfer."""
    try:
        import torch # This statement takes so long. So long. So long. So long.
        def get(x):
            return sn.torch(torch, x, True)
        async def main():
            a = torch.tensor([1., 2., 3.], device='cpu')
            b = torch.tensor([1., 2., 3.], device='cuda' if torch.cuda.is_available() else 'cpu')
            assert ((await get(a)) == (await get(b))).all()
        asyncio.run(main())
    except ImportError: pass # pragma: no cover
def test11():
    """Low-level functions as substitutes for `asyncio.Future`s."""
    h = sn.Handler((8, 24, 64), 8)
    h.handle()
    n, got = 0, False
    def feedback():
        nonlocal n, got;  n += 1
        if n == 4: got = True
        return got and np.zeros((2, 96))
    h.data(None, np.zeros((2, 96)))
    h.handle(feedback)
    h.handle()
    h.handle()
    assert not got
    h.handle()
    assert got
test0()
test1()
test2()
test3()
test4()
test5()
test6()
test7()
test8()
asyncio.run(test9())
test10()
test11()
print('Tests OK')



async def benchmark(N=64*10):
    """Raw number-shuffling performance."""
    h = sn.Handler((8, 24, 64), 8)
    iterations, feedback = 0, None
    def check_feedback(fb, *_):
        assert fb is not None and fb.shape == (64,) and fb[0] == .2
    async def await_feedback(fut):
        check_feedback(await fut)
    send_data = np.random.randn(N)
    start, duration = time.monotonic(), 10.
    name = sn.Namer('benchmark')
    while time.monotonic() - start < duration:
        await h.wait()
        h.data(name, data=send_data)
        # asyncio.ensure_future(await_feedback(h.query(name, 64))) # 15% slowdown.
        h.query(name, 64, callback=check_feedback)
        data, query, data_error, query_error = h.handle(feedback)
        feedback = np.full((query.shape[0], data.shape[1]), .2) if data is not None else None
        iterations += 1
    h.discard()
    thr = N*4 * (96/64) * iterations / duration
    print('With', N*96//64, 'values, throughput:', thr, 'bytes/sec', f'({round(thr/1024/1024*100)/100} MiB/s)', f'({round(iterations/duration*100)/100} it/s)')
for i in range(10):
    asyncio.run(benchmark(64*50 * (i+1)))