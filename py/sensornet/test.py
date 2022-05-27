"""
Tests and benchmarks for this Python implementation of a sensor network.

Not counting other bottlenecks, expect around 10000 steps/second or less, which should be enough for most AI models.

```bash
python3 sensor-network/py/sensornet/test.py
```

Sample run:

```
Tests OK
With 4800 values, throughput: 293637120.0 bytes/sec (280.03 MiB/s) (15293.6 it/s)
With 9600 values, throughput: 555648000.0 bytes/sec (529.91 MiB/s) (14470.0 it/s)
With 14400 values, throughput: 779875200.0 bytes/sec (743.75 MiB/s) (13539.5 it/s)
With 19200 values, throughput: 1009182720.0 bytes/sec (962.43 MiB/s) (13140.4 it/s)
With 24000 values, throughput: 1214438400.0 bytes/sec (1158.18 MiB/s) (12650.4 it/s)
With 28800 values, throughput: 1376202240.0 bytes/sec (1312.45 MiB/s) (11946.2 it/s)
With 33600 values, throughput: 1541366400.0 bytes/sec (1469.96 MiB/s) (11468.5 it/s)
With 38400 values, throughput: 1674055680.0 bytes/sec (1596.5 MiB/s) (10898.8 it/s)
With 43200 values, throughput: 1824076800.0 bytes/sec (1739.58 MiB/s) (10556.0 it/s)
With 48000 values, throughput: 1990272000.0 bytes/sec (1898.07 MiB/s) (10366.0 it/s)
```

To measure [test coverage](http://www.kaner.com/pdfs/pnsqc00.pdf), use [Coverage](https://coverage.readthedocs.io/en/6.3.1/) or an equivalent. Should be 100% or there's a problem.

```bash
coverage run --branch sensor-network/py/sensornet/test.py
coverage report
coverage html
```

# Lessons learned

- You think `import numpy as np` in a module is a good way to use other modules in Python? No, it's a global name. Passing it through local variables everywhere is twice as fast. (However, caching locally-defined functions in default args has no effect on performance.)
"""



import __init__ as sn
import numpy as np
import asyncio
import time



@sn.run
def test0():
    """No shape, so no handling."""
    h = sn.Handler()
    assert h.handle(None, None)[0].shape == (0,0)
    h.data('a', data=[1.])
    h.query('b', query=1, callback=lambda fb: ...)
    h.query('c', query=1)
    assert h.handle(None, None)[0].shape == (0,0)
@sn.run
def test1():
    """Already-named data, and transmission error."""
    h = sn.Handler(8,8,8,8, 64)
    h.data(data = np.zeros((3, 96)), error = np.full((3, 96), -.7))
    data, query, data_error, query_error = h.handle(None, None)
    assert (data == np.zeros((3, 96))).all()
    assert (query == np.zeros((0, 32))).all()
    assert (data_error == np.full((3, 96), -.7)).all()
    assert query_error == None
@sn.run
def test2():
    """Different kinds of names."""
    h = sn.Handler(8,8,8,8, 64)
    h.data(name=('test',), data=np.array([-.4, -.2, .2, .4]))
    h.data(
        name=('test', (-.2, .2, lambda start,end,total: start/total*2-1)),
        data=np.array([-.4, -.2, .2, .4]),
    )
    h.data(name=((1,1,1,1,1,1,1,1),(1,1,1,1,1,1,1,1),(1,1,1,1)), data=np.array([-.4, -.2, .2, .4]))
    h.commit()
    data, query, *_ = h.handle(None, None)
    assert data.shape == (3, 96)
@sn.run
def test3():
    """Named error."""
    h = sn.Handler(8,8,8,8, 64)
    def yes_feedback(fb, *_): assert fb is not None
    h.query(name=('test',), query=16, callback=yes_feedback)
    h.handle(None, None)
    h.handle(np.zeros((1, 96)), None)
@sn.run
def test4():
    """Name's errors."""
    h = sn.Handler(8,8,8,8, 64)
    d = np.array([1.])
    h.data(name=((0,True,1),), data=d)
    try:
        def bad(*_): assert False # pragma: no cover
        h.data(name=(bad,), data=d); assert False
    except TypeError:
        pass # Unwrapped function.
    try:
        h.data(name=(0,), data=d); assert False
    except TypeError:
        pass # Unwrapped number.
    try:
        h.data(name=(['obedience'],), data=d); assert False
    except TypeError:
        pass # Unknown type.
@sn.run
def test5():
    """Sensors are auto-called at each step."""
    h = sn.Handler(8,8,8,8, 64)
    def eh_feedback(fb, *_): assert fb is not None
    h.sensors.append(lambda h: h.query(name=('test',), query=3, callback=eh_feedback))
    h.sensors.append(lambda h: h.query(name=('test',), query=3, callback=eh_feedback))
    assert h.handle(None, None)[1].shape == (2, 32)
    h.handle(np.zeros((2, 96)), None)
@sn.run
def test6():
    """Errors thrown by `callback` are re-thrown."""
    h = sn.Handler(8,8,8,8, 64)
    def err1(*_): raise KeyboardInterrupt()
    def err2(*_): raise TypeError('damn')
    h.query(name='death', query=1, callback=err1)
    assert h.handle(None, None)[1].shape == (1, 32)
    try: h.handle(None, None); assert False
    except KeyboardInterrupt: pass
    h.query(query=np.zeros((5, 32)), error=np.full((5, 32), -.5), callback=err2)
    assert h.handle(None, None)[1].shape == (5, 32)
    try: h.handle(None, None); assert False
    except TypeError: pass
@sn.run
def test7():
    """Non-1D data and feedback."""
    h = sn.Handler(8,8,8,8, 64)
    got = False
    def yes_feedback(fb, *_): nonlocal got;  assert fb.shape == (2,3,4);  got = True
    h.data(name=('test',), data=np.zeros((2,3,4)), error=np.full((2,3,4), -.4))
    h.query(name=('test',), query=(2,3,4), callback=yes_feedback)
    data, query, *_ = h.handle(None, None)
    assert data.shape == (1,96) and query.shape == (1,32)
    h.handle(np.zeros((1,96)), None)
    assert got
@sn.run
def test8():
    """Async operations."""
    sn.shape(8,8,8,8, 64)
    sn.shape(8,8,8,8, 64)
    assert sn.cell_shape == (8,8,8,8, 64)
    assert sn.cell_size == 96
    name = ('test',)
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
            sn.data(name='bees', data=np.zeros((16,)))
            fb = give_feedback_later(*(await sn.handle(fb)))
        await asyncio.sleep(.1)
        await fb # To silence a warning.
        sn.discard()
    asyncio.run(main())
@sn.run
async def test9():
    """Pass-through of (synthetic) handler data to another one."""
    sn.shape(8,8,8,8, 64)
    shape1, shape2 = (13,96), (13,32)
    fut = sn.pipe(np.random.rand(*shape1)*2-1, np.random.rand(*shape2)*2-1, np.zeros(shape1), np.zeros(shape2))
    data, query, *_ = sn.handle(None, None)
    assert data.shape == shape1 and query.shape == shape2
    sn.commit()
    sn.handle(np.zeros(shape1), None)
    assert (await fut).shape == shape1
    sn.discard()
@sn.run
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
@sn.run
def test11():
    """Low-level functions as substitutes for `asyncio.Future`s."""
    h = sn.Handler(8,8,8,8, 64)
    h.handle(None, None)
    n, got = 0, False
    def feedback():
        nonlocal n, got;  n += 1
        if n == 4: got = True
        return got and np.zeros((0, 96))
    h.data(None, np.zeros((2, 96)))
    h.handle(feedback, None)
    h.handle(None, None)
    h.handle(None, None)
    assert not got
    h.handle(None, None)
    assert got
@sn.run
async def test12():
    """Waiting for data to arrive."""
    h = sn.Handler(8,8,8,8, 64)
    async def data_later():
        await asyncio.sleep(.2)
        h.data('hey listen', np.zeros((128,)))
    async def query_later():
        await asyncio.sleep(.2)
        h.query('forces of evil gather', 16)
    asyncio.ensure_future(data_later())
    assert (await h.handle())[0].shape == (2, 96)
    asyncio.ensure_future(query_later())
    assert (await h.handle())[1].shape == (1, 32)
@sn.run
async def test13():
    """`Filter`ing data for specifically-named cells."""
    def good(data, *_):
        good.b = True
        assert (data[:, -64:].flatten()[:3] == np.array([.1, .2, .3])).all()
    def bad(data, *_): # pragma: no cover
        assert False
    h = sn.Handler(8,8,8,8, 64, listeners=[sn.Filter((None, 'this one'), good), sn.Filter(('no'), bad)])
    h.data(name=('mm not this one',), data=np.array([1., 2., 3.]))
    h.data(name=('yes', 'this one'), data=np.array([.1, .2, .3]))
    h.data(name=('this one', 'does not match'), data=np.array([.1, .2, .5]))
    data, query, data_error, query_error = await h.handle()
    assert good.b
    assert sn.Filter((None, 'this one'))(data, cell_shape=h.cell_shape).sum() == 1
    h.listeners[0](data, data_error, h.cell_shape)
print('Tests OK')



async def benchmark(N=64*10):
    """Raw number-shuffling performance."""
    h = sn.Handler(8,8,8,8, 64) # `backend=sn.Torch()` seems slower.
    iterations, feedback = 0, None
    def check_feedback(fb, *_):
        assert fb is not None and fb.shape == (64,) and fb[0] == .2
    async def await_feedback(fut): # pragma: no cover
        check_feedback(await fut)
    randn_src = h.backend.random if not hasattr(h.backend, 'randn') else h.backend
    send_data = randn_src.randn(N)
    start, duration = time.monotonic(), 10.
    name = ('benchmark',)
    while time.monotonic() - start < duration:
        h.data(name, data=send_data)
        # asyncio.ensure_future(await_feedback(h.query(name, 64))) # 15% slowdown.
        h.query(name, 64, callback=check_feedback)
        data, query, data_error, query_error = await h.handle(feedback)
        feedback = h.backend.full((query.shape[0], data.shape[1]), .2) if data is not None else None
        iterations += 1
    h.discard()
    thr = N*4 * (96/64) * iterations / duration
    print('With', N*96//64, 'values, throughput:', thr, 'bytes/sec', f'({round(thr/1024/1024*100)/100} MiB/s)', f'({round(iterations/duration*100)/100} it/s)')
for i in range(10):
    asyncio.run(benchmark(64*50 * (i+1)))