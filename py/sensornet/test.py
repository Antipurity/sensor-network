"""
Tests and benchmarks for this Python implementation of a sensor network.

Not counting other bottlenecks, expect several thousand steps/second, which should be enough for a lot of AI models (especially real-time-interactive ones).

```bash
python3 sensor-network/py/sensornet/test.py
```

Sample run:

```
Tests OK
With 4800 values, throughput: 478767360.0 bytes/sec (456.59 MiB/s) (24935.8 it/s)
With 9600 values, throughput: 844185600.0 bytes/sec (805.08 MiB/s) (21984.0 it/s)
With 14400 values, throughput: 1178807040.0 bytes/sec (1124.2 MiB/s) (20465.4 it/s)
With 19200 values, throughput: 1447395840.0 bytes/sec (1380.34 MiB/s) (18846.3 it/s)
With 24000 values, throughput: 1666368000.0 bytes/sec (1589.17 MiB/s) (17358.0 it/s)
With 28800 values, throughput: 1859293440.0 bytes/sec (1773.16 MiB/s) (16139.7 it/s)
With 33600 values, throughput: 2068563840.0 bytes/sec (1972.74 MiB/s) (15391.1 it/s)
With 38400 values, throughput: 2240578560.0 bytes/sec (2136.78 MiB/s) (14587.1 it/s)
With 43200 values, throughput: 2390929920.0 bytes/sec (2280.17 MiB/s) (13836.4 it/s)
With 48000 values, throughput: 2521152000.0 bytes/sec (2404.36 MiB/s) (13131.0 it/s)
```

To measure [test coverage](http://www.kaner.com/pdfs/pnsqc00.pdf), use [Coverage](https://coverage.readthedocs.io/en/6.3.1/) or an equivalent. Should be 100% or there's a problem.

```bash
coverage run --branch sensor-network/py/sensornet/test.py
coverage report
coverage html
```

# Lessons learned

- You think `import numpy as np` in a module is a good way to use other modules in Python? No, it's a global name. Passing it through local variables everywhere is twice as fast. (However, caching locally-defined functions in default args has no effect on performance.)

- Constructing `sn.Float` without keyword args in its `__init__` is essentially free, with with `dims=1`, it's slightly noticeable. So `sn.Int(...)` can be freely constructed at-use, whereas `sn.Float(...)` should be pre-constructed.

- `np.prod(shape)` is slower than `p=1;  for i in shape: p*=i`.
"""



import __init__ as sn
import numpy as np
import asyncio
import time



@sn.run
async def test0():
    """No shape, so no handling. (Mostly for test-coverage.)"""
    h = sn.Handler(info={'analog':True, 'choices_per_cell':2**8})
    assert h.handle(None, None)[0].shape == (0,0)
    h.set('a', data=[1.], type=sn.Float(1))
    h.query('b', type=2).close()
    h.query('c', type=(2,)).close()
    h.set('d', 3, 8)
    e = h.query('e', type=sn.Float(1))
    f = h.query('f', type=2)
    h.set(data=np.zeros((2, 0)))
    h.query(type=np.zeros((3, 0))).cancel()
    h.query(type=np.zeros((3, 0)), callback=lambda fb: ...)
    assert h.handle(None, None)[0].shape == (0,0)
    await e
    await f
@sn.run
def test1():
    """Already-named data, and transmission error."""
    h = sn.Handler(8,8,8,8, 64)
    assert h.cell_shape == (8,8,8,8, 64) and h.cell_size == 96
    h.set(data = np.zeros((3, 96)), error = np.full((3, 64), .3))
    data, query, error = h.handle(None, None)
    assert (data == np.zeros((3, 96))).all()
    assert (query == np.zeros((0, 32))).all()
    assert (error == np.full((3, 64), .3)).all()
@sn.run
def test2():
    """Different kinds of names."""
    h = sn.Handler(8,8,8,8, 64)
    h.set(name=('test',), data=np.array([-.4, -.2, .2, .4]), type=sn.Float(4))
    h.set(name=((1,1,1,1,1,1,1,1),(1,1,1,1,1,1,1,1),(1,1,1,1)), data=np.array([-.4, -.2, .2, .4]), type=sn.Float(4))
    h.commit()
    data, query, error = h.handle(None, None)
    assert data.shape == (2, 96)
@sn.run
async def test3():
    """Named error."""
    h = sn.Handler(8,8,8,8, 64, info={'analog':True, 'choices_per_cell':2**16})
    test = h.query(name='test', type=sn.Float(17))
    h.query(type=np.zeros((1, 32)), callback=lambda fb: ...)
    await h.handle(None)
    h.set('burn', 9, 10)
    data, query, error = await h.handle(np.zeros((2, 96)))
    assert data.shape == (1,96) and h.Int.decode_bits(h.backend, data[:, 32 : 32+16])[0] == 9000
    assert (await test) is not None
@sn.run
def test4():
    """Name's errors."""
    h = sn.Handler(8,8,8,8, 64)
    d = np.array([1.])
    h.set(name=((0,True,1),), data=d, type=sn.Float(1))
    try:
        h.set(name=(0,), data=d, type=sn.Float(1)); assert False
    except TypeError:
        pass # Unwrapped number.
    try:
        h.set(name=(['obedience'],), data=d, type=sn.Float(1)); assert False
    except TypeError:
        pass # Unknown type.
@sn.run
async def test5():
    """Sensors are auto-called at each step."""
    h = sn.Handler(8,8,8,8, 64)
    fbs = []
    def sensor(h):
        async def asensor(fb):
            assert (await fb) is not None
        fbs.append(sn.run(asensor, h.query(name='insolence', type=sn.Float(3))))
    h.sensors.add(lambda h: sensor(h))
    h.sensors.add(lambda h: sensor(h))
    assert h.handle(None, None)[1].shape == (2, 32)
    h.handle(np.zeros((2, 96)), None)
    assert len(fbs) == 4
@sn.run
async def test6():
    """Non-1D data and feedback."""
    h = sn.Handler(8,8,8,8, 64)
    h.set(name=('test',), data=np.random.rand(2,3,4), type=sn.Float(2,3,4), error=.6)
    test = h.query(name=('test',), type=sn.Float(2,3,4))
    data, query, error = h.handle(None, None)
    assert data.shape == (6,96) and query.shape == (6,32) # 2*3 cells of 4 values and 60 zeros.
    h.handle(np.zeros((6,96)), None)
    assert (await test).shape == (2,3,4)
@sn.run
def test7():
    """Async operations."""
    sn.cell_shape = (8,8,8,8, 64)
    sn.set('event') # This updates `sn.cell_size`.
    assert sn.cell_shape == (8,8,8,8, 64)
    assert sn.cell_size == 96
    name = ('test',)
    n = 0
    finished = 0
    async def request_data(h, maybe=False):
        nonlocal finished
        fb = await (h.query(name, sn.Float(3,5)) if maybe else h.get(name, sn.Float(3,5)))
        finished += 1
        if not maybe: assert fb.shape == (3,5)
    async def give_feedback_later(data, query, error):
        nonlocal n
        await asyncio.sleep(.1)
        n += 1
        return np.zeros((query.shape[0], data.shape[1])) if n==30 or n>60 else None
    @sn.run
    async def main():
        for _ in range(5):
            sn.run(request_data, sn)
        sn.run(request_data, sn, True)
        fb = None
        while finished < 6:
            sn.set(name='bees', data=np.zeros((16,)), type=sn.Float(16))
            fb = give_feedback_later(*(await sn.handle(fb)))
        await asyncio.sleep(.1)
        await fb # To silence a warning.
        sn.discard()
@sn.run
async def test8():
    """Pass-through of (synthetic) handler data to another one."""
    sn.cell_shape = (8,8,8,8, 64)
    shape1, shape2, shape3 = (13,96), (13,32), (13,64)
    fut = sn.pipe(np.random.rand(*shape1)*2-1, np.random.rand(*shape2)*2-1, np.zeros(shape3))
    data, query, error = sn.handle(None, None)
    assert data.shape == shape1 and query.shape == shape2
    sn.commit()
    sn.handle(np.zeros(shape1), None)
    assert (await fut).shape == shape1
    sn.discard()
@sn.run
def test9():
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
def test10():
    """Low-level functions as substitutes for `asyncio.Future`s."""
    h = sn.Handler(8,8,8,8, 64)
    h.handle(None, None)
    n, got = 0, False
    def feedback():
        nonlocal n, got;  n += 1
        if n == 4: got = True
        return got and np.zeros((0, 96))
    h.set(data=np.zeros((2, 96)), error=.05)
    h.handle(feedback, None)
    h.handle(None, None)
    h.handle(None, None)
    assert not got
    h.handle(None, None)
    assert got
@sn.run
async def test11():
    """Waiting for data to arrive."""
    h = sn.Handler(8,8,8,8, 64)
    async def data_later():
        await asyncio.sleep(.2)
        h.set('hey listen', np.zeros(128), sn.Float(2, 128))
    async def query_later():
        await asyncio.sleep(.2)
        assert (await h.query('forces of evil gather', sn.Float(16))) is None
    sn.run(data_later)
    assert (await h.handle())[0].shape == (2, 96)
    sn.run(query_later)
    assert (await h.handle())[1].shape == (1, 32)
@sn.run
async def test12():
    """`Filter`ing data, for remote debugging, AKA spying on good people. [REDACTED]"""
@sn.run
async def test13():
    """Modifying names."""
    h = sn.Handler(8,8,8,8, 64)
    fork = h.fork(lambda name: ('z',))
    fork.set(name='btgrnonets', data=[1., 2.], type=sn.Float(2))
    data, query, error = await h.handle()
    template = sn._name_template(np, lambda p: sn._str_to_floats(np, p), h.cell_shape, (None, 'z'))
    assert ((template != template) | (data[0, :template.shape[-1]] == template)).all()
@sn.run
def test14():
    """Errors thrown by `callback` are re-thrown."""
    h = sn.Handler(8,8,8,8, 64)
    def err1(*_): raise KeyboardInterrupt()
    def err2(*_): raise TypeError('damn')
    h.query(type=np.zeros((1, 32)), callback=err1)
    assert h.handle(None, None)[1].shape == (1, 32)
    try: h.handle(None, None); assert False
    except KeyboardInterrupt: pass
    h.query(type=np.zeros((5, 32)), callback=err2)
    assert h.handle(None, None)[1].shape == (5, 32)
    try: h.handle(None, None); assert False
    except TypeError: pass
@sn.run
async def test15():
    """Bad names."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        h = sn.Handler(8,8,8,8, 64, info={'choices_per_cell':2**2})
        h.set(('a', 'b', 'c', 'd'), 0, 3)
        h.set(('a', 'b', (0,1,2,3,4,5,6,7,8)), 1, 3)
        def fn(): ... # pragma: no cover
        try: h.set(('x', fn), 2, 3); assert False
        except TypeError: pass
@sn.run
async def test16():
    """Integer queries."""
    h = sn.Handler(8,8,8,8, 64, info={'choices_per_cell':2**4})
    async def do(expect):
        i = h.query('assemble that 16-bit number please 1', 65536)
        fb = None
        data, query, error = await h.handle(fb)
        assert data.shape == (0, 96) and query.shape == (4, 32) and error is None
        fb = np.ones((query.shape[0], 96)) if expect is not None else None
        h.handle(fb, None) # (`await h.handle(fb)` here infinitely awaits extra data, which is fine if in a detached execution thread, but not here.)
        assert (await i) == expect
    await do(65535)
    await do(None)
    i = h.query('itty 2-bitty num', 4)
    await h.handle(None)
    h.handle(np.ones((1, 96)), None)
    assert (await i) == 3
    i = h.query('atty-quadratty num', 16)
    await h.handle(None)
    h.handle(np.ones((1, 96)), None)
    assert (await i) == 15
@sn.run
async def test17():
    """Lists and goals."""
    h = sn.Handler(8,8,8,8, 64, info={'choices_per_cell':2**4})
    h.set('miew', [1, None], sn.List(16, 15))
    h.set('fmemor', [1, 6], sn.Goal([16, 16]))
    L1 = h.get('hgtngege', [16, 14])
    L2 = h.get('nveerny', sn.Goal([16, 16]))
    L3 = h.query('nveerny', sn.Goal(sn.List(16, 16)))
    data, query, error = await h.handle(None)
    assert data.shape == (3, 96) and query.shape == (6, 32) and error is None
    h.handle(np.ones((6, 96)), None)
    assert (await L1) == [15, 1] # 15%14 == 1
    assert ((await L2) == [15, 15]).all()
    assert (await L3) == [15, 15]
    assert repr(sn.Goal(sn.List(sn.Int(16), sn.Float(15)))) == 'sn.Goal(sn.List(sn.Int(16),sn.Float(15)))'
@sn.run
async def test18():
    """Events."""
    h = sn.Handler(8,8,8,8, 64, info={'analog':True, 'choices_per_cell':2**4})
    h.set('important event occured')
    h.set('and another one', type=h.Event())
    h.set('and another one', .5, sn.IntFloat())
    h.set('and another one', [.5], sn.IntFloat(1, mu=256, domain=(-5,5)))
    L1 = h.query('survive', sn.IntFloat(mu=256, domain=(-5, 5)))
    L2 = h.query('or get your spine used to make furniture', sn.IntFloat())
    assert (await h.handle(None))[0].shape == (4, 96)
    h.handle(np.ones((2, 96)), None)
    assert (await L1) == 5.
    assert (await L2) == 1.
    repr(h.Event()), repr(h.IntFloat())
@sn.run
async def test19():
    """2D images."""
    h = sn.Handler(8,8,8,8, 64, info={'analog':True})
    h.set('in the end, only silence was left', np.random.rand(3, 256, 256), sn.Float(4, 512, 512, dims=2))
    img = h.query('not even images existed anymore', sn.Float(16, 16, dims=2))
    assert (await h.handle())[0].shape == (3*(256/8)**2, 96)
    h.set('no event happened, not even this one')
    fb = np.random.rand(4, 96)
    assert (await h.handle(fb))[0].shape == (1, 96)
    assert (await img).shape == (16, 16)
@sn.run
async def test20():
    """Dicts."""
    h = sn.Handler(8,8,8,8, 64, info={'analog':True})
    T = {'a': sn.Float(10), 'b': sn.Float(10), 'c': sn.Float(10)}
    h.set('…', {'a': np.random.rand(5), 'b': None}, T)
    L1 = h.query('…', T)
    L2 = h.get('…', T)
    W = h.submit()
    assert (await h.handle())[0].shape == (1, 96)
    await W
    h.set('…')
    assert (await h.handle(np.random.rand(6, 96)))[0].shape == (1, 96)
    assert isinstance(await L1, dict)
    assert isinstance(await L2, dict)
    assert repr(sn.Dict(a=5, b=6)) == 'sn.Dict(a=sn.Int(5),b=sn.Int(6))'
print('Tests OK')



async def benchmark(N=64*10):
    """`Float` number-shuffling performance."""
    h = sn.Handler(8,8,8,8, 64, info={'analog':True, 'choices_per_cell':2**8})
    iterations, feedback = 0, None
    randn_src = h.backend.random if not hasattr(h.backend, 'randn') else h.backend
    send_data = randn_src.randn(N)
    start, duration = time.monotonic(), 10.
    name = ('benchmark',)
    f32 = h.backend.float32
    set_type, query_type = sn.Float(N), sn.Float(256)
    while time.monotonic() - start < duration:
        h.set(name, data=send_data, type=set_type)
        h.query(name, query_type).close()
        data, query, error = await h.handle(feedback)
        feedback = h.backend.full((query.shape[0], data.shape[1]), .2, dtype=f32) if data is not None else None
        iterations += 1
    h.discard()
    thr = N*4 * (96/64) * iterations / duration
    print('With', N*96//64, 'values, throughput:', thr, 'bytes/sec', f'({round(thr/1024/1024*100)/100} MiB/s)', f'({round(iterations/duration*100)/100} it/s)')
for i in range(10):
    asyncio.run(benchmark(64*50 * (i+1)))