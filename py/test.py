"""
TODO:

To measure [test coverage](http://www.kaner.com/pdfs/pnsqc00.pdf), use [Coverage](https://coverage.readthedocs.io/en/6.3.1/) or an equivalent.
"""
#   (The interface is not hard to use for the "functions that wait for the handler's decision, where some can spawn new functions and such", right?)
#     (Its flexibility & convenience as an RL interface is kinda amazing. There are just no limitations, at all, there's only writing down ideas.)



import sensornet as sn
import numpy as np
import time



# Benchmark.
#   TODO: Extract into a function, accepting `N`.
sn.shape((8, 24, 64), 8)
iterations, feedback = 0, None
N = 64*10
def check_feedback(fb, *_):
    assert fb is None or fb.shape == (N,) and fb[0] == .2
send_data = np.random.randn(N)
start, duration = time.monotonic(), 10.
while time.monotonic() - start < duration:
    sn.send(('test 1',), data=send_data, on_feedback=check_feedback)
    data, error, no_data, no_feedback = sn.handle(feedback)
    feedback = np.full_like(data, .2)
    iterations += 1
print('bytes/sec', N*4 * iterations / duration) # TODO:

# TODO: Analyse reports of what's not covered by tests, and cover it.