"""
TODO:

To measure test coverage, use https://coverage.readthedocs.io/en/6.3.1/
"""



import sensornet as sn
import numpy as np
import time



# Benchmark.
start, duration = time.monotonic(), 10.
sn.shape((8, 24, 64), 8)
iterations, feedback = 0, None
def check_feedback(fb, *_):
    assert fb is None or fb.shape == (4,) and fb[0] == .2 # TODO: Is it ever non-None?
while time.monotonic() - start < duration:
    sn.send(('test 1',), data=np.array([-.2, .2, .5, -1]), on_feedback=check_feedback)
    data, error, no_data, no_feedback = sn.handle(feedback)
    feedback = -data
    iterations += 1
print('iters/sec', iterations / duration) # TODO: Bytes per second, not iters per sec. And, with data-size-per-step pre-determined.
# TODO: Send random data, respond with negated-data feedback, and measure throughput.
#   TODO: How do we measure time?

# TODO: Analyse reports of what's not covered by tests, and cover it.




# TODO: Also launch tests if this module is executed directly: correctness, then throughput.
#   (The interface is not hard to use for the "functions that wait for the handler's decision, where some can spawn new functions and such", right?)
#     (Its flexibility & convenience as an RL interface is kinda amazing. There are just no limitations, at all, there's only writing down ideas.)
#   ...So what tests do we implement, exactly?
#     At least "generate random data, give random feedback, and check that the feedback is correct"; probably measuring throughput already. From there, could check coverage, and come up with tests that increase that.
#   ...Also, maybe separate tests into a separate file that imports this one?