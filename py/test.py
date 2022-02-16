"""
TODO:
"""



import sensornet as sn
import numpy as np



sn.shape((8, 24, 64), 8)
sn.send(('test 1',), data=np.array([-.2, .2, .5, -1]))
# TODO: ...Fix bugs.
print(sn.handle())
# TODO: Send random data, respond with random feedback, and measure throughput.

# TODO: Analyse reports of what's not covered by test, and cover it.




# TODO: Also launch tests if this module is executed directly: correctness, then throughput.
#   (The interface is not hard to use for the "functions that wait for the handler's decision, where some can spawn new functions and such", right?)
#     (Its flexibility & convenience as an RL interface is kinda amazing. There are just no limitations, at all, there's only writing down ideas.)
#   ...So what tests do we implement, exactly?
#     At least "generate random data, give random feedback, and check that the feedback is correct"; probably measuring throughput already. From there, could check coverage, and come up with tests that increase that.
#   ...Also, maybe separate tests into a separate file that imports this one?