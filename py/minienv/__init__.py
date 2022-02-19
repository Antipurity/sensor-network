"""
TODO:
"""



# In case `sensornet` is not installed as a package.
import sys
sn = sys.modules['sensornet']



# TODO: On import, push the world-thread to `sn.sensors` (with just world-resets and world-continues actions, to really test exploration, optionally), exposing `.reset(**opts)` to re-init everything, and `.explored()` and `.collected()` to measure performance… No reward (exploration yo); observations: node's remaining resource, node's randomly-initialized vector, and each neighbor's vector, our health (0…1 but exposed as -1…1, increasing with picked-up resource, slowly decreasing, terminating the thread when 0, and no threads means a reset); actions: take-reward (.1), fork (health is split evenly between the threads), suicide (add our health to the cell's), goto-neighbor (vector output, and the nearest-neighbor vector is picked)…

# TODO: 1000 nodes, each connected to 2|3 random ones, and a few containing some reward that can be taken with an action, which transfers a portion of the reward to a successor node.