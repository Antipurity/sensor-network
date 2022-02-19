"""
This simple sensor-network environment consists of a few parts:

- TODO: The node-graph to explore.

- TODO:

Reset with `.reset(**options)` (see `.options` for what is allowed), read metrics with `.explored()` and `.collected()`
"""



def reset(**opts):
    options.update(default_options)
    options.update(opts)
    metrics['explored'], metrics['collected'] = 0, 0

    nodes.clear()
    nodes['all'] = []
    _create_nodes(['0'], '0')
    # TODO: And add random connections to all nodes.
    pass
def explored(): return metrics['explored']
def collected(): return metrics['collected']



# In case `sensornet` is not installed as a package.
import sys
sn = sys.modules['sensornet']



options = {}
default_options = {
    # The graph to explore.
    'max_nodes': 1024,
    'node_name_size': 64,
    'child_probabilities': {1:.7, 2:.1, 3:.1, 4:.1},
    'loopback_to_start_probability': .75,
    'random_connection_probability': .1,
    'avg_resource': .1, # The actual initial resource in a node is 0…avg_resource*2.
    # TODO: What other options do we want `reset` to control?
}
metrics = {
    'explored':0,
    'collected':0,
}
nodes = {
    'all': [], # Node IDs, for random sampling.
    '0': [], # Node ID to `[neighbor_ids, name_vec, visited, resource]`.
}
# TODO: How are the agents handled?...



# TODO: A function to create a random node name.
def _create_nodes(ids, start_id): # TODO: Should we also pass in the parent? ...But then, queues are separated, so distribution is uneven...
    """Creates the node tree in `nodes`."""
    while len(ids):
        pass
    # TODO: A function to create a node with a given ID, and a given start ID: `_create_node(id, start_id, new_queue)`.
    #   TODO: ...Wait, but how do we reconcile this func-centric view with a list of unaugmented nodes?... Should we actually pass in the array of node IDs to create?
    #   TODO: How do we generate the level, exactly?
    #     TODO: First create a tree (with a queue of the uncreated frontier) with loopback-connections, then for each node, possibly create random connections.
    #     TODO: How do we test the generation of a level?



reset()
sn.sensors.append(lambda sn: ...) # TODO: WHAT do we append? What does this do? Is this the main world re/creation loop, deploying an agent if the level is empty?
# TODO: On import, push the world-thread to `sn.sensors` (with just world-resets and world-continues actions, to really test exploration, optionally), exposing `.reset(**opts)` to re-init everything, and `.explored()` and `.collected()` to measure performance… No reward (exploration yo) (or maybe, an option to allow forks that do want that reward?); observations: node's remaining resource, node's randomly-initialized vector, and each neighbor's vector, our health (0…1 but exposed as -1…1, increasing with picked-up resource, slowly decreasing (going to a random node), terminating the thread when 0, and no threads means a reset); actions: take-reward (.1), fork (health is split evenly between the threads), suicide (add our health to the cell's), goto-neighbor (vector output, and the nearest-neighbor vector is picked)…
#   ...Maybe don't have health? Or maybe, to have a bit more stuff to explore, do have health...