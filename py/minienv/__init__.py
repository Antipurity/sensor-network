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
    if not opts['stop']:
        _create_nodes('start')
def explored(): return metrics['explored']
def collected(): return metrics['collected']



# In case `sensornet` is not installed as a package.
import sys
try:
    sn = sys.modules['sensornet']
except KeyError:
    raise ImportError('Please `import sensornet` first') from None



options = {}
default_options = {
    # Kill-switch.
    'stop': False,
    # The graph to explore.
    'max_nodes': 1024,
    'node_name_size': 64,
    'child_probabilities': {1:.7, 2:.1, 3:.1, 4:.1},
    'loopback_to_start_probability': .75,
    'loopback_to_parent_probability': .25,
    'random_connection_probability': .1,
    'avg_resource': .1, # The actual initial resource in a node is 0…avg_resource*2.
    # The top-level-action options.
    'can_reset_the_world': True, # If checked, each step has a 50% chance of resetting the world, which only adds to exploration.
    # Agents that act on the graph.
    # TODO: What other options do we want `reset` to control?
}
metrics = {
    'explored':0,
    'collected':0,
}
nodes = {
    'all': [], # Node IDs, for random sampling.
    'start': [], # Node ID to `[neighbor_ids, name_vec, visited, resource]`.
}
# TODO: How are the agents handled?...
#   Do we want them to be contained in functions, and have an array (or set) of per-agent Tasks which we can `.cancel()` when resetting?
#   TODO: No reward (exploration yo) (or maybe, an option to allow forks that do want that reward?); observations: node's remaining resource, node's randomly-initialized vector, and each neighbor's vector, our health (0…1 but exposed as -1…1, increasing with picked-up resource, slowly decreasing (going to a random node), terminating the thread when 0, and no threads means a reset); actions: take-reward (.1), fork (health is split evenly between the threads), suicide (add our health to the cell's), goto-neighbor (vector output, and the nearest-neighbor vector is picked)…



import random
import numpy as np



def _random_node_name():
    return ''.join([random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ') for _ in range(16)])
def _create_nodes(start_id):
    """Creates the node tree-like graph with loopback connections (which decrease the probability of successful exploration) in `nodes`."""
    ids = [start_id]
    nodes['all'] = []
    prob = random.random
    def new_node(parent_id, id = None):
        if id is None: id = _random_node_name()
        neighbors = []
        if prob() < options['loopback_to_start_probability']: neighbors.append(start_id)
        if prob() < options['loopback_to_parent_probability'] and parent_id is not None:
            neighbors.append(parent_id)
        if prob() < options['random_connection_probability']*.5 and len(nodes['all']):
            neighbors.append(random.choice(nodes['all']))
        if prob() < options['random_connection_probability']*.5 and len(nodes['all']):
            nodes[random.choice(nodes['all'])][0].append(id)
        nodes[id] = [
            neighbors,
            np.random.rand(options['node_name_size']),
            False,
            random.uniform(0, options['avg_resource']),
        ]
        nodes['all'].append(id)
        ids.append(id)
    new_node(None, start_id)
    children_prob_sum = sum(options['child_probabilities'].values())
    while len(ids) and len(nodes['all']) < options['max_nodes']:
        # Take a random id from `ids`, then create children.
        i = random.randrange(len(ids))
        ids[i], ids[-1] = ids[-1], ids[i]
        id, children = ids.pop(), 0
        # Pick the child-count, and create children.
        p = prob() * children_prob_sum
        for ch in options['child_probabilities'].keys():
            p -= options['child_probabilities'][ch]
            if p < 0:
                children = ch
                break
        for _ in range(children): new_node(id)
    # Note: we're cutting off at `max_nodes`, so there's a chance for some nodes to have no neighbors and be inescapable black holes.
    #   It's a feature, since this gives a reason for forking to exist.



def _maybe_reset_the_world(fb, *_):
    if fb is not None and (fb[0] > 0).all():
        reset()
def _top_level_actions(sn):
    if options['can_reset_the_world']:
        sn.send(name=('world', 'reset'), data=1, on_feedback=_maybe_reset_the_world)
    if not options['stop']:
        pass # TODO: Also deploy an agent if there are none.



# If you really need many environments, launch many processes, and enjoy the benefits of actual parallelism.
reset()
sn.sensors.append(_top_level_actions)