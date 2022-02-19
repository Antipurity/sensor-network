"""
This simple sensor-network environment consists of a few parts:

- TODO: The node-graph to explore.

- TODO: Agents that explore.

- The world possibly ending at each step, just to make exploration more difficult.

Reset with `.reset(**options)` (see `.options` for what is allowed), read the main metric with `.explored()` (0…1). Don't overfit to this metric.
"""



def reset(**opts):
    options.update(default_options)
    options.update(opts)
    metrics['nodes'], metrics['explored'], metrics['collected'] = 0, 0, 0

    nodes.clear()
    for name in agents.keys():
        agents[name][0].cancel()
        del agents[name]
    if not opts['stop']:
        _create_nodes('start')
        options['please_make_an_agent'] = True
    metrics['nodes'] = len(nodes['all'])
def explored(): return metrics['explored'] / metrics['nodes']



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
    'node_name_size': 32,
    'child_probabilities': {1:.7, 2:.1, 3:.1, 4:.1},
    'loopback_to_start_probability': .75,
    'loopback_to_parent_probability': .25,
    'random_connection_probability': .1,
    'avg_resource': .1, # The actual initial resource in a node is 0…avg_resource*2.
    # The top-level-action options.
    'can_reset_the_world': True, # If checked, each step has a 50% chance of resetting the world, which only adds to exploration.
    # Agents that act on the graph.
    'max_agents': 16, # 1 to disallow forking.
    'allow_suicide': True, # Look at all this proof-of-exploration opportunity.
    'allow_fork_with_resource_goal': True,
    'step_takes_resource': .01, # The resource is moved to a random node, so total resources are conserved.
    'resource_consumption_speed': .1,
}
metrics = {
    'nodes':0,
    'explored':0,
    'collected':0,
}
nodes = {
    'all': [], # Node IDs, for random sampling.
    'start': [], # Node ID to `[neighbor_ids, name_vec, visited, resource]`.
}
agents = {
    # Agent ID (randomly-generated) to `[task, at, resource, hunger]`:
    #   `.cancel()`able task, node ID, 0…1 resource, bool of whether resource-consumption is reward.
}



import random
import asyncio
import numpy as np



def agent(sn, at='start', resource=1., hunger=False):
    """Creates an agent, two-way-bound to wander the graph."""
    name = _random_name()
    async def loop():
        reward = 0.
        while True:
            _, at, resource, hunger = agents[name]
            # nodes[at] = [neighbor_ids, name_vec, visited, resource]
            # TODO: Mark the node as explored, and increment metrics['explored'] if it previously was not.
            # TODO: Bleed our resource. If <0, suicide and return.
            # TODO: Send node's and agent's data.
            #   node's remaining resource,
            #   node's own randomly-initialized vector,
            #   each neighbor's vector (the name includes the neighbor ID and the node ID),
            #   our health (0…1 but exposed as -1…1, increasing with picked-up resource, slowly decreasing (going to a random node), terminating the thread when 0, and no threads means a reset),
            #   all sent with the `reward`, which is then reset to `0.`.
            # TODO: Determine which of the 4 actions we can do: take-reward if nodes[at][3]>0, fork if len(agents.keys())<options['max_agents'], suicide if options['allow_suicide'], goto-neighbor if len(nodes[at][0]).
            # TODO: If no actions are OK, continue. (Won't ever get non-forking actions, but at least we'll eat up resources and annoy the handler.)
            # TODO: Get action's data.
            #   Of length options['node_name_size'] + actions, always.
            # TODO: Execute the action, with the max number in it.
            #   take-reward (can't take the node to below 0, can't take us to above 1) (if `hunger`, also sets `reward`),
            #   fork (unless will have too many agents) (health is split evenly between the threads) (possibly setting the `hunger` flag, if the extra data calls for it),
            #   suicide (add our resource to the cell's),
            #   goto-neighbor (vector output, and the nearest-neighbor vector among neighbors is picked)
    agents[name] = [asyncio.ensure_future(loop()), at, resource, hunger]



def _random_name():
    return ''.join([random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ') for _ in range(16)])
def _create_nodes(start_id):
    """Creates the node tree-like graph with loopback connections (which decrease the probability of successful exploration) in `nodes`."""
    ids = [start_id]
    nodes['all'] = []
    prob = random.random
    def new_node(parent_id, id = None):
        if id is None: id = _random_name()
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
        if not len(agents['all']):
            if options['please_make_an_agent']:
                agent(sn)
                options['please_make_an_agent'] = False
            else: # Everyone. Dead.
                reset()



# If you really need many environments, launch many processes, and enjoy the benefits of actual parallelism.
reset()
sn.sensors.append(_top_level_actions)