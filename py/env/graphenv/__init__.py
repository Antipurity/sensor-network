"""
This simple sensor-network environment consists of a few parts:

- The node-graph to explore.
    - Each node has a 'name' vector for directing the traversal, and the node's resource amount (never less than 0).
    - Each node has directed edges to a few neighbors.
        - Nodes mostly form a tree, sometimes with random edges, to increase path length.
        - Nodes usually link back to the starting node, to make sabotaging exploration likely.

- Agents that explore nodes.
    - One agent starts at the starting node.
    - Each agent knows its current position (a node), the agent's resource amount, and whether consuming resources gives it reward.
    - Each agent loses some of its resource each step, which is given to some random node. Total resources are always preserved.
    - Observations:
        - Agent resource & hunger-ness.
        - Node resource.
        - Node 'name'.
        - All neighbor 'names'.
        - (`sensornet`'s cell shape should reserve 3 name-parts.)
    - Actions:
        - Take resources from the node for ourselves.
        - Fork. Possibly into an agent that hungers.
        - Go to a neighbor. 'Which neighbor' is the nearest-neighbor of the received data in node-neighbor 'names'.
        - Un-fork. Kick the bucket. Bite the dust. If all agents meet their maker, the world resets.

- The world can end at each step if chosen, just to make exploration more difficult.

Reset with `.reset(**options)` (see `.options` for what can be changed, which is quite a lot), read the desired metric with `.explored()` (0…1). Don't overfit to this metric, this is *exploration*, not *reward*.
"""



def reset(**opts):
    """Destroys and recreates the world."""
    options.update(default_options)
    options.update(opts)
    metrics['nodes'], metrics['explored'], metrics['collected'] = 0, 0, 0

    nodes.clear()
    for name in agents.keys():
        agents[name][0].cancel()
    agents.clear()
    if not options['stop']:
        nodes['start'] = _random_name()
        _create_nodes(nodes['start'])
        options['please_make_an_agent'] = True
    metrics['nodes'] = len(nodes['all'])
def explored(): return metrics['explored'] / metrics['nodes']
def reachable(): return _reachable(*[agents[a][1] for a in agents.keys()])



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
    'max_nodes': 1000,
    'node_name_size': 16,
    'child_probabilities': {1:.7, 2:.1, 3:.1, 4:.1},
    'loopback_to_start_probability': .75,
    'loopback_to_parent_probability': .25,
    'random_connection_probability': .03,
    'avg_resource': .1, # The actual initial resource in a node is 0…avg_resource*2.
    # The top-level-action options.
    'can_reset_the_world': True, # If checked, each step has a 50% chance of resetting the world, which only adds to exploration.
    # Agents that act on the graph.
    'max_agents': 16, # 1 to disallow forking.
    'allow_suicide': True, # Look at all this proof-of-exploration opportunity.
    'allow_fork_with_resource_goal': True, # Exploration that can choose to maximize.
    'step_takes_resources': .01, # The resource is moved to a random node, so total resources are conserved.
    'resource_consumption_speed': .1,
    # Verbose output.
    'debug': False,
}
metrics = {
    'nodes':0,
    'explored':0,
    'collected':0,
}
nodes = {
    'all': [], # Node IDs, for random sampling.
    'start': '',
    # Node ID to `[neighbor_ids, name_vec, resource, visited]`.
}
agents = {
    # Agent ID (randomly-generated) to `[task, at, resource, hunger]`:
    #   `.cancel()`able task, node ID, 0…1 resource, bool of whether resource-consumption is reward.
}



import random
import asyncio
import numpy as np



def agent(sn, at=..., resource=1., hunger=False):
    """Creates an agent, two-way-bound to wander the graph."""
    if at is ...: at = nodes['start']
    name = _random_name()
    async def loop():
        reward = 0.
        while name in agents:
            _, at, resource, hunger = agents[name]
            neighbors, at_name_vec, at_resource, at_visited = nodes[at]
            # Keep track of exploration.
            if not at_visited:
                nodes[at][3] = True
                metrics['explored'] += 1
            # Send observations.
            sn.data(name=(name, at, 'agent resource'), data=np.array([resource*2-1, 1. if hunger else -1.]), reward=reward)
            sn.data(name=(at, 'node resource'), data=np.array([at_resource*2-1]), reward=reward)
            sn.data(name=(at, 'name_vec'), data=at_name_vec, reward=reward)
            for i, ng in enumerate(neighbors):
                sn.data(name=(at, ng, 'neighbor '+str(i)), data=nodes[ng][1], reward=reward)
            reward = 0.
            # Receive actions.
            actions = 4
            act = await sn.query(name=(name, at, 'act'), query = options['node_name_size'] + actions)
            if act is None: continue # Re-send observations on dropped packets.
            data, acts = act[:-actions], act[-actions:]
            # Bleed onto a random node. Die if unfortunate.
            dresource = min(options['step_takes_resources'], resource)
            resource -= dresource
            nodes[random.choice(nodes['all'])][2] += dresource
            at_resource = nodes[at][2]
            if options['debug']: print('agent tick', name, 'at', at[:8], 'with', len(neighbors), 'neighbors')
            if resource <= 0:
                if options['debug']: print('agent DIED', name)
                del agents[name]
                break
            agents[name][2] = resource
            # Take resources from the cell.
            if acts[0]>0 and at_resource > 0.:
                dresource = min(options['resource_consumption_speed'], at_resource, 1. - resource)
                resource += dresource;  agents[name][2] = resource
                at_resource -= dresource;  nodes[at][2] = at_resource
                if hunger: reward = dresource
                if options['debug']: print('    agent drew resources at', at[:8])
            # Fork. (Casually, as if you see action-space-alteration in RL often.)
            if acts[1]>0 and len(agents.keys()) < options['max_agents']:
                ours = .5
                agent(sn, at, resource*(1-ours), options['allow_fork_with_resource_goal'] and data[0]>0)
                resource *= ours;  agents[name][2] = resource
                if options['debug']: print('    agent forked at', at[:8])
            # Goto neighbor.
            if acts[2]>0 and len(neighbors):
                diffs = data - np.stack([nodes[ng][1] for ng in neighbors])
                nearest_neighbor_i = np.argmin(np.sum(np.abs(diffs), -1))
                prev_at = at
                at = neighbors[nearest_neighbor_i]
                agents[name][1] = at
                if options['debug']: print('    agent goes to neighbor', prev_at[:8], '→', at[:8], '('+str(nearest_neighbor_i)+'/'+str(len(neighbors))+')')
            # Un-fork.
            if acts[3]>0 and options['allow_suicide']:
                if options['debug']: print('    agent UNFORKED')
                nodes[at][2] += resource
                del agents[name]
                break
        if not agents:
            reset(**options)
            agent(sn)
    agents[name] = [asyncio.ensure_future(loop()), at, resource, hunger]
    return name



def _random_name():
    return ''.join([random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ') for _ in range(16)])
def _create_nodes(start_id):
    """Creates the node tree-like graph with loopback connections (which decrease the probability of successful exploration) in `nodes`."""
    ids = [start_id]
    nodes['all'] = []
    prob = random.random
    def new_node(parent_id, id = None):
        if id is None: id = _random_name()
        if parent_id is not None: nodes[parent_id][0].append(id) # Parents know about children.
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
            random.uniform(0, options['avg_resource']),
            False,
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
        for _ in range(children):
            if len(nodes['all']) < options['max_nodes']: new_node(id)
    # Note: we're cutting off at `max_nodes`, so there's a chance for some nodes to have no neighbors and be inescapable black holes.
    #   It's a feature, since this gives a reason for forking to exist.
def _reachable(*node_names):
    stack, visited = list(node_names), set(node_names)
    while len(stack):
        name = stack.pop()
        neighbors = nodes[name][0]
        for ng in neighbors:
            if ng not in visited:
                stack.append(ng)
                visited.add(ng)
    return len(visited) / len(nodes['all'])



def _maybe_reset_the_world(fb):
    if fb is not None and (fb[0] > 0).all():
        if options['debug']: print('world resets', fb is not None and fb[0])
        reset(**options)
    else:
        if options['debug']: print('world continues', fb is not None and fb[0], 'agents', list(agents.keys()))
def _top_level_actions(sn):
    if options['can_reset_the_world']:
        sn.query(name=('world', 'reset'), query=sn.cell_shape[-1], callback=_maybe_reset_the_world)
    if not options['stop'] and not agents and options['please_make_an_agent']:
        agent(sn)
        options['please_make_an_agent'] = False



# If you really need many environments, launch many processes, and enjoy the benefits of actual parallelism.
reset()
sn.sensors.append(_top_level_actions)