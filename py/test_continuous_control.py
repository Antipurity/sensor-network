"""
Can RL be done not via approximating the loss (discounted sum of future distances here, AKA return), but via BPTT? Mathematically, sure, but what are the obstacles to doing it practically?

This 2D env has 1 spaceship and a repulsor in the middle; the actions control acceleration. All smooth, so grad-min should be able to find best actions. Effective exploration would be able to cover a lot of the torus.

---

We implement an RNN that minimizes the distance between goal-conditioned paths and goals, sampling goals from the recent past, predicting the next-frame, and minimizing its distance to the goal.
"""



import torch
import torch.nn as nn
device = 'cuda' if torch.cuda.is_available() else 'cpu'



from model.log import log, clear, finish

import random



def env_init(batch_size): # → state, hidden_state
    """Generate initial positions & velocities of the agent."""
    return torch.rand(batch_size, 2, device=device), torch.zeros(batch_size, 2, device=device)
def env_step(posit, veloc, accel): # → state, hidden_state
    """`env_step(state, hidden_state, action)`

    Add acceleration to velocity, and add velocity to position.

    Max action-acceleration is `1e-3`.

    There's a repulsor at `(.5, .5)` that adds a force proportional to `3e-5 / (pos - (.5, .5)).square()` to acceleration, to make exploration not as trivial. There's friction that multiplies velocity (hidden state) by `0.99` each step.

    If you must know how it feels, use this simple HTML page:

    ```html
    <div id="target" style="position:absolute; background-color: red; width:5px; height:5px"></div>
    <script>
    target = document.getElementById('target')
    let mousex=0, mousey=0
    onmousemove = evt => { mousex = evt.clientX / innerWidth, mousey = evt.clientY / innerHeight }

    let x=0, y=0, dx=0, dy=0
    setInterval(() => {
    let mx = mousex-x, my=mousey-y, m = Math.hypot(mx, my)
    if (m) mx /= m/1e-3, my /= m/1e-3

    let flen = ((x-.5)**2 + (y-.5)**2 + 1e-5), f = 3e-5 / flen
    let fx = f * (x-.5)/flen, fy = f * (y-.5)/flen

    let ddx = mx+fx, ddy = my+fy
    dx+=ddx, dy+=ddy, x+=dx, y+=dy
    dx*=.99, dy*=.99, x=(x%1+1)%1, y=(y%1+1)%1
    target.style.left = (x * innerWidth) + 'px'
    target.style.top = (y * innerHeight) + 'px'
    }, 50) // 20 FPS
    </script>
    ```"""
    accel = accel.detach()[..., :2]
    accel = accel * 1e-3 / 2
    accel = accel / (accel.square().sum(-1, keepdim=True).sqrt().max(torch.tensor(1., device=device)))
    force_center = torch.ones(posit.shape[0], 2, device=device)/2
    force_len = (posit - force_center).square() + 1e-5
    force = 3e-5 / force_len
    accel = accel + force * (posit - force_center) / force_len
    veloc = (veloc + accel) * .99
    posit = torch.remainder(posit + veloc, 1.)
    return posit, veloc



class ReplaySample:
    __slots__ = ('state', 'action', 'as_goal')
    def __init__(self, state, action, as_goal):
        self.state, self.action, self.as_goal = state, action, as_goal
class ReplayBuffer:
    """Stores the in-order sequence of most-recent events. Needs `max_len=1024`. Supports `len(rb)`, `rb.append(data)`, `rb[index]`, `rb.sample_best()`."""
    def __init__(self, max_len=1024):
        self.head, self.max_len = 0, max_len
        self.buffer = []
    def __len__(self):
        """How many data-samples are currently contained. Use this to decide the indices to sample at."""
        return len(self.buffer)
    def append(self, data):
        """Appends a data-sample to the replay buffer.

        Save unrolls."""
        if len(self.buffer) == self.max_len:
            self.buffer[self.head] = data
        else:
            self.buffer.append(data)
        self.head = (self.head + 1) % self.max_len
    def __getitem__(self, index):
        """Returns the sample at the given index. For example, use `random.randrange(len(replay_buffer))`.

        Consecutive indices are guaranteed to be a part of a contiguous sequence.

        Remember unrolls, either with independent samples, consecutive pairs, or ordered faraway sequences."""
        assert isinstance(index, int)
        return self.buffer[(self.head + index) % len(self)]
    def __iter__(self):
        for i in range(len(self)): yield self[i]
    def sample_best(self, samples=16, combine=lambda a,b: a.combine(b)):
        """A primitive algorithm for sampling likely-best data.

        For picking at-unroll-time goals.

        Args:
        - `samples=16`: how many samples to choose among. We don't have a precomputed data structure.
        - `combine = lambda a,b: a.combine(b)`: the actual chooser."""
        N, result = len(self), None
        for _ in range(samples):
            sample = self[random.randrange(N)]
            result = sample if result is None else combine(result, sample)
            # (This creates an O(N) chain of computations, though O(log(N)) is possible.)
        return result



class SkipConnection(nn.Module):
    """Linearize gradients, to make learning easier."""
    def __init__(self, *fn): super().__init__();  self.fn = nn.Sequential(*fn)
    def forward(self, x):
        y = self.fn(x)
        return y if x.shape[-1]<y.shape[-1] else x + y if x.shape == y.shape else x[..., :y.shape[-1]] + y
def to_np(x): return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
def cat(*a, dim=-1): return torch.cat(a, dim)



batch_size = 100
input_sz, embed_sz, action_sz = 4, 64, 64
lr = 1e-3

replay_buffer = ReplayBuffer(max_len=64) # of ReplaySample
replays_per_unroll = 1



def net(ins, outs, hidden=embed_sz):
    return nn.Sequential(
        SkipConnection(nn.Linear(ins, hidden)),
        SkipConnection(nn.ReLU(), nn.LayerNorm(hidden), nn.Linear(hidden, hidden)),
        nn.ReLU(), nn.LayerNorm(hidden), nn.Linear(hidden, outs),
    ).to(device)
dist = net(action_sz + 4*input_sz + 4*input_sz, action_sz + 1)
#   (0|action, src, dst) → (min_action, min_dist)
#   (Learns min-dist spanning trees that go to a destination, and that min-dist for tree-selection.)
#   (Usable for both gradient-ascent and self-imitation, and possibly good-embeddings-learning.)

optim = torch.optim.Adam([*dist.parameters()], lr=lr)



def act_dist(action, src, dst):
    """Returns a tuple `(action, dist)`, having refined the action (possibly `None`) & source & destination."""
    if action is None:
        action = torch.zeros(*src.shape[:-1], action_sz, device=device)
    ad = dist(cat(action, fold(fold(src)), fold(fold(dst))))
    return ad[..., :-1], ad[..., -1:]
def fold(x):
    """Increase sensitivity to 0…1 actions, doubling the size of the input."""
    return cat(x, 1 - 2*x.abs())
def as_goal(input): return cat(input[..., :2], torch.ones(*input.shape[:-1], 2, device=device))
# def dist_to_steps(dist): return 2 ** (dist-1)
# def steps_to_dist(step): return 1 + step.log2()
def dist_to_steps(dist): return dist # Log-space above seems to produce somewhat more accurate maps than this lin-space.
def steps_to_dist(step): return step



# For debugging.
def pos_histogram(plt, label):
    """That replay buffer contains lots of past positions. This func plots those as a 2D histogram, in addition to arrows indicating which action to take to get from anywhere to a random position."""
    with torch.no_grad():
        states = []
        for ch in replay_buffer:
            states.append(ch.state[..., 0:2])
        plt.hist2d((torch.cat(states)[:,0]+1).cpu().numpy(), torch.cat(states)[:,1].cpu().numpy(), bins=(100,100), range=((0,2), (0,1)), cmap='nipy_spectral', label=label, zorder=0)

        # Display action-arrows everywhere.
        GS = 16 # grid size
        dst_pos = pos_histogram.dst_pos
        dst = as_goal(dst_pos.expand(GS*GS, 2))
        plt.scatter(dst_pos[0,0].cpu(), dst_pos[0,1].cpu(), c='white', zorder=3)
        x, y = torch.linspace(0.,1.,GS, device=device), torch.linspace(0.,1.,GS, device=device)
        x = x.reshape(GS,1,1).expand(GS,GS,1).reshape(GS*GS,1)
        y = y.reshape(1,GS,1).expand(GS,GS,1).reshape(GS*GS,1)
        veloc = torch.zeros(GS*GS, 2, device=device)
        src = cat(x, y, veloc)
        acts, dists = act_dist(None, src, dst)
        plt.imshow(dists.reshape(GS,GS).t().cpu(), extent=(0,1,0,1), origin='lower', cmap='brg', zorder=1)
        plt.quiver(x.cpu(), y.cpu(), acts[:,0].reshape(GS,GS).t().cpu(), acts[:,1].reshape(GS,GS).t().cpu(), color='white', scale_units='xy', angles='xy', units='xy', zorder=2)
def onclick(event):
    """When clicking the distance/action plot, set destination and redraw."""
    if event.xdata is not None and event.ydata is not None and 0 < event.xdata < 1 and 0 < event.ydata < 1:
        pos_histogram.dst_pos = torch.tensor([[float(event.xdata), float(event.ydata)]], device=device)
        try: finish(False)
        except RuntimeError: pass # If an error, we're already in a `finish`.
import matplotlib
matplotlib.pyplot.subplots()[0].canvas.mpl_connect('button_press_event', onclick)
pos_histogram.dst_pos = torch.rand(1,2, device=device)



# The main loop, which steps the environment and trains `step`.
action = torch.randn(batch_size, action_sz, device=device)
goal = torch.randn(batch_size, embed_sz, device=device)
steps_to_goal = torch.rand(batch_size, 1, device=device)
state, hidden_state = env_init(batch_size=batch_size)
def maybe_reset_goal(input, input_to_goal_dist):
    """Changes the unroll's goal, when the previous one either gets reached or doesn't seem to be getting anywhere, to a goal that may be hard to predict (so that we may learn it).

    Returns a tuple of how many goals were reached and how many goals have timed out."""
    global goal, steps_to_goal
    with torch.no_grad():
        # dst = replay_buffer.sample_best().state # Not ever choosing `.as_goal` for simplicity.
        dst = as_goal((input + .3*torch.randn_like(input, device=device)).clamp(0,1)) # TODO: Cheating.
        old_dist, new_dist = input_to_goal_dist, act_dist(None, input, dst)[1]
        reached, out_of_time = old_dist < .5, steps_to_goal < 0
        change = reached | out_of_time

        goal = torch.where(change, dst, goal)
        steps_to_goal = torch.where(change, dist_to_steps(new_dist) + 4, steps_to_goal - 1)
    return reached.float().sum(), out_of_time.float().sum()
def replay(reached_vs_timeout):
    """Replays samples from the buffer.

    Picks samples at i<j<k and i+1, and refines min-distances and min-distance actions."""
    L = len(replay_buffer)
    if L < 4: return
    dist_loss, ground_loss, meta_loss = 0,0,0
    for _ in range(replays_per_unroll): # Look, concatenation and variable-management are hard.
        i = random.randint(0, L-3)
        I = i + 1
        j = random.randint(i+1, L-2)
        k = random.randint(j+1, L-1)
        a,A,b,c = replay_buffer[i], replay_buffer[I], replay_buffer[j], replay_buffer[k]

        # All actions & distances.
        no_act = torch.zeros(batch_size, action_sz, device=device)
        sa, sA, sb, sc, sg = a.state, A.state, b.state, c.state, c.as_goal
        aaA, daA = act_dist(no_act, sa, sA)
        aab, dab = act_dist(no_act, sa, sb)
        aac, dac = act_dist(no_act, sa, sc)
        aag, dag = act_dist(no_act, sa, sg)
        abc, dbc = act_dist(no_act, sb, sc)
        abg, dbg = act_dist(no_act, sb, sg)
        # Don't think if what we have is good enough.
        dist_cond = dab+dbc < k-i
        dist_target = torch.where(dist_cond, (dab+dbc).detach(), torch.tensor(float(k-i), device=device))
        act_target = torch.where(dist_cond, aab.detach(), a.action)
        #   TODO: (If we're doing the min, then should also do the min with `j-i + dbc` & `a.action`, and with `dab + k-j` & `aab.detach()`.)

        # Learn distance, to be the min of seen 1+log2 of steps (index-differences).
        def dstl(d,D):
            """Always nonzero, but fades if dist is too high; prefers lower dists."""
            mult = (dist_to_steps(d.detach()) - D) + 1
            mult = (mult+1).clamp(.1,15)
            if isinstance(D, int): D = torch.tensor(float(D), device=device)
            mult = torch.where(D > 1, mult, torch.tensor(1., device=device)) # Why, PyTorch?
            return (mult * (d - steps_to_dist(D)).square()).sum()
        dist_loss = dist_loss + dstl(daA, I-i)
        dist_loss = dist_loss + dstl(dab, j-i)
        dist_loss = dist_loss + dstl(dbc, k-j) + dstl(dbg, k-j)
        dist_loss = dist_loss + dstl(dac, dist_target) + dstl(dag, dist_target)

        # TODO: Learn `dist` (action-inputs always get their min-dist and min-dist-action learned (an auto-encoder, but potential DDPG search could then be performed)) (input zero-actions represent any-action dist-minima).
        #   TODO: Write down what exactly we want to learn here, as to-do items.

        # Learn ground-actions. And meta-actions to k, to be the dist-min of actions to j.
        def actl(d,D, a,A):
            """Tries to cut off anything not-min-dist, if in lin-space."""
            d = dist_to_steps(d)
            mult = (d.detach() - D.detach()).clamp(0,15)
            mult = mult + (mult > 0).float()
            if isinstance(D, int): D = torch.tensor(float(D), device=device)
            mult = torch.where( D>1.5, mult, torch.tensor(1., device=device) )
            return (mult * (a - A).square()).sum()
        ground_loss = ground_loss + actl(dab, j-i, aab, a.action)
        meta_loss = meta_loss + actl(dac, dist_target, aac, act_target)
        meta_loss = meta_loss + actl(dac, dist_target, aag, act_target)

    (dist_loss + ground_loss + meta_loss).backward()
    optim.step();  optim.zero_grad(True)

    # Log debugging info.
    log(0, False, pos = pos_histogram)
    log(1, False, reached = to_np(reached_vs_timeout[0]), timeout = to_np(reached_vs_timeout[1]))
    log(2, False, dist_loss = to_np(dist_loss / batch_size / replays_per_unroll), ground_loss = to_np(ground_loss / batch_size / replays_per_unroll), meta_loss = to_np(meta_loss / batch_size / replays_per_unroll))



for iter in range(500000):
    with torch.no_grad():
        state, hidden_state = env_step(state, hidden_state, action)
        full_state = cat(state, hidden_state)
        action = torch.zeros(batch_size, action_sz, device=device)
        input_to_goal_dist, action = act_dist(action, full_state, goal)
        if iter % 100 < 50: action = action + torch.randn(batch_size, action_sz, device=device)*.4 # TODO: (Seems to slightly improve dists, maybe?)

        replay_buffer.append(ReplaySample(
            full_state,
            action,
            as_goal(full_state),
            #   Want to go to places, not caring about final velocity.
        ))
        if random.randint(1,100)==1: state, hidden_state = env_init(batch_size=batch_size) # TODO: Resetting doesn't help…
    replay(maybe_reset_goal(full_state, input_to_goal_dist))
finish()

# TODO: Run & fix.
#   TODO: Why can't actions follow the gradient of distance? Why is action diversity getting washed out? Why has every attempt at self-imitation-learning failed?
#   TODO: Why isn't distance learned well?
#   (Maybe, try using the `dist` net?   …May actually be a good idea, allowing us to merge dist-net and action-net together (only 1 extra number for `act` to output). Abolish the explicit joint-embedding boundary, and gain in both efficiency and ease-of-use.)
#     (Gotta be real: `embed`-dists look like an NN gone bad, whereas `dist`-dists look reasonable… Though it does become nicer with enough time. Maybe the `dist` net is the way to go after all?…)







# …Could also, instead of maximizing uncertainty (which is 2× slower than pure dist estimation due to ensembling), maximize regret (real dist is lower/better than predicted dist, computed at unroll-time from encountered dst-embeddings) by goals.
#   (Probably a good idea anyway, since uncertainty-estimation is so slow.)
#   TODO: Maybe, also print the unroll-time dist-misprediction from the state at previous goal-setting to the present, since we know how many steps it's supposed to take? (Since the dist loss doesn't look like it improves at all, over 20k epochs.)
#     (…Would have been so much simpler to implement with merged dist & act, practically automatic…)




























# …Augmentating images in computer vision to make NN representations invariant to them, is equivalent to doing that for consecutive RNN steps with body/eye movement — though not exactly to full-RNN-state invariance/prediction…
#   …With `embed` dist learning, we're *kinda* doing something similar, making close-in-time images similar and far-in-time images distinct. Though whether this could possibly succeed in learning reprs on MNIST or something is not clear.

# …We only try to improve the reachability of one goal at a time, which is synonymous with "non-scalable". Is there no way to construct representations of exponentially-many goals, and update many goals at once… Can embedding-prediction make similar goals the same and distinct goals different?…
#   We can `embed` then measure distances in embedding-space rather than with a NN. Benefits are unclear.

# …Also, similarly to Random Network Distillation, the hypothetical teacher (`goal`-proposer) could maximize misprediction (available even at unroll-time): of the distance (should be -1), or with joint-embedding, of the next-state embedding…
#   This is pretty much what AdaGoal is about.





# …We can actually learn to achieve not just states, but arbitrary funcs of states (such as (pos,vel)→pos or state→return), by simply adding transitions from any states to those funcs-of-states to the replay buffer. (Very good: we can actually erase info, don't have to specify everything exactly.)
#   (…Those funcs can even be human input.)
#   …So is some goal-framework shaping up, like a func or class that gets called at every step and returns the new goal-state and whether the old one was reached, internally probabilistically keeping track of some recent most-error-on-dist states?…

# …Possibly, log [NAS-WithOut-Training](https://arxiv.org/pdf/2006.04647.pdf) score, where in a batch and in a NN layer, we compute per-example binary codes of ReLU activations (1 when input>0, 0 when input<0), then compute the sum of pairwise abs-differences?

# Usable for GANs, and for synth grad (least-magnitude): …DDPG's trick of "take the min of 2 nets" is really quite clever, since ReLU-nets are piecewise linear functions, so in non-trained regions, the linear pieces would be getting further and further away from data.

# …Simple goal-dependent sparsity (for avoiding catastrophic forgetting) on Transformer cells could be implemented as: (maybe have a neural net from goal to a mask, and) turn its 0-|0+ into 0|1, and multiply a cell's output by that (AKA drop some). If we pass-through the gradient from output's multiplication, then we can even start at all-1 and sparsify exactly as the tasks require; if we randomly turn on 0s, then we can un-sparsify too.

# Best control for `sn` would allow *arbitrary* human data (if limited, then `sn` is hardly an AI-based human-machine interface enabler) to control the goal (if just actions, then human capabilities won't get multiplied, `sn` will just be tiring and weird at best). Max sensitivity to an outcome, but min sensitivity to possible-outcomes: maximize [mutual info](https://en.wikipedia.org/wiki/Mutual_information), AKA channel capacity. (Or [pointwise MI](https://en.wikipedia.org/wiki/Pointwise_mutual_information): `log(p(y|x) / p(y))`.)
#   Without further grounding, we may only need an SSL method, to make the compressed-history the goal: the simple `leads_to(ev(prev))=sg ev(next)` BYOL-on-RNNs, or maybe even our embed-space dist-learning. Needs further research.
#   Do we want a separate channel for human actions, or would mixing them with all other data suffice?
#     (I guess for now, we should refine intent-amplification, and worry about plugging in intent later.)
#     (If we want a separate channel, then damn, I guess those `import env` one-liners aren't gonna fly with `sn` anymore.)
#   …However: making the agent learn to 'control' human data, and simply cutting out chunks of how-humans-want-to-be-controlled via reward, *may* create much richer experiences, without the tedious need to make the agent explore & acquire skills manually.



# …Gotta get back, back to the past: in `test.py`, implement ML pathfinding, and try to not just explore one graph but *learn* to explore `minienv`'s graphs. (I don't think any RL exploration method can *learn* to explore, only explore. So if it works, it's cool.)