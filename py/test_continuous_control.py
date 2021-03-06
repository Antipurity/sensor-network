"""
`sensornet` requires a good way for humans to control what their computers do with their sensor data.

- Keyboard? Tedious.
- Webcam and body movements? Not only even more tedious, but also forces each user to create a whole new language for control, which in practice means only having a couple of predefined gestures. Learning that 'language' by maximizing mutual info AKA channel capacity might as well be impossible without Internet-scale data (like large language models use), due to how clean and snappy (and zero-shot) the users expect their interactions to be.
- Any BMI readings? Same caveats as the webcam point above, though less physically tiring.

The core issue is that humans need to specify every action one-by-one. But they only care about goals. And so, we must learn to reach any goal from anywhere, as we do here.

Empower users, don't burden them.

# Neural pathfinding

Everything is conditioned on a user-specified goal (an input: anything, such as reward or actual-inputs or [te](https://cliport.github.io/)[xt](https://github.com/openai/CLIP)); the metric to learn & minimize is how many steps it takes to get there.

It's not complicated.

It's so not-complicated that classical RL techniques (learn the real dists, and self-imitate actions with lower dists, then at unroll-time pick the best of a few actions) perform better than anything we try, at least in this trivial env.

- `floyd`, where we take a minibatch and do all distance-predictions and find shortest paths there? Only hurts.
  - Removing the i==j diagonal from pairwise prediction matrices ([as implemented here](https://discuss.pytorch.org/t/keep-off-diagonal-elements-only-from-square-matrix/54379)) for efficiency. Decreases performance.

- Distance levels, where each next dist-level filters its prediction-targets to make them lower than prev dist-level? They slow down dist-learning too much.

- Action-independent distance? No, easy cheating by considering several actions at unroll-time is at least twice as fast.

(…Also, the inverse-RL equivalent with goal-conditioning is easy, because we don't need to learn the value function, only find a goal for which the trajectory always decreases its distance-to-goal appropriately.)
"""



import torch
import torch.nn as nn
device = 'cuda' if torch.cuda.is_available() else 'cpu'



from model.log import log, clear, finish
from model.momentum_copy import MomentumCopy

import random



def env_init(batch_sz): # → state, hidden_state
    """Generate initial positions & velocities of the agent."""
    return torch.rand(batch_sz, 2, device=device), torch.zeros(batch_sz, 2, device=device)
def env_step(posit, veloc, accel): # → state, hidden_state
    """`env_step(state, hidden_state, action)`

    Add acceleration to velocity, and add velocity to position.

    Max action-acceleration is `1e-2`. There's friction that multiplies velocity (hidden state) by `0.9` each step.

    (There used to be a repulsor at `(.5, .5)` that would add a force proportional to `3e-5 / (pos - (.5, .5)).square()` to acceleration, to make exploration not as trivial. But it wasn't very interesting, and just annoying for visualizations. A proper maze would have been much better.)

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
    if (m) mx /= m/1e-2, my /= m/1e-2

    let flen = ((x-.5)**2 + (y-.5)**2 + 1e-5), f = 3e-5 / flen
    let fx = f * (x-.5)/flen, fy = f * (y-.5)/flen

    let ddx = mx+0*fx, ddy = my+0*fy // No repulsor.
    dx+=ddx, dy+=ddy, x+=dx, y+=dy
    dx*=.9, dy*=.9, x=(x%1+1)%1, y=(y%1+1)%1
    target.style.left = (x * innerWidth) + 'px'
    target.style.top = (y * innerHeight) + 'px'
    }, 50) // 20 FPS
    </script>
    ```"""
    accel = accel.detach()[..., :2]
    accel = accel * 1e-2 / (accel.square().sum(-1, keepdim=True).sqrt().clamp(1) + 1e-5)
    force_center = torch.ones(posit.shape[0], 2, device=device)/2
    force_len = (posit - force_center).square() + 1e-5
    force = 3e-5 / force_len
    accel = accel + force * (posit - force_center) / force_len
    #   (The repulsor doesn't really add interesting dynamics, nor a challenge.)
    #     (Hardly a big exploration challenge that it was originally envisioned as. A maze would have been so much better.)
    veloc = (veloc + accel).clamp(-2,2) * .9
    posit = torch.remainder(posit + veloc, 1.)
    return posit, veloc



class ReplaySample:
    """A snapshot of a trajectory's step.
    - `.time`: the iteration index of this step. Differences of these are distances between steps.
    - `.state`: inputs (position & velocity in this env).
    - `.action`: outputs (acceleration in this env).
    - `.goals`: potential goal-spaces that this can be described as; the unrolls can then go to any of those. A tuple, with the first element being the state and the rest being same-shape whatever.
    - `.noise`: the noise that the `.action` used to get generated. Integrated into prediction targets so that randomness doesn't get averaged out.
    - `.goal` and `.goal_timeout`: the destination that the unroll was trying to reach, and the estimation of how many steps that would take (see `maybe_reset_goal`). Used for making sure that unreachable goals do eventually get marked as unreachable in `dist`, and stop misleading the `floyd` search into uselessness."""
    __slots__ = ('time', 'state', 'action', 'noise', 'goals', 'goal', 'goal_timeout')
    def __init__(self, time, state, action, noise, goals, goal, goal_timeout):
        s = self
        s.time, s.state, s.action, s.noise, s.goals, s.goal, s.goal_timeout = time, state, action, noise, goals, goal, goal_timeout
class ReplayBuffer:
    """Stores the in-order sequence of most-recent events. Needs `max_len=1024`. Supports `len(rb)`, `rb.append(data)`, `rb[index]`."""
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



class SkipConnection(nn.Module):
    """Linearize gradients, to make learning easier."""
    def __init__(self, *fn): super().__init__();  self.fn = nn.Sequential(*fn)
    def forward(self, x):
        y = self.fn(x)
        return y if x.shape[-1]<y.shape[-1] else x + y if x.shape == y.shape else x[..., :y.shape[-1]] + y
def to_np(x): return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
def cat(*a, dim=-1): return torch.cat(a, dim)



batch_sz = 100
input_sz, embed_sz, action_sz, noise_sz = 4, 64, 2, 4
dist_levels = 2 # For quantile regression.
#   (If an action leads to a plan that only rarely leads to a good result, then min-extraction would work on refining that plan as soon as possible. It also gives robustness to non-optimal policies in the replay buffer. But, the min is slower to learn than the average.)
#   Hard to learn, so here, we've basically disabled it.
lr = 1e-3

replay_buffer = ReplayBuffer(max_len=128) # of ReplaySample
replays_per_step = 2 # How many samples each unroll-step replays with loss. At least 2.
#   (If this N is more than 2, `floyd` does more than nothing.)
#   (Experiments seem to indicate no benefit to using `floyd`.)
#   The from-replay transitions (`N*(N-1)/2`) should be kept constant by varying `batch_size`.
#     N=2, batch_sz=100, 100% chance to self-imitate: pretty good at 9k.
#     N=5, batch_sz=10, 100% chance to self-imitate (making `floyd` useless): pretty good at 9k.
#       50%: worse.



def net(ins, outs, hidden=embed_sz, layers=3):
    return nn.Sequential(
        SkipConnection(nn.Linear(ins, hidden)),
        *[SkipConnection(nn.ReLU(), nn.LayerNorm(hidden), nn.Linear(hidden, hidden)) for _ in range(layers)],
        SkipConnection(nn.ReLU(), nn.LayerNorm(hidden), nn.Linear(hidden, outs)),
    ).to(device)
#   (Min-dist and min-dist action.)
act = net(input_sz + input_sz + noise_sz, action_sz, layers=3) # (src, dst, noise) → act
dist = net(input_sz + action_sz + input_sz, dist_levels, layers=3) # (src, act, dst) → dist

act_slow = MomentumCopy(act, .999) # Stabilize targets to prevent collapse.
dist_slow = MomentumCopy(dist, .999) # Stabilize targets to prevent collapse.

optim = torch.optim.Adam([*act.parameters(), *dist.parameters()], lr=lr)



def act_(src, dst, noise=..., nn=act):
    """Determines where to go to reach a goal."""
    assert src.shape == dst.shape
    if noise is ...:
        noise = torch.randn(*src.shape[:-1], noise_sz, device=device)
    return nn(cat(src, dst, noise))
def dist_(src, act, dst, nn=dist):
    """`r`: the (distribution of the) metric that actions should minimize. `r[..., -1:]` is the actual one-number metric, others are just a ladder for learning that."""
    assert src.shape == dst.shape
    return 2 ** nn(cat(src, act, dst))



def pos_only(input): return cat(input[..., :2], torch.ones(*input.shape[:-1], 2, device=device))



def floyd(d, *a):
    """
    Floyd—Warshall algorithm: computes all-to-all shortest distances & actions. Given a one-step adjacency matrix, gives its (differentiable) transitive closure.

    Computes `d[i,j] = min(d[i,j], d[i,k] + d[k,j])` until settled. Needs `O(N**3)` sequential time; `O(N)` GPU commands, for each considered midpoint.

    Inputs:
    - `d`: distances to minimize, shaped `(..., N, N, 1)`.
    - `*a`: first-step actions, shaped `(..., N, N, ?)`.

    Outputs: `(d, *a)`

    With this, self-imitation learning can make full use of a sampled minibatch, both using replayed actions and reusing solved subtasks for learning-time search. (Though in practice, we can't get good performance.)

    (Similar to [lifted structured embedding](https://arxiv.org/abs/1511.06452), but founded in pathfinding instead of better-loss considerations.) (Made possible by the ability to instantly know the RL-return (distance) between any 2 samples.)
    """
    assert d.shape[-3] == d.shape[-2] and d.shape[-1] == 1
    assert all(d.shape[:-1] == A.shape[:-1] for A in a)
    for k in range(d.shape[-3]):
        d_through_midpoint = d[..., :, k:k+1, :] + d[..., k:k+1, :, :]
        disallow = d < d_through_midpoint
        d = torch.where(disallow, d, d_through_midpoint)
        a = [torch.where(disallow, A, A[..., :, k:k+1, :]) for A in a]
    return d, *a



# For debugging.
def pos_histogram(plt, label, *_):
    """That replay buffer contains lots of past positions. This func plots those as a 2D histogram, in addition to arrows indicating which action to take to get from anywhere to a random position."""
    with torch.no_grad():
        states = []
        for ch in replay_buffer:
            states.append(ch.state[..., 0:2])
        plt.hist2d((torch.cat(states)[:,0]+1).cpu().numpy(), torch.cat(states)[:,1].cpu().numpy(), bins=(100,100), range=((0,2), (0,1)), cmap='nipy_spectral', label=label, zorder=0)

        # Display action-arrows everywhere.
        GS = 16 # grid size
        dst_pos = pos_histogram.dst_pos
        dst = pos_only(dst_pos.expand(GS*GS, 2))
        plt.scatter(dst_pos[0,0].cpu(), dst_pos[0,1].cpu(), c='white', zorder=3)
        x, y = torch.linspace(0.,1.,GS, device=device), torch.linspace(0.,1.,GS, device=device)
        x = x.reshape(1,GS,1).expand(GS,GS,1).reshape(GS*GS,1)
        y = y.reshape(GS,1,1).expand(GS,GS,1).reshape(GS*GS,1)
        veloc = torch.zeros(GS*GS, 2, device=device)
        src = cat(x, y, veloc)
        acts = act_(src, dst)
        dists = dist_(src, acts, dst)[..., -1:]
        plt.imshow(dists.reshape(GS,GS).cpu(), extent=(0,1,0,1), origin='lower', cmap='brg', zorder=1)
        plt.quiver(x.cpu(), y.cpu(), acts[:,0].reshape(GS,GS).cpu(), acts[:,1].reshape(GS,GS).cpu(), color='white', scale_units='xy', angles='xy', units='xy', zorder=2)

        # Unroll a sample trajectory, to visually gauge how well goal-chasing works.
        #   (Cheat here too.)
        src = cat(pos_histogram.src_pos, torch.zeros(1,2, device=device))
        dst = pos_only(dst_pos)
        mid = [src[0, :2]]
        for _ in range(64):
            action = act_(src, dst)
            D = dist_(src, action, dst)[..., -1:]
            action, D = cheat(src, torch.randn_like(action), dst, action, D)
            action, D = cheat(src, torch.randn_like(action), dst, action, D)
            action, D = cheat(src, torch.randn_like(action), dst, action, D)
            state, hidden_state = env_step(src[:, :2], src[:, 2:], action)
            src = cat(state, hidden_state)
            mid.append(src[0, :2])

        plt.scatter((dst_pos[0,0]+1).cpu(), dst_pos[0,1].cpu(), c='white', zorder=5)
        xy = torch.stack(mid,0)
        pre, uv = xy[:-1], xy[1:] - xy[:-1]
        plt.quiver((pre[:,0]+1).cpu(), pre[:,1].cpu(), uv[:,0].cpu(), uv[:,1].cpu(), color='white', scale=1, scale_units='xy', angles='xy', units='xy', zorder=4)
def onclick(event):
    """When clicking the distance/action plot, set destination and redraw."""
    x, y = event.xdata, event.ydata
    if x is not None and y is not None and 0 < x < 2 and 0 < y < 1:
        if x < 1:
            pos_histogram.dst_pos = torch.tensor([[float(x), float(y)]], device=device)
        else:
            pos_histogram.src_pos = torch.tensor([[float(x)-1, float(y)]], device=device)
        try: finish(False)
        except RuntimeError: pass # If an error, we're already in a `finish`.
import matplotlib
matplotlib.pyplot.subplots()[0].canvas.mpl_connect('button_press_event', onclick)
pos_histogram.dst_pos = torch.ones(1,2, device=device)/2
pos_histogram.src_pos = torch.rand(1,2, device=device)



# The main loop, which steps the environment and trains `step`.
action = torch.randn(batch_sz, action_sz, device=device)
goal = torch.randn(batch_sz, input_sz, device=device)
steps_to_goal = torch.rand(batch_sz, 1, device=device)
state, hidden_state = env_init(batch_sz=batch_sz)
def maybe_reset_goal(input):
    """Sometimes changes the unroll's goal, when the previous path runs longer than predicted, to a goal that may be hard to predict (so that we may learn it).

    (Exact reachability is kinda impossible in continuous envs with func approximation, so we only change on timeout.)

    Possible destinations: a random state or a random XY position from `replay_buffer`. (It is important that the loss learns that unreachable goals are unreachable until proven otherwise, or else the `floyd` search would keep exploiting out-of-distribution "small" ghost-dists and prevent improvement.)

    Returns: some interesting-to-plot values for `replay`."""
    global goal, steps_to_goal
    with torch.no_grad():
        s = random.choice(replay_buffer)
        dst = s.state[torch.randperm(batch_sz, device=device)]
        if random.randint(1,2) == 1: # In 50% of the cases, go to an XY position.
            dst = pos_only(dst)
        if random.randint(1,4) == 1: # In 25% of cases, try go to a non-replay-buffer destination.
            dst = pos_only(torch.rand(batch_sz, input_sz, device=device))
            # (Note: this doesn't help make the blobs more expansive. Doesn't do anything of note, except for *maybe* better-looking distances.)
        src = input

        out_of_time = steps_to_goal < 0
        goal = torch.where(out_of_time, dst, goal)
        act = act_(src, dst, nn=act_slow)
        steps_to_goal = torch.where(out_of_time, dist_(src, act, dst, nn=dist_slow)[..., -1:] + 4, steps_to_goal - 1)

        steps_to_goal_err = dist_(src, act, goal, nn=dist_slow)[..., -1:] - steps_to_goal
        steps_to_goal_regret = -steps_to_goal_err.clamp(None, 0) # (The regret of that long-ago state that `steps_to_goal` was evaluated at, compared to on-policy eval.)
    return out_of_time.float().sum(), steps_to_goal_regret.sum()
def replay(timeout, steps_to_goal_regret):
    """Replays samples from the buffer.

    Picks many faraway samples, computes all pairwise distances & actions, and makes the shortest paths on the minibatch's dense graph serve as prediction targets. (Search, to reuse solved subtasks.)

    Also incorporates the buffer's actions where they're closer than predicted ([self-imitation](https://arxiv.org/abs/1806.05635)), and learns to reach non-full-state goals."""
    N = replays_per_step
    if len(replay_buffer) < N: return
    indices = [random.randrange(len(replay_buffer)-1) for _ in range(N)]
    samples = [replay_buffer[i] for i in indices]
    times = torch.stack([s.time for s in samples], 1) # batch_sz × N × 1
    states = torch.stack([s.state for s in samples], 1) # batch_sz × N × input_sz
    actions = torch.stack([s.action for s in samples], 1) # batch_sz × N × action_sz
    noises = torch.stack([s.noise for s in samples], 1) # batch_sz × N × noise_sz

    # Learn distances & actions.
    def expand(src_or_dst, x):
        """_×N×… to _×N×N×…. `src_or_dst` is 0|1, representing which `N` is broadcasted."""
        return x.unsqueeze(src_or_dst+1).expand(batch_sz, N, N, x.shape[-1])
    def l_dist(d_p, d_t):
        """A prediction loss that extracts min-dists, by making each next dist-level only predict the target if it's less than the prev dist-level (plus eps=1 for getting around func-approx)."""
        assert d_t.shape[-1] == 1
        with torch.no_grad():
            # Quantile regression.
            p = 1. / (torch.arange(0, dist_levels, device=d_p.device) + 2)
            mult = torch.where(d_p < d_t, p, 1-p)
        return (mult * (d_p - d_t).square()).sum()
        #   (Proper quantile regression does need `.abs()` instead of `.square()`, but that takes too long to learn.)
    dist_loss, action_loss = 0,0
    noss = expand(0, noises)
    srcs = expand(0, states)
    for g in range(max(len(s.goals) for s in samples)):
        # Predict distances & actions, for all src→dst pairs.
        goals = torch.stack([s.goals[g] for s in samples], 1)
        dsts = expand(1, goals)
        n = noss
        # Compute prediction targets.
        with torch.no_grad():
            a_i = act_(srcs, dsts, n, nn=act_slow) # Initial.
            d_i = dist_(srcs, a_i, dsts, nn=dist_slow)[..., -1:].clamp(1) # Initial min-dist.
            # Incorporate self-imitation knowledge: when a path is shorter than predicted, use it.
            #   (Maximize the regret by mining for paths that we'd regret not taking, then minimize it by taking them.)
            i, j = expand(0, times), expand(1, times)
            unmask = torch.rand_like(d_i) < 1. # .5 for 50%.
            #   For combining. Can't leave it to i>j since undoing a transition can take a LOT of extra work.
            #   (But it doesn't help.)
            #   (And, while this masking is a glaring weak spot, attempts to ensure that each path always includes at least 1 from-replay action only hurt performance, a lot.)
            cond = (i < j) & (unmask | (j-i < d_i))
            d_j = torch.where(cond, j-i, d_i+1) # `+1` *slowly* penalizes unconnected components.
            a_j = torch.where(cond, expand(0, actions), a_i)
            # Use the minibatch fully, by actually computing shortest paths, but only for full states.
            if g == 0: # (Only full states can act as midpoints for pathfinding.)
                d_t, a_t, n = floyd(d_j, a_j, n)
                d_t, a_t, n = d_t, a_t, n
                d0, a0, n0 = d_t, a_t, n # Preserve state-info of full-state for goals.
            else: # (Goals incorporate full-state plans directly.)
                cond = d0 < d_j
                d_t = torch.where(cond, d0, d_j)
                a_t = torch.where(cond, a0, a_j)
                n = torch.where(cond, n0, n)

        # Learn the max-regret distance, to compare policy-action distances with.
        dist_loss = dist_loss + l_dist(dist_(srcs, a_t, dsts), d_t)

        # (Note: `cheat`ing right here isn't terrible, but not as good as `cheat`ing at unroll-time.)

        # Imitate actions wherever we found a better path, to minimize regret.
        #   SIL: https://arxiv.org/pdf/1806.05635.pdf
        #   (Differences: no explicit probabilities; `floyd`; upper-bounding the `regret`; no `dist_loss += 1e-2 * regret.square().sum()`.)
        a_p = act_(srcs, dsts, n) # Prediction.
        with torch.no_grad():
            # (With N=5, `nn=dist_slow` performs better than `nn=dist`.)
            d_p = dist_(srcs, a_p, dsts, nn=dist_slow)[..., -1:]
            regret = (d_p - d_t).clamp(0)
        action_loss = action_loss + (regret.detach().clamp(0, 15) * (a_p - a_t).square()).sum()



    (dist_loss + action_loss).backward()
    optim.step();  optim.zero_grad(True)
    act_slow.update();  dist_slow.update()

    # Log debugging info.
    M = batch_sz * N
    log(0, False, pos = pos_histogram)
    log(1, False, timeout = to_np(timeout), dist_regret = to_np(steps_to_goal_regret))
    log(2, False, dist_loss = to_np(dist_loss / M), action_loss = to_np(action_loss / M))



def cheat(src, act, dst, action, D, nn=dist):
    """Unroll-time search, AKA cheating."""
    d = dist_(src, act, dst, nn=nn)[..., -1:]
    c = d < D
    return torch.where(c, act, action), torch.where(c, d, D)
for iter in range(500000):
    with torch.no_grad():
        full_state = cat(state, hidden_state)
        noise = torch.randn(batch_sz, noise_sz, device=device)
        action = act_(full_state, goal, noise)

        # Epsilon-greedy exploration.
        if iter % 100 < 50: # (With cheating, we don't really need this.)
            if random.randint(1,2)==1: action = torch.randn_like(action)

        # A small unroll-time search.
        #   (Too good. Can't go without.)
        D = dist_(full_state, action, goal)[..., -1:]
        action, D = cheat(full_state, torch.randn_like(action), goal, action, D)
        action, D = cheat(full_state, torch.randn_like(action), goal, action, D)
        action, D = cheat(full_state, torch.randn_like(action), goal, action, D)

        replay_buffer.append(ReplaySample(
            torch.full((batch_sz, 1), iter, dtype=torch.float32, device=device),
            full_state,
            action,
            noise,
            (full_state, pos_only(full_state)),
            goal,
            steps_to_goal,
        ))

        state, hidden_state = env_step(state, hidden_state, action)
    replay(*maybe_reset_goal(full_state))
finish()