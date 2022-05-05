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
from model.momentum_copy import MomentumCopy

import random



def env_init(batch_sz): # → state, hidden_state
    """Generate initial positions & velocities of the agent."""
    return torch.rand(batch_sz, 2, device=device), torch.zeros(batch_sz, 2, device=device)
def env_step(posit, veloc, accel): # → state, hidden_state
    """`env_step(state, hidden_state, action)`

    Add acceleration to velocity, and add velocity to position.

    Max action-acceleration is `1e-2`.

    There's a repulsor at `(.5, .5)` that adds a force proportional to `3e-5 / (pos - (.5, .5)).square()` to acceleration, to make exploration not as trivial. There's friction that multiplies velocity (hidden state) by `0.9` each step.

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

    let ddx = mx+fx, ddy = my+fy
    dx+=ddx, dy+=ddy, x+=dx, y+=dy
    dx*=.9, dy*=.9, x=(x%1+1)%1, y=(y%1+1)%1
    target.style.left = (x * innerWidth) + 'px'
    target.style.top = (y * innerHeight) + 'px'
    }, 50) // 20 FPS
    </script>
    ```"""
    accel = accel.detach()[..., :2]
    accel = accel * 1e-2 / (accel.square().sum(-1, keepdim=True).sqrt().clamp(1) + 1e-5)
    # force_center = torch.ones(posit.shape[0], 2, device=device)/2
    # force_len = (posit - force_center).square() + 1e-5
    # force = 3e-5 / force_len
    # accel = accel + force * (posit - force_center) / force_len # TODO: Can we learn anything if we disable the attractor? …Now, actually starts being able to often reach the destination by 10k… So maybe re-enable it?… We're having trouble learning anything outside the clusters at the corners.
    veloc = (veloc + accel) * .9
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



batch_sz = 50
input_sz, embed_sz, action_sz, noise_sz = 4, 64, 2, 4
lr = 1e-3

replay_buffer = ReplayBuffer(max_len=128) # of ReplaySample
replays_per_step = 2 # How many samples each unroll-step replays with loss. At least 2.
dist_levels = 2 # Each dist-level filters the predicted distance target to reduce it at each level.
#   (So the more levels we have, the more robust our action-learning will be to non-optimal policies in the replay buffer.)
#   (Each non-first level also performs a full `floyd` min-distance search (O(N**3)) for getting the best prediction targets.)



def net(ins, outs, hidden=embed_sz, layers=3):
    return nn.Sequential(
        SkipConnection(nn.Linear(ins, hidden)),
        *[SkipConnection(nn.ReLU(), nn.LayerNorm(hidden), nn.Linear(hidden, hidden)) for _ in range(layers)],
        SkipConnection(nn.ReLU(), nn.LayerNorm(hidden), nn.Linear(hidden, outs)),
    ).to(device)
embed = net(1 + input_sz + 1, embed_sz) # (src_or_dst, input, lvl) → emb
act = net(embed_sz + embed_sz + noise_sz, action_sz, layers=0) # (src, dst, noise) → action


dist = net(input_sz + input_sz + noise_sz + 1, action_sz + 1)
#   (src, dst, lvl) → (min_action, min_dist)
#   (Learns min-dist spanning trees that go to a destination, and that min-dist for tree-selection.)
#   (Usable for self-imitation, and possibly good-embeddings-learning. …"Good embeddings" implies that we can embed a single input in isolation, which we can't, really…)

dist_slow = MomentumCopy(dist, .999) # Stabilize targets to prevent collapse.
embed_slow = MomentumCopy(dist, .999) # Stabilize targets to prevent collapse.

optim = torch.optim.Adam([*embed.parameters(), *act.parameters(), *dist.parameters()], lr=lr)



def embed_(src_or_dst, input, lvl=1, nn=embed):
    """Locally-isometric embeddings: `dist_` between src & dst embeddings is an increasing func of the steps it takes to go."""
    src_or_dst = torch.full((*input.shape[:-1], 1), src_or_dst, device=device)
    lvl = torch.full((*input.shape[:-1], 1), lvl, device=device)
    return nn(cat(src_or_dst, input, lvl))
def dist_(src_emb, dst_emb):
    """Returns the shortest-path distance from src to dst, so that we can distinguish bad actions."""
    return 2 ** ((src_emb - dst_emb).abs().mean(-1, keepdim=True) + .25)
def act_(src_emb, dst_emb, noise = ...):
    """Returns the shortest-path action from src to dst."""
    if noise is ...:
        noise = torch.randn(*src_emb.shape[-1], noise_sz, device=device)
    return act(cat(src_emb, dst_emb, noise))



def act_dist(src, dst, noise=None, nn=dist, lvl=1.): # TODO: Work toward removing this.
    """Returns a tuple `(action, dist)`, having refined the action (possibly `None`) & source & destination."""
    if noise is None:
        noise = torch.randn(*src.shape[:-1], noise_sz, device=device)
    lvl = torch.full([*src.shape[:-1], 1], lvl, device=device)
    ad = nn(cat(2*src-1, 2*dst-1, noise, lvl))
    return ad[..., :-1], 2**ad[..., -1:]
def pos_only(input): return cat(input[..., :2], torch.ones(*input.shape[:-1], 2, device=device))



def floyd(d, *a):
    """
    Floyd—Warshall algorithm: computes all-to-all shortest distances & actions. Given a one-step adjacency matrix, gives its (differentiable) transitive closure.

    Computes `d[i,j] = min(d[i,j], d[i,k] + d[k,j])` until settled. Needs `O(N**3)` sequential time; `O(N)` GPU commands, for each considered midpoint.

    Inputs:
    - `d`: distances to minimize, shaped `(..., N, N, 1)`.
    - `*a`: first-step actions, shaped `(..., N, N, ?)`.

    Outputs: `(d, *a)`

    With this, self-imitation learning can make full use of a sampled minibatch, both using replayed actions and reusing solved subtasks for learning-time search. For this, when considering `i`-time and `j`-time samples `A` & `B`, `d,a` should be: if `i<j and i-j<dist(A,B)`, then `i-j, A→B`, else `dist(A,B), act(A,B)`.

    (Similar to [lifted structured embedding](https://arxiv.org/abs/1511.06452), but founded in pathfinding instead of better-loss considerations.) (Made possible by the ability to instantly know the RL-return (distance) between any 2 samples.)
    """
    assert d.shape[-3] == d.shape[-2] and d.shape[-1] == 1
    assert all(d.shape[-2] == A.shape[-3] == A.shape[-2] for A in a)
    for k in range(d.shape[-3]):
        d_through_midpoint = d[..., :, k:k+1, :] + d[..., k:k+1, :, :]
        cond = d < d_through_midpoint
        d = torch.where(cond, d, d_through_midpoint)
        a = [torch.where(cond, A, A[..., :, k:k+1, :]) for A in a]
    return d, *a



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
        dst = embed_(1, pos_only(dst_pos.expand(GS*GS, 2)))
        plt.scatter(dst_pos[0,0].cpu(), dst_pos[0,1].cpu(), c='white', zorder=3)
        x, y = torch.linspace(0.,1.,GS, device=device), torch.linspace(0.,1.,GS, device=device)
        x = x.reshape(1,GS,1).expand(GS,GS,1).reshape(GS*GS,1)
        y = y.reshape(GS,1,1).expand(GS,GS,1).reshape(GS*GS,1)
        veloc = torch.zeros(GS*GS, 2, device=device)
        src = embed_(0, cat(x, y, veloc))
        acts, dists = act_(src, dst), dist_(src, dst)
        plt.imshow(dists.reshape(GS,GS).cpu(), extent=(0,1,0,1), origin='lower', cmap='brg', zorder=1)
        plt.quiver(x.cpu(), y.cpu(), acts[:,0].reshape(GS,GS).cpu(), acts[:,1].reshape(GS,GS).cpu(), color='white', scale_units='xy', angles='xy', units='xy', zorder=2)

        # Unroll a sample trajectory, to visually gauge how well goal-chasing works.
        src = torch.cat((pos_histogram.src_pos, torch.zeros(1,2, device=device)), -1)
        dst = embed_(1, pos_only(dst_pos))
        mid = [src[0, :2]]
        for _ in range(64):
            action = act_(embed_(0, src), dst)
            state, hidden_state = env_step(src[:, :2], src[:, 2:], action)
            src = torch.cat((state, hidden_state), -1)
            mid.append(src[0, :2])
        plt.scatter((dst[0,0]+1).cpu(), dst[0,1].cpu(), c='white', zorder=5)
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
    """Sometimes changes the unroll's goal, when the previous one either gets reached or doesn't seem to be getting anywhere, to a goal that may be hard to predict (so that we may learn it).

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
        dst = embed_(1, dst, nn=embed_slow)
        src = embed_(0, input, nn=embed_slow)

        wrap_offsets = torch.tensor([
            [-1.,-1,0,0], [0,-1,0,0], [1,-1,0,0],
            [-1., 0,0,0], [0, 0,0,0], [1, 0,0,0],
            [-1., 1,0,0], [0, 1,0,0], [1, 1,0,0],
        ], device=device).unsqueeze(-2)
        old_dist = (embed_(0, pos_only(input + wrap_offsets), nn=embed_slow) - goal).abs().sum(-1, keepdim=True).min(0)[0]
        new_dist = dist_(src, dst)
        reached, out_of_time = old_dist < .05, steps_to_goal < 0
        change = reached | out_of_time

        goal = torch.where(change, dst, goal)
        steps_to_goal = torch.where(change, new_dist + 4, steps_to_goal - 1)

        reached_rating = 1 - (old_dist - .05).clamp(0,1) # Smoother.
        steps_to_goal_err = dist_(src, goal) - steps_to_goal
        steps_to_goal_regret = -steps_to_goal_err.clamp(None, 0) # (The regret of that long-ago state that `steps_to_goal` was evaluated at, compared to on-policy eval.)
    return reached_rating.sum(), reached.float().sum(), out_of_time.float().sum(), steps_to_goal_regret.sum()
def replay(reached_rating, reached, timeout, steps_to_goal_regret):
    """Replays samples from the buffer.

    Picks many faraway samples, computes all pairwise distances & actions, and makes the shortest paths on the minibatch's dense graph serve as prediction targets. (Search, to reuse solved subtasks.)

    Also incorporates the buffer's actions where they're closer than predicted ([self-imitation](https://arxiv.org/abs/1806.05635)), and learns to reach non-full-state goals."""
    N = replays_per_step
    if len(replay_buffer) < N: return
    samples = random.choices(replay_buffer, k=N) # Python 3.6+
    times = torch.stack([s.time for s in samples], 1) # batch_sz × N × 1
    states = torch.stack([s.state for s in samples], 1) # batch_sz × N × input_sz
    actions = torch.stack([s.action for s in samples], 1) # batch_sz × N × action_sz
    noises = torch.stack([s.noise for s in samples], 1) # batch_sz × N × noise_sz

    # Learn distances & actions.
    dist_loss, action_loss = 0,0
    def expand(x, src_or_dst):
        """N×… to N×N×…. `src_or_dst` is 1|2, representing which `N` is broadcasted."""
        return x.unsqueeze(src_or_dst).expand(batch_sz, N, N, x.shape[-1])
    def l_dist(d,D):
        """Prediction-loss that prefers lower-dists: always nonzero, but fades a bit if dist is too high."""
        with torch.no_grad(): mult = (d.detach() - D + 1).clamp(.3,3) # TODO: …Maybe we DO want to be able to pass in another `d` for here, namely, the previous level's distance prediction.
        # TODO: …Wait: non-first dist levels can just completely discard all distance-targets above the prev-level's. Do so, whenever the alternative `d` is non-`None`.
        #   TODO: …That current "gated prediction of the prev level" alternative would take a long time to get notified of new low-distances; to squeeze out all efficiency that we can, we should *sharp-filter* the real targets with the prev level…
        return (mult * (d - D).square()).sum()
    max_goals = max(len(s.goals) for s in samples)
    for lvl in range(dist_levels):
        last = lvl == dist_levels-1
        prev_lvl = ((lvl-1 if lvl>0 else 0) / (dist_levels-1))*2-1 # -1…1
        next_lvl = (lvl / (dist_levels-1))*2-1 # -1…1
        #   The first dist-level is grounded in itself, others in their previous levels.
        states_e = embed_(0, states, lvl=prev_lvl, nn=embed_slow) # Target.
        states_e1 = embed_(0, states, lvl=next_lvl) # Prediction.
        srcs, srcs1 = expand(states_e, 1), expand(states_e1, 1)
        noss = expand(noises, 1)
        for g in range(max_goals):
            # Predict distances & actions, for all src→dst pairs.
            goals = torch.stack([s.goals[g] for s in samples], 1)
            goals_e = embed_(1, states, lvl=prev_lvl, nn=embed_slow) # Target.
            goals_e1 = embed_(1, states, lvl=next_lvl) # Prediction.
            dsts, dsts1 = expand(goals_e, 2), expand(goals_e1, 2)
            n = noss
            # Compute prediction targets.
            with torch.no_grad():
                d = dist_(srcs, dsts)
                if last: a = act_(srcs, dsts, n)
                # Incorporate self-imitation knowledge: when a path is shorter than predicted, use it.
                #   (Maximize the regret by mining for paths that we'd regret not taking, then minimize it by taking them.)
                i, j = expand(times, 1), expand(times, 2)
                cond = i < j
                if lvl>0: cond = cond & (j-i < d) # First dist-level has to not filter sharply.
                d = torch.where(cond, j-i, d + 1).clamp(0) # Slowly penalize unconnected components.
                if last: a = torch.where(cond, expand(actions, 1), a)
                # Use the minibatch fully, by actually computing shortest paths.
                if g == 0: # (Only full states can act as midpoints for pathfinding, since goal-spaces have less info than the full-space.)
                    if lvl>0:
                        if not last: d = floyd(d)
                        else: d,a,n = floyd(d,a,n)
                else: # (Goals should take from full-state plans directly.)
                    cond = d0 < d
                    d = torch.where(cond, d0, d)
                    if last:
                        a = torch.where(cond, a0, a)
                        n = torch.where(cond, n0, n)
            if g == 0: # Preserve state-info of full-state for goals.
                if not last: d0 = d
                else: d0, a0, n0 = d, a, n
            # Compute losses.
            d1 = dist_(srcs1, dsts1)
            dist_loss = dist_loss + l_dist(d1, d)
            if last:
                a1 = act_(srcs1, dsts1, n)
                act_gating = (d < d1).float().detach()
                action_loss = action_loss + (act_gating * (a1 - a).square()).sum()
                #   (No momentum slowing for `d1` is less correct but easier to implement.)

    # Unreachability loss: after having tried and failed to reach goals, must remember that we failed, so that `floyd` doesn't keep thinking that distance is small.
    #   (Safe to use the estimated-once ever-decreasing `steps_to_goal` as the lower bound on dists because if any midpoint did know a path to `goal`, it would have taken it, so midpoints' timeout-distances are accurate too.)
    #   (On-policy penalization of unconnected components.)
    goals = torch.stack([s.goal for s in samples], 1) # batch_sz × N × input_sz
    goal_timeouts = torch.stack([s.goal_timeout for s in samples], 1) # batch_sz × N × 1
    goal_dists = act_dist(states, goals, noises, lvl=-1)[1] # TODO: …Should we just use `dist_` here?… …Do we need to re-embed `states` and `goals` right here just for this?…
    dist_loss = dist_loss + (goal_dists - goal_dists.max(goal_timeouts).detach()).square().sum()

    # TODO: Just go ahead and move to embeddings (& DDPG), if only so that the compute-cost is substantially lower (and, diversity of actions should no longer be affected by distances being overly-huge).

    # TODO: Run & fix.

    (dist_loss + action_loss).backward()
    optim.step();  optim.zero_grad(True)
    dist_slow.update();  embed_slow.update()

    # Log debugging info.
    log(0, False, pos = pos_histogram)
    log(1, False, reached_rating = to_np(reached_rating), reached = to_np(reached), timeout = to_np(timeout), dist_regret = to_np(steps_to_goal_regret))
    log(2, False, dist_loss = to_np(dist_loss / batch_sz / N), action_loss = to_np(action_loss / batch_sz / N))



for iter in range(500000):
    with torch.no_grad():
        full_state = cat(state, hidden_state)
        noise = torch.randn(batch_sz, noise_sz, device=device)
        action = act_(embed_(0, full_state), goal, noise)
        # if iter % 100 < (70 - iter//1000): action = goal[..., :2] - state # TODO: (Literally very much cheating, suggesting trajectories that go toward the goals.) (…Somehow, we've reached a state where we don't need this cheating. Maybe remove.)
        if iter % 100 >= 50:
            if random.randint(1,2)==1: action = torch.randn(batch_sz, action_sz, device=device)

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

# TODO: Run & fix.








# TODO: Return to locally-isometric embeddings. (With faraway-sample batches, brings big benefits: NN calls go from `O(N*N)` to `O(N)`, so that `floyd` can actually be not trivially-cheap in comparison.)
#   TODO: Replace all 4 remaining `act_dist` calls with `act_` and `dist_` calls on embeddings. Especially in the loss.
# TODO: Have `next_embed(prev_emb, action) → next_emb`.
#   (Not just a gimmick: helps with stochasticity, DDPG, and possibly actions that don't change state if we're not lazy in rewriting the dist-loss.)
#   TODO: Have a loss that trains it with consecutive samples. (Literally BYOL. Or MuZero.)
#   TODO: Use this for DDPG: grad-min the distance of post-action embeddings. (If embs are locally-isometric and kinda preserve angles, then the loss should be very smooth in `post_emb` space, so if action-space is also smooth, learning should be very quick.)



# (…Might even wrap this in a `model/` framework that manages the replay, with arbitrarily-many possibly-async goal states and everything?…)




























# …Augmentating images in computer vision to make NN representations invariant to them, is equivalent to doing that for consecutive RNN steps with body/eye movement (inputs, in any envs). With `embed` dist learning, we're *kinda* doing something similar, making close-in-time images similar and far-in-time images distinct. Though whether this could possibly succeed in learning reprs on CIFAR10 must be tested.






# Best control for `sn` would allow *arbitrary* human data (if limited, then `sn` is hardly an AI-based human-machine interface enabler) to control the goal (if just actions, then human capabilities won't get multiplied, `sn` will just be tiring and weird at best). Max sensitivity to an outcome, but min sensitivity to possible-outcomes: maximize [mutual info](https://en.wikipedia.org/wiki/Mutual_information), AKA channel capacity. (Or [pointwise MI](https://en.wikipedia.org/wiki/Pointwise_mutual_information): `log(p(y|x) / p(y))`.)
#   Without further grounding, we may only need an SSL method, to make the compressed-history the goal: the simple `leads_to(ev(prev))=sg ev(next)` BYOL-on-RNNs, or maybe even our embed-space dist-learning. Needs further research.
#   Do we want a separate channel for human actions, or would mixing them with all other data suffice?
#     (I guess for now, we should refine intent-amplification, and worry about plugging in intent later.)
#     (If we want a separate channel, then damn, I guess those `import env` one-liners aren't gonna fly with `sn` anymore.)
#   …However: making the agent learn to 'control' human data, and simply cutting out chunks of how-humans-want-to-be-controlled via reward, *may* create much richer experiences, without the tedious need to make the agent explore & acquire skills manually.