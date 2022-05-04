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
    # force_center = torch.ones(posit.shape[0], 2, device=device)/2
    # force_len = (posit - force_center).square() + 1e-5
    # force = 3e-5 / force_len
    # accel = accel + force * (posit - force_center) / force_len # TODO: Can we learn anything if we disable the attractor?
    veloc = (veloc + accel) * .99
    posit = torch.remainder(posit + veloc, 1.)
    return posit, veloc



class ReplaySample:
    """A snapshot of a trajectory's step. `goals` is a tuple, with the first element being the state and the rest being same-shape whatever; the NN would then learn to go to any goal when requested."""
    __slots__ = ('time', 'state', 'action', 'goals', 'noise')
    def __init__(self, time, state, action, goals, noise):
        self.time, self.state, self.action, self.goals, self.noise = time, state, action, goals, noise
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
input_sz, embed_sz, action_sz, noise_sz = 4, 64, 2, 8
lr = 1e-3

replay_buffer = ReplayBuffer(max_len=64) # of ReplaySample
replays_per_step = 4 # How many samples each unroll-step replays with loss. At least 2.
dist_levels = 2 # Each dist-level filters the predicted distance target to reduce it at each level.
#   (So the more levels we have, the more robust our action-learning will be to non-optimal policies in the replay buffer.)
#   (Each non-first level also performs a full `floyd` min-distance search (O(N**3)) for getting the best prediction targets.)



def net(ins, outs, hidden=embed_sz):
    return nn.Sequential(
        SkipConnection(nn.Linear(ins, hidden)),
        SkipConnection(nn.ReLU(), nn.LayerNorm(hidden), nn.Linear(hidden, hidden)),
        SkipConnection(nn.ReLU(), nn.LayerNorm(hidden), nn.Linear(hidden, hidden)),
        SkipConnection(nn.ReLU(), nn.LayerNorm(hidden), nn.Linear(hidden, hidden)),
        SkipConnection(nn.ReLU(), nn.LayerNorm(hidden), nn.Linear(hidden, hidden)),
        nn.ReLU(), nn.LayerNorm(hidden), nn.Linear(hidden, outs),
    ).to(device)
dist = net(input_sz + input_sz + noise_sz + 1, action_sz + 1)
#   (src, dst, lvl) → (min_action, min_dist)
#   (Learns min-dist spanning trees that go to a destination, and that min-dist for tree-selection.)
#   (Usable for self-imitation, and possibly good-embeddings-learning. …"Good embeddings" implies that we can embed a single input in isolation, which we can't, really…)
#   Loss-wise:
#     - (…This was updated since then though…)
#     - `min_dist` with replayed `action`s is the index-diff;
#     - `min_dist` with 0 is the index-diff;
#     - `min_action` with 0 is the replayed `action`.
#     - (For tighter losses, we sample not just faraway i→j but i→j→k, and make i→k losses also consider i→j + j→k for exponential-combining. Has not been ablated.)
#     - DDPG: generate an action from 0 and estimate it.
#     - All losses are multiplied by how much the prediction-target improves/lessens the dist.

dist_slow = MomentumCopy(dist, .999) # Stabilize targets to prevent collapse.

optim = torch.optim.Adam([*dist.parameters()], lr=lr)



def act_dist(src, dst, noise=None, nn=dist, lvl=1.):
    """Returns a tuple `(action, dist)`, having refined the action (possibly `None`) & source & destination."""
    if noise is None:
        noise = torch.randn(*src.shape[:-1], noise_sz, device=device)
    lvl = torch.full([*src.shape[:-1], 1], lvl, device=device)
    ad = nn(cat(2*src-1, 2*dst-1, noise, lvl))
    return ad[..., :-1], 2**ad[..., -1:]
def pos_only(input): return cat(input[..., :2], torch.ones(*input.shape[:-1], 2, device=device))



def floyd(d, a = None):
    """
    Floyd—Warshall algorithm: computes all-to-all shortest distances & actions. Given a one-step adjacency matrix, gives its (differentiable) transitive closure.

    Computes `d[i,j] = min(d[i,j], d[i,k] + d[k,j])` until settled. Needs `O(N**3)` sequential time; `O(N)` GPU commands, for each considered midpoint.

    Inputs:
    - `d`: distances to minimize, shaped `(..., N, N, 1)`.
    - `a = None`: first-step actions, shaped `(..., N, N, ?)`.

    Outputs: `(d,a)`

    With this, self-imitation learning can make full use of a sampled minibatch, both using replayed actions and reusing solved subtasks for learning-time search. For this, when considering `i`-time and `j`-time samples `A` & `B`, `d,a` should be: if `i<j and i-j<dist(A,B)`, then `i-j, A→B`, else `dist(A,B), act(A,B)`.

    (Similar to [lifted structured embedding](https://arxiv.org/abs/1511.06452), but founded in pathfinding instead of better-loss considerations.) (Made possible by the ability to instantly know the RL-return (distance) between any 2 samples.)
    """
    assert d.shape[-1] == 1
    assert d.shape[-3] == d.shape[-2]
    if a is not None: assert d.shape[-2] == a.shape[-3] == a.shape[-2]
    for k in range(d.shape[0]):
        d_through_midpoint = d[..., :, k:k+1, :] + d[..., k:k+1, :, :]
        cond = d < d_through_midpoint
        d = torch.where(cond, d, d_through_midpoint)
        if a is not None: a = torch.where(cond, a, a[..., :, k:k+1, :])
    return d,a



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
        dst = pos_only(dst_pos.expand(GS*GS, 2))
        plt.scatter(dst_pos[0,0].cpu(), dst_pos[0,1].cpu(), c='white', zorder=3)
        x, y = torch.linspace(0.,1.,GS, device=device), torch.linspace(0.,1.,GS, device=device)
        x = x.reshape(1,GS,1).expand(GS,GS,1).reshape(GS*GS,1)
        y = y.reshape(GS,1,1).expand(GS,GS,1).reshape(GS*GS,1)
        veloc = torch.zeros(GS*GS, 2, device=device)
        src = cat(x, y, veloc)
        acts, dists = act_dist(src, dst)
        plt.imshow(dists.reshape(GS,GS).cpu(), extent=(0,1,0,1), origin='lower', cmap='brg', zorder=1)
        plt.quiver(x.cpu(), y.cpu(), acts[:,0].reshape(GS,GS).cpu(), acts[:,1].reshape(GS,GS).cpu(), color='white', scale_units='xy', angles='xy', units='xy', zorder=2)
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
action = torch.randn(batch_sz, action_sz, device=device)
goal = torch.randn(batch_sz, input_sz, device=device)
steps_to_goal = torch.rand(batch_sz, 1, device=device)
state, hidden_state = env_init(batch_sz=batch_sz)
def maybe_reset_goal(input):
    """Changes the unroll's goal, when the previous one either gets reached or doesn't seem to be getting anywhere, to a goal that may be hard to predict (so that we may learn it).

    Returns a tuple of how many goals were reached and how many goals have timed out."""
    global goal, steps_to_goal
    with torch.no_grad():
        dst = pos_only(torch.remainder(input + .3*torch.randn_like(input, device=device), 1.)) # TODO: Cheating. Would like to select a max-misprediction destination; how?
        old_dist = (pos_only(input) - goal).abs().sum(-1, keepdim=True)
        new_dist = act_dist(input, dst)[1]
        reached, out_of_time = old_dist < .01, steps_to_goal < 0
        change = reached | out_of_time

        goal = torch.where(change, dst, goal)
        steps_to_goal = torch.where(change, new_dist + 4, steps_to_goal - 1)
    reached = 1 - (old_dist - .01).clamp(0,1) # Smoother.
    return reached.sum(), out_of_time.float().sum()
def replay(reached_vs_timeout):
    """Replays samples from the buffer.

    Picks many faraway samples, computes all pairwise distances & actions, and makes the shortest paths on the minibatch's dense graph serve as prediction targets. (Search, to reuse solved subtasks.)

    Also incorporates the buffer's actions where they're closer than predicted ([self-imitation](https://arxiv.org/abs/1806.05635)), and learns to reach non-full-state goals."""
    N = replays_per_step
    if len(replay_buffer) < N: return
    samples = random.choices(replay_buffer, k=N) # Python 3.6+
    times = torch.stack([s.time for s in samples], 0) # N × batch_sz × 1
    states = torch.stack([s.state for s in samples], 0) # N × batch_sz × input_sz
    actions = torch.stack([s.action for s in samples], 0) # N × batch_sz × action_sz
    noises = torch.stack([s.noise for s in samples], 0) # N × batch_sz × noise_sz

    # Learn distances & actions.
    dist_loss, action_loss = 0,0
    def expand(x, src_or_dst):
        """N×… to N×N×…. `src_or_dst` is 0|1, representing which `N` is broadcasted."""
        return x.unsqueeze(src_or_dst).expand(N, N, batch_sz, x.shape[-1])
    def dstl(d,D):
        """Prediction-loss that prefers lower-dists: always nonzero, but fades a bit if dist is too high."""
        with torch.no_grad(): mult = (d.detach() - D + 1).clamp(.3,3)
        return (mult * (d - D).square()).sum()
    max_goals = max(len(s.goals) for s in samples)
    for lvl in range(dist_levels):
        for g in range(max_goals):
            # Predict distances & actions, for all src→dst pairs.
            goals = torch.stack([s.goals[g] for s in samples], 0)
            srcs, dsts, noss = expand(states, 0), expand(goals, 1), expand(noises, 0)
            lvl2 = (lvl / dist_levels)*2-1 # -1…1: prev level.
            lvl3 = ((lvl+1) / dist_levels)*2-1 # -1…1: next level.
            # Compute prediction targets.
            with torch.no_grad():
                a,d = act_dist(srcs, dsts, noss, lvl=lvl2, nn=dist_slow) # (Slow for stability.)
                # Incorporate self-imitation knowledge: when a path is shorter than predicted, use it.
                i, j = expand(times, 0), expand(times, 1)
                cond = (i < j) & (j-i < d)
                d = torch.where(cond, j-i, d)
                a = torch.where(cond, expand(actions, 0), a)
                # Use the minibatch fully, by actually computing shortest paths.
                if g == 0: # (Only full states can act as midpoints for pathfinding, since goal-spaces have less info than the full-space.)
                    d,a = floyd(d,a)
                else: # (Goals should take from full-state plans directly.)
                    cond = d0 < d
                    d = torch.where(cond, d0, d)
                    a = torch.where(cond, a0, a)
            if g == 0: d0, a0 = d, a # Preserve state-info for goals.
            a1, d1 = act_dist(srcs, dsts, noss, lvl=lvl3)
            dist_loss = dist_loss + dstl(d1, d)
            action_loss = action_loss + (a1 - a).square().sum()
    # TODO: Run & fix.

    (dist_loss + action_loss).backward()
    optim.step();  optim.zero_grad(True)
    dist_slow.update()

    # Log debugging info.
    log(0, False, pos = pos_histogram)
    log(1, False, reached = to_np(reached_vs_timeout[0]), timeout = to_np(reached_vs_timeout[1]))
    log(2, False, dist_loss = to_np(dist_loss / batch_sz / replays_per_step), action_loss = to_np(action_loss / batch_sz / replays_per_step))



for iter in range(500000):
    with torch.no_grad():
        full_state = cat(state, hidden_state)
        noise = torch.randn(batch_sz, noise_sz, device=device)
        action, _ = act_dist(full_state, goal, noise, nn=dist_slow)
        # if iter % 100 < 50: action = torch.rand(batch_sz, action_sz, device=device)*2-1 # TODO:
        # if iter % 100 < (70 - iter//1000): action = goal[..., :2] - state # TODO: (Literally very much cheating, suggesting trajectories that go toward the goals.) # TODO: (Can actually hold up for 70k epochs, but when this policy is fully gone, it all breaks.)

        # TODO: …How to compute (or approximate) (expectation of) distance misprediction, so that when we sample new goals, we can maximize the regret: store whenever real dist is lower/better than predicted dist?…
        #   Do we want another class, which maintains several timestamped max-metric samples? To add, the new sample is compared with several others that are removed, and the min-metric (max-regret) sample does not get added back; when replaying, the metric has to be updated…

        replay_buffer.append(ReplaySample(
            torch.full((batch_sz, 1), iter, dtype=torch.float32, device=device),
            full_state,
            action,
            (full_state, pos_only(full_state)),
            #   Want to go to places, not caring about final velocity.
            noise,
        ))

        state, hidden_state = env_step(state, hidden_state, action)
    replay(maybe_reset_goal(full_state))
finish()

# TODO: Run & fix.








# …Could also return to embeddings & locally-isometric maps. This would also allow us to learn MuZero-like embeddings, where prev-frame plus action equals next-frame via prediction (but, not clear if it's better than making src-embedding action-dependent, and learn both local min-dist and global min-dist; but in [CLIP](https://arxiv.org/abs/2103.00020), a contrastive objective is 4× more data-efficient than a predictive objective).
#   (…If src-emb is action-dependent, then we can make single-step embeddings *exactly* equal, solving our "but dist-learning isn't *really* a superset of contrastive learning" conundrum… But to actually imagine trajectories, need `(src_emb, act) → dst_emb`, which is an explicit addon, like in MuZero…)
#     (…Not just a gimmick: it would help with both stochasticity and actions that don't change the state.)
#     (…ALSO: this can be used for DDPG without learning a separate action-dependent dist sub-network like we do now with `dist`. …And this DDPG in a locally-isometric space could actually perform really well, since angles are often already correct and so the optima are very wide…)
#   (…With faraway-sample batches, this could bring big benefits: `dist(src,dst)` would need `O(N^2)` NN evaluations, but with embeddings, only `O(N)` embeds are needed to have a rich distance-predicting loss, extracting as much info from a trajectory as possible.)
#   TODO: …So do we return-to-`embed` first, or do the pathfind-framework with `dist` first?
#     TODO: Definitely transition to embeddings, with next-action emb-prediction.



# (…Might even wrap this in a `model/` framework that manages the replay, with arbitrarily-many possibly-async goal states and everything?…)




























# …Augmentating images in computer vision to make NN representations invariant to them, is equivalent to doing that for consecutive RNN steps with body/eye movement (inputs, in any envs). With `embed` dist learning, we're *kinda* doing something similar, making close-in-time images similar and far-in-time images distinct. Though whether this could possibly succeed in learning reprs on CIFAR10 must be tested.






# Best control for `sn` would allow *arbitrary* human data (if limited, then `sn` is hardly an AI-based human-machine interface enabler) to control the goal (if just actions, then human capabilities won't get multiplied, `sn` will just be tiring and weird at best). Max sensitivity to an outcome, but min sensitivity to possible-outcomes: maximize [mutual info](https://en.wikipedia.org/wiki/Mutual_information), AKA channel capacity. (Or [pointwise MI](https://en.wikipedia.org/wiki/Pointwise_mutual_information): `log(p(y|x) / p(y))`.)
#   Without further grounding, we may only need an SSL method, to make the compressed-history the goal: the simple `leads_to(ev(prev))=sg ev(next)` BYOL-on-RNNs, or maybe even our embed-space dist-learning. Needs further research.
#   Do we want a separate channel for human actions, or would mixing them with all other data suffice?
#     (I guess for now, we should refine intent-amplification, and worry about plugging in intent later.)
#     (If we want a separate channel, then damn, I guess those `import env` one-liners aren't gonna fly with `sn` anymore.)
#   …However: making the agent learn to 'control' human data, and simply cutting out chunks of how-humans-want-to-be-controlled via reward, *may* create much richer experiences, without the tedious need to make the agent explore & acquire skills manually.