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
    # force_center = torch.ones(posit.shape[0], 2, device=device)/2
    # force_len = (posit - force_center).square() + 1e-5
    # force = 3e-5 / force_len
    # accel = accel + force * (posit - force_center) / force_len # TODO: Can we learn anything if we disable the attractor?
    veloc = (veloc + accel) * .99
    posit = torch.remainder(posit + veloc, 1.)
    return posit, veloc



class ReplaySample:
    __slots__ = ('time', 'state', 'action', 'as_goal', 'noise')
    def __init__(self, time, state, action, as_goal, noise):
        self.time, self.state, self.action, self.as_goal, self.noise = time, state, action, as_goal, noise
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



batch_size = 100
input_sz, embed_sz, action_sz, noise_sz = 4, 64, 2, 8
lr = 1e-3

replay_buffer = ReplayBuffer(max_len=64) # of ReplaySample
replays_per_unroll = 4



def net(ins, outs, hidden=embed_sz):
    return nn.Sequential(
        SkipConnection(nn.Linear(ins, hidden)),
        SkipConnection(nn.ReLU(), nn.LayerNorm(hidden), nn.Linear(hidden, hidden)),
        SkipConnection(nn.ReLU(), nn.LayerNorm(hidden), nn.Linear(hidden, hidden)),
        SkipConnection(nn.ReLU(), nn.LayerNorm(hidden), nn.Linear(hidden, hidden)),
        SkipConnection(nn.ReLU(), nn.LayerNorm(hidden), nn.Linear(hidden, hidden)),
        nn.ReLU(), nn.LayerNorm(hidden), nn.Linear(hidden, outs),
    ).to(device)
dist = net(action_sz + input_sz + input_sz + noise_sz + 1, action_sz + 1)
#   (0|action, src, dst, lvl) → (min_action, min_dist)
#   (Learns min-dist spanning trees that go to a destination, and that min-dist for tree-selection.)
#   (Usable for both gradient-ascent and self-imitation, and possibly good-embeddings-learning.)
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



def act_dist(action, src, dst, noise=None, nn=dist, lvl=1.):
    """Returns a tuple `(action, dist)`, having refined the action (possibly `None`) & source & destination."""
    if action is None:
        action = torch.zeros(*src.shape[:-1], action_sz, device=device)
    if noise is None:
        noise = torch.randn(*src.shape[:-1], noise_sz, device=device)
    lvl = torch.full([*src.shape[:-1], 1], lvl, device=device)
    ad = nn(cat(action, 2*src-1, 2*dst-1, noise, lvl))
    return ad[..., :-1], 2**ad[..., -1:]
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
        x = x.reshape(1,GS,1).expand(GS,GS,1).reshape(GS*GS,1)
        y = y.reshape(GS,1,1).expand(GS,GS,1).reshape(GS*GS,1)
        veloc = torch.zeros(GS*GS, 2, device=device)
        src = cat(x, y, veloc)
        acts, dists = act_dist(None, src, dst)
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
action = torch.randn(batch_size, action_sz, device=device)
goal = torch.randn(batch_size, input_sz, device=device)
steps_to_goal = torch.rand(batch_size, 1, device=device)
state, hidden_state = env_init(batch_size=batch_size)
def maybe_reset_goal(input):
    """Changes the unroll's goal, when the previous one either gets reached or doesn't seem to be getting anywhere, to a goal that may be hard to predict (so that we may learn it).

    Returns a tuple of how many goals were reached and how many goals have timed out."""
    global goal, steps_to_goal
    with torch.no_grad():
        dst = as_goal(torch.remainder(input + .3*torch.randn_like(input, device=device), 1.)) # TODO: Cheating. Would like to select a max-misprediction destination; how?
        old_dist = (as_goal(input) - goal).abs().sum(-1, keepdim=True)
        new_dist = act_dist(None, input, dst)[1]
        reached, out_of_time = old_dist < .01, steps_to_goal < 0
        change = reached | out_of_time

        goal = torch.where(change, dst, goal)
        steps_to_goal = torch.where(change, dist_to_steps(new_dist) + 4, steps_to_goal - 1)
    reached = 1 - (old_dist - .01).clamp(0,1) # Smoother.
    return reached.sum(), out_of_time.float().sum()
def replay(reached_vs_timeout):
    """Replays samples from the buffer.

    Picks samples at i<j, and refines min-distances (both global and action-dependent) and min-distance actions."""
    L = len(replay_buffer)
    if L < 2: return
    dist_loss, action_loss, ddpg_loss = 0,0,0
    for _ in range(replays_per_unroll): # Look, concatenation and variable-management are hard.
        # Variables.
        a,b = replay_buffer[random.randint(0, L-1)], replay_buffer[random.randint(0, L-1)]
        i,j = a.time, b.time
        if i>j: i,j,a,b = j,i,b,a
        if i==j: return
        aag, dag = act_dist(None, a.state, b.as_goal, a.noise)

        # Learn faraway distances to goal-states, using timestamp-differences.
        def dstl(d,D):
            """Always nonzero, but fades if dist is too high; prefers lower dists."""
            mult = (dist_to_steps(d.detach()) - D + 1).clamp(.3,3) # (…A non-0 lower bound is required for learning, but having that biases the algo… May want to instead construct a sequence of levels, each more min-dist than the last…)
            return (mult * (d - steps_to_dist(D)).square()).sum()
        dag1 = act_dist(    None, a.state, b.as_goal, lvl=-1)[1]
        Dag1 = act_dist(a.action, a.state, b.as_goal, lvl=-1)[1]
        dag2 = act_dist(    None, a.state, b.as_goal)[1] # lvl=1
        Dag2 = act_dist(a.action, a.state, b.as_goal)[1] # lvl=1
        dist_loss = dist_loss + dstl(dag1, j-i) # Global-min dist.
        dist_loss = dist_loss + dstl(Dag1, j-i) # Local-min dist.
        dist_loss = dist_loss + dstl(dag2, dag1.detach()) # Global-min dist.
        dist_loss = dist_loss + dstl(Dag2, Dag1.detach()) # Local-min dist.
        # (We have 2 levels, where each filters the prediction-targets to be lower.)
        #   (So the more levels we have, the more robust our action-learning will be to non-optimal policies.)

        # TODO: …Maybe, re-introduce triplets, not for dist-learning, but as the simplest primitive for combining subtasks?… (…Would need to learn everything with non-goal destinations too…)

        # Self-imitation: make globally-min-dist actions predict the min-dist actions from the replay.
        with torch.no_grad():
            dag = act_dist(    None, a.state, b.as_goal, nn=dist_slow)[1]
            Dag = act_dist(a.action, a.state, b.as_goal, nn=dist_slow)[1].clamp(None, j-i)
        action_loss = action_loss + (aag - torch.where(dag < Dag, aag, a.action)).square().sum()

        # DDPG: a learned loss for globally-min actions.
        ddpg_loss = ddpg_loss + act_dist(aag, a.state, b.as_goal, nn=dist_slow)[1].sum()

    (dist_loss + action_loss + ddpg_loss).backward()
    optim.step();  optim.zero_grad(True)
    dist_slow.update()

    # Log debugging info.
    log(0, False, pos = pos_histogram)
    log(1, False, reached = to_np(reached_vs_timeout[0]), timeout = to_np(reached_vs_timeout[1]))
    log(2, False, dist_loss = to_np(dist_loss / batch_size / replays_per_unroll), action_loss = to_np(action_loss / batch_size / replays_per_unroll), ddpg_loss = to_np(ddpg_loss / batch_size / replays_per_unroll))



no_act = torch.zeros(batch_size, action_sz, device=device)
for iter in range(500000):
    with torch.no_grad():
        full_state = cat(state, hidden_state)
        noise = torch.randn(batch_size, noise_sz, device=device)
        action, _ = act_dist(no_act, full_state, goal, noise, nn=dist_slow)
        # if iter % 100 < 50: action = torch.rand(batch_size, action_sz, device=device)*2-1 # TODO:
        # if iter % 100 < (70 - iter//1000): action = goal[..., :2] - state # TODO: (Literally very much cheating, suggesting trajectories that go toward the goals.) # TODO: (Can actually hold up for 70k epochs, but when this policy is fully gone, it all breaks.)

        # TODO: …How to compute (or approximate) (expectation of) distance misprediction, so that when we sample new goals, we can maximize the regret: store whenever real dist is lower/better than predicted dist?…
        #   Do we want another class, which maintains several timestamped max-metric samples? To add, the new sample is compared with several others that are removed, and the min-metric (max-regret) sample does not get added back; when replaying, the metric has to be updated…

        replay_buffer.append(ReplaySample(
            iter,
            full_state,
            action,
            as_goal(full_state),
            #   Want to go to places, not caring about final velocity.
            noise,
        ))

        state, hidden_state = env_step(state, hidden_state, action)
    replay(maybe_reset_goal(full_state))
finish()

# TODO: Run & fix.








# …Could also return to embeddings & locally-isometric maps. This would also allow us to learn MuZero-like embeddings, where prev-frame plus action equals next-frame via prediction (but, not clear if it's better than making src-embedding action-dependent, and learn both local min-dist and global min-dist; but in [CLIP](https://arxiv.org/abs/2103.00020), a contrastive objective is 4× more data-efficient than a predictive objective).
#   (…If src-emb is action-dependent, then we can make single-step embeddings *exactly* equal, solving our "but dist-learning isn't *really* a superset of contrastive learning" conundrum… But to actually imagine trajectories, need `(src_emb, act) → dst_emb`, which is extra…)



# …Sure, we *can* sample i→j→k triplets, and do all the symbolic-search-tricks of a<b&b<c⇒a<c (pred-targets being the min-dist of everything predicted (dist from `act_dist`) & replayed (index-diff), or making j→k use a real/predicted i→j action, etc). But, just faraway-sampling of i→j seems to be enough in this cont-env.
#   …Except we don't actually converge without a helping hand.

# …Maybe we can actually come up with a framework for sampling any-size faraway trajectory samples, where the further samples use the min-dist subtask of the whole trajectory (for subtask-combining, have to consider *learned* dists & actions), solving a dynamic-programming problem (a bit like learning-time search)… Would it be quadratic-time, or linear-time? What would the problem look like, exactly? We have src & dst, so I'm pretty sure it would be quadratic…
#   Each intermediate dist/action should be replaced by its pred-target for those after it. This would create a linear-time greedy algorithm. But this wouldn't consider combinations, huh…
#   min[src, mid, dst] — cubic? What are the min-equations?
#     min[src, src, dst] = dist(src, dst)
#     min[src, src+1, dst] = dist(src, dst)
#     min[src, mid+1, dst] = min(min[src, mid, dst], min[src, mid, mid] + min[mid, dst, dst])
#     min[src, dst, dst] is the sought-after answer.
#   …So is it possible to simplify these, and *not* go cubic? …Don't really think so…
#     …But we *can* just shuffle midpoints randomly, and only do one step of that minimization.
#       Okay, this all suddenly sounds like a very good idea: the ability to make `replays_per_unroll` not just a slowdown-hyperparam but easily make it do good stuff for us.
#       (Can probably only do this dist-min for actions; dist will probably catch up by seeing what the actions can do, or will correct them if wrongly low.)
#       (…This is pretty much learning-time step-skipping planning, isn't it?)
#   …Can we write this down on a sampled batch, which is not necessarily in any order, but for which we do have timestamps (and so can just discard prediction targets for i>=j)? (Like [lifted structural loss](https://lilianweng.github.io/posts/2021-05-31-contrastive/).)
#     TODO: Write dist-minimization on a minibatch down.
#       (I mean, we basically need a matrix of predicted-distances (`dist_slow` computes it), and compute a matrix of predicted-actions, and refine it with itself by comparing each entry's dist with a sum of through-midpoint dist; the target is predicted-act (replay-acts may be for bottoming-out only) where midpoint gives lower pred-dist, else the replay-act if time-diff is less than pred-dist-diff else no-change. …Too indistinct to write down…)



# (…Might even wrap this in a `model/` framework that manages the replay, with arbitrarily-many possibly-async goal states and everything?…)




























# …Augmentating images in computer vision to make NN representations invariant to them, is equivalent to doing that for consecutive RNN steps with body/eye movement (inputs, in any envs). With `embed` dist learning, we're *kinda* doing something similar, making close-in-time images similar and far-in-time images distinct. Though whether this could possibly succeed in learning reprs on CIFAR10 must be tested.






# Best control for `sn` would allow *arbitrary* human data (if limited, then `sn` is hardly an AI-based human-machine interface enabler) to control the goal (if just actions, then human capabilities won't get multiplied, `sn` will just be tiring and weird at best). Max sensitivity to an outcome, but min sensitivity to possible-outcomes: maximize [mutual info](https://en.wikipedia.org/wiki/Mutual_information), AKA channel capacity. (Or [pointwise MI](https://en.wikipedia.org/wiki/Pointwise_mutual_information): `log(p(y|x) / p(y))`.)
#   Without further grounding, we may only need an SSL method, to make the compressed-history the goal: the simple `leads_to(ev(prev))=sg ev(next)` BYOL-on-RNNs, or maybe even our embed-space dist-learning. Needs further research.
#   Do we want a separate channel for human actions, or would mixing them with all other data suffice?
#     (I guess for now, we should refine intent-amplification, and worry about plugging in intent later.)
#     (If we want a separate channel, then damn, I guess those `import env` one-liners aren't gonna fly with `sn` anymore.)
#   …However: making the agent learn to 'control' human data, and simply cutting out chunks of how-humans-want-to-be-controlled via reward, *may* create much richer experiences, without the tedious need to make the agent explore & acquire skills manually.