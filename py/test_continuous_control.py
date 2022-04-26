"""
Can RL be done not via approximating the loss (discounted sum of future distances here, AKA return), but via BPTT? Mathematically, sure, but what are the obstacles to doing it practically?

This 2D env has 1 spaceship and a repulsor in the middle; the actions control acceleration. All smooth, so grad-min should be able to find best actions. Effective exploration would be able to cover a lot of the torus.

---

We implement an RNN that minimizes the distance between goal-conditioned paths and goals, sampling goals from the recent past, predicting the next-frame, and minimizing its distance to the goal.
"""



import torch
import torch.nn as nn
device = 'cuda' if torch.cuda.is_available() else 'cpu'



from model.log import log, clear

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
    __slots__ = ('uncertainty', 'state', 'action', 'as_goal')
    def __init__(self, uncertainty, state, action, as_goal):
        self.uncertainty, self.state, self.action, self.as_goal = uncertainty, state, action, as_goal
    def combine(self, o):
        """Pick the max-uncertainty replay sample."""
        if o.uncertainty is None: return self
        if self.uncertainty is None: return o
        c = self.uncertainty > o.uncertainty
        return ReplaySample(
            torch.where(c, self.uncertainty, o.uncertainty),
            torch.where(c, self.state, o.state),
            torch.where(c, self.action, o.action),
            torch.where(c, self.as_goal, o.as_goal),
        )
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

replay_buffer = ReplayBuffer(max_len=1024) # of ReplaySample
replays_per_unroll = 1



def net(ins, outs, hidden=action_sz):
    return nn.Sequential(
        SkipConnection(nn.Linear(ins, hidden)),
        SkipConnection(nn.ReLU(), nn.LayerNorm(hidden), nn.Linear(hidden, hidden)),
        nn.ReLU(), nn.LayerNorm(hidden), nn.Linear(hidden, outs),
    ).to(device)
embed = [net(1+input_sz, embed_sz), net(1+input_sz, embed_sz)] # (src_or_dst: -1|1, input) → emb
#   (Locally-isometric maps of the environment.)
#   (Temporally-close events are close here, far events are far.)
#   (We have an ensemble of 2, so that we can estimate uncertainty.)
act = net(embed_sz + embed_sz, action_sz) # (src, dst) → action
#   (Min-dist spanning trees that go to a destination.)

optim = torch.optim.Adam([*embed[0].parameters(), *embed[1].parameters(), *act.parameters()], lr=lr)



def embed_(src_or_dst, input, which_from_ensemble=0):
    """Convenience: `embed`s into a locally-isometric map for easy generalization and dist-estimation."""
    x = cat(torch.ones(input.shape[0], 1, device=device) * (1 if src_or_dst else -1), input)
    return embed[which_from_ensemble](x)
def dist_(src, dst):
    """Convenience: dist between `embed`dings, or rather, `1+log2(steps)`."""
    return (src - dst).square().mean(-1, keepdim=True)
def dist_to_steps(dist): return 2 ** (dist-1)



# For debugging.
def pos_histogram(plt, label):
    """That replay buffer contains lots of past positions. This func plots those as a 2D histogram, in addition to arrows indicating which action to take to get from anywhere to a random position."""
    with torch.no_grad():
        x, y = [], []
        for ch in replay_buffer:
            x.append(ch.state[..., 0]), y.append(ch.state[..., 1])
        plt.hist2d(torch.cat(x).cpu().numpy(), torch.cat(y).cpu().numpy(), bins=(100,100), range=((0,1), (0,1)), cmap='nipy_spectral', label=label)

        # Display action-arrows everywhere.
        GS = 16 # grid size
        dst = embed_(1, cat(torch.rand(GS*GS, 2, device=device), torch.ones(GS*GS, 2, device=device)))
        x, y = torch.linspace(0.,1.,GS, device=device), torch.linspace(0.,1.,GS, device=device)
        x = x.reshape(GS,1,1).expand(GS,GS,1).reshape(GS*GS,1)
        y = y.reshape(1,GS,1).expand(GS,GS,1).reshape(GS*GS,1)
        veloc = torch.zeros(GS*GS, 2, device=device)
        src = embed_(0, cat(x, y, veloc))
        acts = act(cat(src, dst))
        plt.quiver(x.cpu(), y.cpu(), acts[:,0].reshape(GS,GS).cpu(), acts[:,1].reshape(GS,GS).cpu(), color='white', scale_units='xy', angles='xy', units='xy')



# The main loop, which steps the environment and trains `step`.
action = torch.randn(batch_size, action_sz, device=device)
goal = torch.randn(batch_size, embed_sz, device=device)
steps_to_goal = torch.rand(batch_size, 1, device=device)
state, hidden_state = env_init(batch_size=batch_size)
def maybe_reset_goal(input):
    """Changes the unroll's goal, when the previous one either gets reached or doesn't seem to be getting anywhere, to a goal that may be hard to predict (so that we may learn it).

    Returns a tuple of how many goals were reached and how many goals have timed out."""
    global goal, steps_to_goal
    with torch.no_grad():
        src = embed_(0, input)
        dst = embed_(1, replay_buffer.sample_best().state) # Not ever choosing `.as_goal` for simplicity.
        old_dist, new_dist = dist_(src, goal), dist_(src, dst)
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
        i = random.randint(0, L-4)
        I = i + 1
        j = random.randint(I+1, L-2)
        k = random.randint(j+1, L-1)
        a,A,b,c = replay_buffer[i], replay_buffer[I], replay_buffer[j], replay_buffer[k]

        # Source/destination embeddings.
        e = embed_
        sa, sb = e(0, a.state, 0), e(0, b.state, 0)
        Sa, Sb = e(0, a.state, 1), e(0, b.state, 1)
        dA, db, dc, dg = e(1, A.state, 0), e(1, b.state, 0), e(1, c.state, 0), e(1, c.as_goal, 0)
        DA, Db, Dc, Dg = e(1, A.state, 1), e(1, b.state, 1), e(1, c.state, 1), e(1, c.as_goal, 1)

        # Distances.
        daA, dab, dac, dbc, dbg = dist_(sa, dA), dist_(sa, db), dist_(sa, dc), dist_(sb, dc), dist_(sb, dg)
        DaA, Dab, Dac, Dbc, Dbg = dist_(Sa, DA), dist_(Sa, Db), dist_(Sa, Dc), dist_(Sb, Dc), dist_(Sb, Dg)

        # Learn distance, to be the min of seen 1+log2 of steps (index-differences).
        def dstl(d,D):
            """Always nonzero, but fades if dist is too high; prefers lower dists."""
            d = dist_to_steps(d)
            mult = (d.detach() - D) + 1
            mult = torch.where(mult>0, mult+1, mult.exp()).clamp(0,15)
            mult = mult if D != 1 else 1.
            return (mult * (d - D).square()).sum()
        dist_loss = dist_loss + dstl(daA, I-i)
        dist_loss = dist_loss + dstl(DaA, I-i)
        dist_loss = dist_loss + dstl(dab, j-i) + dstl(dac, k-i)
        dist_loss = dist_loss + dstl(Dab, j-i) + dstl(Dac, k-i)
        dist_loss = dist_loss + dstl(dbc, k-j) + dstl(dbg, k-j)
        dist_loss = dist_loss + dstl(Dbc, k-j) + dstl(Dbg, k-j)

        # Learn ground-actions.
        ground_loss = ground_loss + (act(cat(sa, dA)) - a.action).square().sum()

        # Learn meta-actions to k, to be the dist-min of actions to j.
        def actl(d,D, a,A):
            """Tries to cut off anything not-min-dist, if in lin-space."""
            d = dist_to_steps(d)
            mult = (d.detach() - D + 1).clamp(0,15)
            mult = torch.where( D>1.5, mult, torch.tensor(1., device=device) )
            return (mult * (a - A).square()).sum()
        act_target = act(cat(sa, db)).detach() # TODO: Is this subtask-combining loss better?…
        # act_target = a.action
        # meta_loss = meta_loss + actl(dac, torch.full_like(dac, k-i), act(cat(sa, dc)), act_target) # TODO: …Hasn't helped, so far… Maybe because we used to not even learn the distance…
        #   …Wouldn't it kinda make sense to learn their-dist-is-better-than-ours acts (`act(a→c) = a.action`), AND learn our-dist-is-better-than-theirs acts (`act(A→c) = act(a→b).detach()`)?…
        meta_loss = meta_loss + actl(dac, (dab+dbc).detach(), act(cat(sa, dc)), act_target) # TODO:

        # Learn meta-actions to goal-of-k.
        meta_loss = meta_loss + actl(dbc, dbc.detach(), act(cat(sb, dg)), act(cat(db, dc)).detach())
        meta_loss = meta_loss + actl(dac, (dab+dbc).detach(), act(cat(sa, dg)), act_target)

        # Set/update the uncertainty of dist-prediction: overwrite if `None`, average otherwise.
        ua = (daA - DaA).abs().sum(-1, keepdim=True) + (dab - Dab).abs().sum(-1, keepdim=True) + (dac - Dac).abs().sum(-1, keepdim=True)
        uc = (dac - Dac).abs().sum(-1, keepdim=True) + (dbc - Dbc).abs().sum(-1, keepdim=True) + (dbg - Dbg).abs().sum(-1, keepdim=True)
        a.uncertainty = ua if a.uncertainty is None else (a.uncertainty + ua)/2
        c.uncertainty = uc if c.uncertainty is None else (c.uncertainty + uc)/2

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
        action = act(cat(embed_(0, full_state), goal))
        # if iter % 100 < 50: action = action + torch.randn(batch_size, action_sz, device=device)*.2 # TODO:

        as_goal = cat(full_state[..., :2], torch.ones(batch_size, 2, device=device)) # TODO:
        # print(full_state.shape, action.shape, as_goal.shape) # TODO: 100×4, 100×64, 100×4 — NOT 1GB MATERIAL, MORE LIKE 30MB, IT MAKES NO SENSE; WHY DO WE NEED SO MUCH GPU MEMORY?
        #   …Wait, why is it no longer taking any GPU memory, even though nothing changed?
        replay_buffer.append(ReplaySample(
            None,
            full_state,
            action,
            as_goal, # Want to go to places, not caring about final velocity.
        ))
        if random.randint(1,100)==1: state, hidden_state = env_init(batch_size=batch_size) # TODO: Resetting doesn't help…
    replay(maybe_reset_goal(full_state))

# TODO: Run & fix.
#   TODO: …Why do all actions end up collapsing to the same action? And why do we end up in the exact same 4 bins on the histogram?
#     Maybe our goal-sampling is very wrong?…
#     TODO: Do we need to inject action-noise after all? …How?…
#       (Making 50% of actions random doesn't seem to be working out.)
#   TODO: Why doesn't the distance loss go down below like .3 at minimum, or 1 on average? And why does it eventually temporarily-explode to ever greater values, such as 80k at 25k epochs or 1M at 26k epochs?
#     (Worst-case, our dist-metric is very inapplicable to continuous spaces…)
#     TODO: Try training a real dist neural net. Does loss go lower than what we have now?
#     TODO: Try both linspace and logspace dists.
#   TODO: …Why does reachability percentage go down over time, from 4% to .5% over 25k epochs?…
#     TODO: …Do we want to compute & log that NASWOT metric after all, since our few 0…1 inputs are likely to be poorly separated initially?…
#     TODO: …Do we want to always use real actions in meta-action-loss, counting on poor plans getting filtered out?… Hasn't improved anything so far…
#   TODO: …Wouldn't it kinda make sense to learn their-dist-is-better-than-ours acts (`act(a→c) = a.action`), AND learn our-dist-is-better-than-theirs acts (`act(a→c) = act(a→b).detach()`)?…
#     (Action-target should be whichever of replayed/predicted actions is the shortest, index-diff vs sum-of-subdists. The dist-to-weigh-with should be the min of these.)
#     (Don't need to think if the suggested plan is just as good as our own is.)
#     TODO: …Should we try this in the board env first?…
#   TODO: …What component can we isolate to ensure that it's working right?…
#     Distances, right? If not this, then only actions exist, right?
#     TODO: Maybe, also print unroll-time the dist-misprediction from the state at previous goal-setting to the present, since we know how many steps it's supposed to take? (Since the dist loss doesn't look like it improves at all, over 20k epochs.)
#     TODO: Also log distances to a random target. (A clear tool for telling whether our distance-learning is failing entirely.)







# …Could also, instead of maximizing uncertainty (which is 2× slower than pure dist estimation), maximize regret (underestimation of dist, computed at unroll-time from encountered dst-embeddings) by goals.




























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