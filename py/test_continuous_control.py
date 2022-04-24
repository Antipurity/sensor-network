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



class ReplayBuffer:
    """Stores the in-order sequence of most-recent events. Needs `max_len=1024`. Supports `len(rb)`, `rb.append(data)`, `rb[index]`, `rb.sample_best()`."""
    def __init__(self, max_len=1024):
        self.head, self.max_len = 0, max_len
        self.buffer = []
    def __len__(self):
        """How many data-samples are currently contained. Use this to decide the indices to sample at."""
        return len(self.buffer)
    def append(self, data):
        """Appends a data-sample to the replay buffer. Typically an array, such as `[ranking, state, action, as_goal]`.

        Save unrolls."""
        if len(self.buffer) == self.max_len:
            self.buffer[self.head] = data
        else:
            self.buffer.append(data)
        self.head = (self.head + 1) % len(self.max_len)
    def __getitem__(self, index):
        """Returns the sample at the given index. For example, use `random.randrange(len(replay_buffer))`.

        Consecutive indices are guaranteed to be a part of a contiguous sequence.

        Remember unrolls, either with independent samples, consecutive pairs, or ordered faraway sequences."""
        assert isinstance(index, int)
        return self.buffer[(self.head + index) % len(self)]
    def __iter__(self):
        for i in range(len(self)): yield self[i]
    def sample_best(self, samples=16, combine=lambda a,b:list(torch.where(a[0]>b[0], x, y) for x,y in zip(a,b))):
        """A primitive algorithm for sampling likely-best data, ranked by the first array item.

        For picking at-unroll goals.

        Args:
        - `samples=16`: how many samples to choose among. We don't have a precomputed data structure.
        - `combine=...`: the actual chooser. By default, compares by the 0th item, and returns a list of max values."""
        N, result = len(self), None
        for _ in range(samples):
            sample = self.sample_data(random.randrange(N))
            result = sample if result is None else combine(result, sample)
            # (This creates an O(N) chain of computations, though O(log(N)) is possible.)
        return result



class SkipConnection(nn.Module):
    """Linearize gradients, to make learning easier."""
    def __init__(self, *fn): super().__init__();  self.fn = nn.Sequential(*fn)
    def forward(self, x):
        y = self.fn(x)
        return x + y if x.shape == y.shape else x[..., :y.shape[-1]] + y
def to_np(x): return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
def cat(*a, dim=-1): return torch.cat(a, dim)



batch_size = 128
input_sz, embed_sz, action_sz = 4, 128, 128
lr = 1e-3

replay_buffer = ReplayBuffer(max_len=1024) # [ranking, input, action, as_goal]
replays_per_unroll = 4



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



# Debugging.
def pos_histogram(plt, label):
    """That replay buffer contains lots of past positions. This func plots those as a 2D histogram."""
    x, y = [], []
    for ch in replay_buffer:
        if ch is not None:
            ranking, state, action, as_goal = ch
            pos = to_np(state)
            x.append(pos[..., 0]), y.append(pos[..., 1])
    import numpy as np
    plt.hist2d(np.concatenate(x), np.concatenate(y), bins=100, range=((0,1), (0,1)), cmap='nipy_spectral', label=label)
    # TODO: `log`: pick a destination randomly (in `as_goal` space), and visualize actions-to-it as arrows. (Assuming that we succeed in learning all-paths, at least a little.)



# The main loop, which steps the environment and trains `step`.
action = torch.randn(batch_size, action_sz, device=device)
goal = torch.randn(batch_size, input_sz, device=device)
# TODO: Also need steps-until-new-goal.
state, hidden_state = env_init(batch_size=batch_size)
def maybe_reset_goal():
    global goal
    with torch.no_grad():
        # TODO: Sometimes (`torch.where`): max-sample a new goal from the replay-buffer (and compute the new estimated-steps-to-reach-it), if reached (abs of goal-state minus goal is less than 1e-3) or out-of-time.
        #   How do we estimate the distance, exactly?…
        pass
    # TODO: Return a tuple of how many times we've reached a goal, and how many times the goal timed out.
def replay(reached_vs_timeout):
    # Replay from the buffer. (Needs Python 3.6+ for convenience.)
    choices = [c for c in random.choices(replay_buffer, k=replays_per_unroll) if c is not None]
    #   TODO: Use our own indices, not completely random: i, i+1, i<j, j<k.
    if len(choices):
        # TODO: Concat each item, properly.
        ranking = torch.cat([c[0] for c in choices], 0)
        state = torch.cat([c[1] for c in choices], 0)
        action = torch.cat([c[2] for c in choices], 0)
        as_goal = torch.cat([c[3] for c in choices], 0)
        N = state.shape[0]

        goal = state[torch.randperm(N, device=device)] # TODO: Needs to be a real future-state now.
        randn = torch.randn(N, action_sz, device=device)

        # TODO: At replay:
        #   TODO: As many times as required (1 by default) (turn to tensors & concatenate):
        #     TODO: Sample 4 indices: i, i+1, i<j, j<k.
        #     TODO: Sample items at those indices.
        #   TODO: Update distances, to be the min of seen log2 of index-differences.
        #     TODO: Also update distances to func-of-destination.
        #     TODO: Set/update the uncertainty (0th item of items), to be the abs-diff of the 2 versions of `embed`-dists: overwrite if `None`, average otherwise.
        #   TODO: Update ground-actions, act(i,i+1)=i.action.
        #   TODO: Update meta-actions to k, to be the dist-min of actions to j.
        #     TODO: Also update meta-actions that point not just to state but also to the `as_goal` func-of-state. (Func-of-goal has no grounding, but it *can* be learned to go to.)
        #   TODO: Perform the gradient-descent update.

        dist_loss = 0
        ground_loss = 0
        meta_loss = 0

        # (dist_loss + ground_loss + meta_loss).backward()
        optim.step();  optim.zero_grad(True)

        # Log debugging info.
        log(0, False, pos = pos_histogram)
        log(1, False, reached = to_np(reached_vs_timeout[0]), timeout = to_np(reached_vs_timeout[1]))
        log(2, False, dist_loss = to_np(dist_loss / N))
        log(3, False, ground_loss = to_np(ground_loss / N))
        log(4, False, meta_loss = to_np(meta_loss / N))



for iter in range(500000):
    with torch.no_grad():
        state, hidden_state = env_step(state, hidden_state, action)
        all_state = cat(state, hidden_state)
        action = act(cat(all_state, goal))

        replay_buffer.append([
            torch.zeros(batch_size, 1, device=device),
            all_state,
            action,
            cat(all_state[:2], torch.ones(batch_size, 2, device=device)), # Wanna go places, not caring about final velocity.
        ])
    replay(maybe_reset_goal())




























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



# TODO: Gotta get back, back to the past:
#   TODO: In `test.py`, implement self-targeting RL (with dist-bootstrapping and `act`-dist-min and self-imitation) and self-targeting BPTT (with `act`-dist-min and a skip connection), and try to not just explore one graph but *learn* to explore `minienv`'s graphs. (I don't think any RL exploration method can *learn* to explore, only explore. So if it works, it's cool.)