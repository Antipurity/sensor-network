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



class SkipConnection(nn.Module):
    def __init__(self, *fn): super().__init__();  self.fn = nn.Sequential(*fn)
    def forward(self, x):
        y = self.fn(x)
        return y + x[..., :y.shape[-1]]
def to_np(x): return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
def cat(*a, dim=-1): return torch.cat(a, dim)



batch_size = 100
input_sz, action_sz = 2, 128
lr = 1e-3

bootstrap_discount = .99 # `future_dist` will have to predict numbers from 0 to 1/(1-this).

replay_buffer = [None] * (1024)
replays_per_step = 8



act = nn.Sequential( # (prev_action, input, goal) → action # TODO: Also the `randn` arg (since we don't have `embed` anymore).
    # (`goal` is sampled from the recent past: `replay_buffer`. It's what we want trajectories to minimize the distance to, to gain competency.)
    SkipConnection(nn.Linear(action_sz + input_sz + input_sz, action_sz)),
    *[SkipConnection(
        nn.ReLU(), nn.LayerNorm(action_sz), nn.Linear(action_sz, action_sz),
    ) for _ in range(1)],
    nn.LayerNorm(action_sz),
).to(device)
future_dist = nn.Sequential( # (action, state, goal) → dist
    # TODO: No distance-learning. Only embedding-distance.
    # (Returns the sum-of-future-L1-distances-to-`goal` for the considered `action`, the less the better.)
    SkipConnection(nn.Linear(action_sz + input_sz + input_sz, action_sz)),
    *[SkipConnection(
        nn.ReLU(), nn.LayerNorm(action_sz), nn.Linear(action_sz, action_sz),
    ) for _ in range(1)],
    nn.Linear(action_sz, 1),
).to(device)

optim = torch.optim.Adam([*act.parameters(), *future_dist.parameters()], lr=lr)



# The main loop, which steps the environment and trains `step`.
action = torch.randn(batch_size, action_sz, device=device)
goal = torch.rand(batch_size, input_sz, device=device)
state, hidden_state = env_init(batch_size=batch_size)
def reset_goal():
    global goal
    with torch.no_grad():
        ch = random.choice(replay_buffer)
        if ch is not None:
            prev_action, prev_state, action, state = ch
            randn = torch.randn(batch_size, action_sz, device=device)
            # goal = state # TODO:
            goal = torch.rand(batch_size, input_sz, device=device)
def pos_histogram(plt, label):
    """That replay buffer contains lots of past positions. This func plots those as a 2D histogram."""
    x, y = [], []
    for ch in replay_buffer:
        if ch is not None:
            prev_action, prev_state, action, state = ch
            pos = to_np(state)
            for i in range(pos.shape[0]):
                x.append(float(pos[i][0])), y.append(float(pos[i][1]))
    plt.hist2d(x, y, bins=100, range=((0,1), (0,1)), cmap='rainbow', label=label)
def replay():
    # Replay from the buffer. (Needs Python 3.6+ for convenience.)
    choices = [c for c in random.choices(replay_buffer, k=replays_per_step) if c is not None]
    if len(choices):
        prev_action = torch.cat([c[0] for c in choices], 0)
        prev_state = torch.cat([c[1] for c in choices], 0)
        action = torch.cat([c[2] for c in choices], 0)
        state = torch.cat([c[3] for c in choices], 0)

        goal = state[torch.randperm(state.shape[0], device=device)]
        randn = torch.randn(state.shape[0], action_sz, device=device)

        # If you wanted to go to `state`, well, success.
        zero_dist_loss = future_dist(cat(prev_action, prev_state, state)).square().sum()
        # TODO: …Do bad things happen because this loss interferes with the loss right below?…
        #   But how would we not?

        # If you wanted to go somewhere else, well, that's 1 more step, assuming that we do go there.
        #   TODO: Slow to propagate. Errors compounding. Must be inferior. Improve.
        action2 = act(cat(prev_action, state, goal))
        prev_dist = future_dist(cat(prev_action, prev_state, goal))
        with torch.no_grad():
            dist = future_dist(cat(action, state, goal))
            dist2 = future_dist(cat(action2, state, goal))
            target_dist = torch.where(dist < dist2, dist, dist2) # Min distance.
            #   (We don't predict `state2` from `action2` here, so this is not perfect.)
            #     TODO: IMPERFECTION; how could we solve it?
            #       …Make `future_dist` take not the current but the previous state?… But isn't its input-action intended as a state-descriptor?…
            #       …Remove the `state` arg entirely, and rely on gradient to make `act` predictive of the next state?…
            #       …Can't see another option here…
            target_dist = 1. + dist * bootstrap_discount
        dist_loss = (prev_dist - target_dist).square().sum()

        # Self-imitation learning: if the past was better than us, copy the past.
        #   (Gradient descent on `dist` by `action2` doesn't seem to be good enough for optimization on trajectories, especially since consecutive steps can easily belong to different-goal trajectories.)
        with torch.no_grad():
            target_action = torch.where(dist < dist2, action, action2) # Min distance.
        self_imitation_loss = (action2 - target_action).square().sum()

        (zero_dist_loss + dist_loss + self_imitation_loss).backward()
        optim.step();  optim.zero_grad(True)

        # Log debugging info.
        N = state.shape[0]
        log(0, False, pos = pos_histogram)
        log(1, False, zero_dist_loss = to_np(zero_dist_loss / N))
        log(2, False, dist_loss = to_np(dist_loss / N))
        log(3, False, self_imitation_loss = to_np(self_imitation_loss / N))



for iter in range(500000):
    prev_action, prev_state = action, state # TODO: No need for these.
    with torch.no_grad():
        state, hidden_state = env_step(state, hidden_state, prev_action)
        all_state = cat(state, hidden_state) # TODO: Use *this* everywhere, to not have to learn through time, for now. (Does it even make sense to learn actions through time? Or does that work out to self-compression?)
        action = act(cat(prev_action, state, goal))

    replay()
    if iter == 1000: clear()

    if random.randint(1, 32) == 1: reset_goal()

    replay_buffer[iter % len(replay_buffer)] = (prev_action.detach(), prev_state.detach(), action.detach(), state.detach()) # TODO: Use the replay-buffer class. And, only store `action` and `state`, since our sampling will be index-based.












# TODO: A class for the replay buffer (with max-len as a hyperparam), with methods for `len(replay_buffer)` for indexing correctly, adding data (arrays, such as `[rating, input, action, as_goal]`; first item is required, possibly `None`, others are whatever), sampling data (at a 0-based index; in-internal-array index is that plus head-pos mod buffer-length), and sampling max-rating data (unroll-time goals) (primitive, on-GPU algo: sample like 10 (an arg) samples (reusing the data-sampling method), and use `torch.where` per-item to find the max-rating among them).

# TODO: Two `embed(src_or_dst: -1|1, input: [action_sz])` nets, for estimating uncertainty.
# TODO: `act(src, dst)`
# TODO: A replay buffer.
# TODO: At unroll-time:
#   TODO: Update env & get a new goal-conditioned action.
#   TODO: Store None & state & action/accel & just-the-pos (`as_goal`: first 2 numbers of state, with constant padding to distinguish it from state) in the replay buffer.
#   TODO: Sometimes (`torch.where`): max-sample a new goal from the replay-buffer (and compute the new estimated-steps-to-reach-it), if reached (abs of goal-state minus goal is less than 1e-3) or out-of-time.
#     TODO: Keep track of how many times we've switched a goal, vs how many times the goal timed out.
#       TODO: `log` these.
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
# TODO: `log`: pick a destination randomly, and visualize actions-to-it as arrows. (Assuming that we succeed in learning all-paths, at least a little.)















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