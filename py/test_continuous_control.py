"""
Can RL be done not via approximating the loss (discounted sum of future distances here, AKA return), but via BPTT? Mathematically, sure, but what are the obstacles to doing it practically?

This 2D env has 1 spaceship and a repulsor in the middle; the actions control acceleration. All smooth, so grad-min should be able to find best actions. Effective exploration would be able to cover a lot of the torus.

---

We implement an RNN that minimizes the distance between goal-conditioned paths and goals, sampling goals from the recent past, predicting the next-frame, and minimizing its distance to the goal.
"""



import torch
import torch.nn as nn
device = 'cuda' if torch.cuda.is_available() else 'cpu'



from model.momentum_copy import MomentumCopy
from model.rnn import RNN
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
    accel = accel[..., :2]
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
        return y * bootstrap_discount + x[..., :y.shape[-1]]
def to_np(x): return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
def cat(*a, dim=-1): return torch.cat(a, dim)



batch_size = 100
input_sz, action_sz = 2, 128
lr = 1e-3

bootstrap_discount = .99

replay_buffer = [None] * (2*1024)



embed = nn.Sequential( # (prev_action, input) → embed_action
    # (Incorporate input into the RNN.)
    # (We 'predict' via joint `embed`ding of both prediction and target, like BYOL, though with not just the previous frame used but with the entire history.)
    SkipConnection(nn.Linear(action_sz + input_sz, action_sz)),
    *[SkipConnection(
        nn.ReLU(), nn.LayerNorm(action_sz),
        nn.Linear(action_sz, action_sz),
     ) for _ in range(1)],
).to(device)
embed_delayed = MomentumCopy(embed, .999)
#   (All prediction targets are delayed, so that gradient serves to contrast different inputs.)
next = nn.Sequential( # (embed_action, goal) → action
    # (Both RNN's post-`embed` transition, and BYOL's predictor.)
    # (`goal` is sampled from the recent past: `replay_buffer`. It's what we want trajectories to minimize the distance to, to gain competency.)
    SkipConnection(nn.Linear(action_sz + action_sz, action_sz)),
    *[SkipConnection(
        nn.ReLU(), nn.LayerNorm(action_sz),
        nn.Linear(action_sz, action_sz),
     ) for _ in range(1)],
).to(device)

class WithInput(nn.Module):
    def __init__(self, embed, next): super().__init__();  self.embed, self.next = embed, next
    def forward(self, prev_action, input, goal):
        embed_action = self.embed(cat(prev_action, input))
        return self.next(cat(embed_action, goal))
def loss(prev_action, action, input, goal):
    global last_losses
    # Next-frame (embedding) prediction: `prev_action = embed_delayed(prev_action, input)`.
    with torch.no_grad():
        next_frame = embed_delayed(cat(prev_action, input))
    next_frame_loss = (prev_action - next_frame).square().sum()
    # Goal (embedding) steering: `prev_action = goal`.
    #   (`goal` should be `embed_delayed(some_prev_action, some_input)`.)
    goal_loss = (prev_action - goal).abs().sum()
    last_losses = next_frame_loss, goal_loss
    return next_frame_loss + goal_loss
step = RNN( # (prev_action, input, goal) → action
    transition = WithInput(embed, next), loss = loss,
    optimizer = lambda p: torch.optim.Adam(p, lr=lr),
    backprop_length = None,
)



# The main loop, which steps the environment and trains `step`.
action = torch.randn(batch_size, action_sz, device=device)
goal = torch.randn(batch_size, action_sz, device=device)
state, hidden_state = env_init(batch_size=batch_size)
last_losses = 0, 0
def reset():
    """Finish a BPTT step, and update the `goal`."""
    global action, goal
    action = step.reset(action)
    with torch.no_grad():
        ch = random.choice(replay_buffer)
        if ch is not None:
            prev_action, prev_state, cur_action, cur_state = ch
            goal = embed_delayed(cat(prev_action, cur_state))
def pos_histogram(plt):
    """That replay buffer contains lots of past positions. This func plots those as a 2D histogram."""
    x, y = [], []
    for ch in replay_buffer:
        if ch is not None:
            prev_action, prev_state, action, state = ch
            pos = to_np(state)
            for i in range(pos.shape[0]):
                x.append(float(pos[i][0])), y.append(float(pos[i][1]))
    plt.hist2d(x, y, bins=100, range=((0,1), (0,1)))
reset()
for iter in range(50000):
    prev_action, prev_state = action, state
    state, hidden_state = env_step(state, hidden_state, prev_action)
    action = step(prev_action, state, goal)

    if random.randint(1, 32) == 1: reset()
    embed_delayed.update()

    replay_buffer[iter % len(replay_buffer)] = (prev_action.detach(), prev_state.detach(), action.detach(), state.detach())

    # TODO: Also `False` to not print to console uselessly.
    log(0, next_frame_loss = to_np(last_losses[0]))
    log(1, goal_loss = to_np(last_losses[1]))
    log(2, pos = pos_histogram)

    # TODO: Run. Ideally, also fix, but this solution is so ambitious that I don't know if it can possibly work.
    #   (Ended up merging RNNs with BYOL in the design, because it seemed so natural. With so much creativity, I fear that it won't work out, no matter how tight the fit is.)
    #   TODO: At least find out why it's broken.
    #   TODO: Why does the loss explode currently? And why do we "try to backward through the graph a second time"?




# TODO: Empirically verify (or contradict) that RL can really be replaced by pointwise minimization.
#   - TODO: Make `next` (or `embed`) accept a completely random vector I guess, for more action diversity.
#   - TODO: …Wait, *mathematically*, if the loss minimizes the future-sum that is the output of the prior `step`, and RL is supposed to minimize that by each action, then isn't our 'action' not the whole sum but merely the residual, AKA the difference between 2 steps?
#   - Tinker with goals:
#     - TODO: During unrolling, try sampling `next`-goals and distance-minimized goals independently, from the replay buffer. (In expectation, equivalent to distance-minimizing to the mean of all goals, so this couldn't be right.)
#     - TODO: During unrolling, try sampling per-step `next`'s and distance-minimized goals.
#     - TODO: During unrolling, try re-sampling the goal ONLY between BPTT steps. (Most conservative.)
#   - *Learn* the loss to minimize:
#     - TODO: RL: learn `future_dist` which represents the (discounted) future sum of all L1 distances between RNN-states (actions) and goals. And minimize that by actions.
#     - TODO: Also try learning not only one-goal loss expectation but all-goals loss expectation, from state & action to that, and make each action minimize that. (The more gradient sources the merrier, right?)



# TODO: Gotta get back, back to the past:
#   TODO: In `test.py`, implement self-targeting RL (with dist-bootstrapping and `next`-dist-min and self-imitation) and self-targeting BPTT (with `next`-dist-min and a skip connection), and try to not just explore one graph but *learn* to explore `minienv`'s graphs. (I don't think any RL exploration method can *learn* to explore, only explore. So if it works, it's cool.)