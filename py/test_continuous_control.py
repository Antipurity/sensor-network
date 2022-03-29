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
    force_center = torch.ones(posit.shape[0], 2, device=device)/2 # TODO: …Why does putting it at the non-center seem to break the env?… …Maybe because the other side doesn't have any force, so it's just an accelerator, not an obstacle.
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

bootstrap_discount = .99

replay_buffer = [None] * (1024)
replays_per_step = 8



next = nn.Sequential( # (prev_action, input, goal) → action # TODO: Also the `randn` arg (since we don't have `embed` anymore).
    # (`goal` is sampled from the recent past: `replay_buffer`. It's what we want trajectories to minimize the distance to, to gain competency.)
    SkipConnection(nn.Linear(action_sz + input_sz + input_sz, action_sz)),
    *[SkipConnection(
        nn.ReLU(), nn.LayerNorm(action_sz),
        nn.Linear(action_sz, action_sz),
    ) for _ in range(1)],
    nn.LayerNorm(action_sz),
).to(device)
state_predictor = nn.Sequential( # (prev_action, prev_state) → input
    *[SkipConnection(
        nn.ReLU(), nn.LayerNorm(action_sz),
        nn.Linear(action_sz, action_sz),
    ) for _ in range(1)],
    nn.Linear(action_sz, input_sz),
).to(device)
future_advantage = nn.Sequential( # (prev_action, goal, action1, action2) → dist2 - dist1
    # (Returns how much lower the sum-of-future-L1-distances-to-`goal` is for `action1` compared to `action2`; the more the better for `action1`.)
    # (A lot of distances are very similar which makes optimization difficult, so predicting differences should be easier.)
    SkipConnection(nn.Linear(action_sz + input_sz + action_sz, action_sz)),
    *[SkipConnection(
        nn.ReLU(), nn.LayerNorm(action_sz),
        nn.Linear(action_sz, action_sz),
    ) for _ in range(1)],
    nn.Linear(action_sz, input_sz),
).to(device)
action_grad = nn.Sequential( # (action, state, goal) → action_grad
    # (Learning from the replay buffer cuts off gradient, so with this synthetic gradient, we could treat actions as RNN states.)
    SkipConnection(nn.Linear(action_sz + input_sz + input_sz, action_sz)),
    *[SkipConnection(
        nn.ReLU(), nn.LayerNorm(action_sz),
        nn.Linear(action_sz, action_sz),
    ) for _ in range(1)],
).to(device)

class WithInput(nn.Module):
    def __init__(self, next, state_predictor, future_advantage, action_grad):
        super().__init__()
        self.next, self.state_predictor, self.future_advantage, self.action_grad = next, state_predictor, future_advantage, action_grad
    def forward(self, prev_action, input, randn, goal):
        return self.next(cat(prev_action, input, goal))
step = WithInput(next, state_predictor, future_advantage, action_grad)
optim = torch.optim.Adam(step.parameters(), lr=lr)



# The main loop, which steps the environment and trains `step`.
action = torch.randn(batch_size, action_sz, requires_grad=True, device=device)
goal = torch.rand(batch_size, input_sz, device=device)
state, hidden_state = env_init(batch_size=batch_size)
total_loss = 0
def reset_goal():
    global goal
    with torch.no_grad():
        ch = random.choice(replay_buffer)
        if ch is not None:
            prev_action, prev_state, cur_action, cur_state = ch
            randn = torch.randn(batch_size, action_sz, device=device)
            # goal = cur_state # TODO:
            goal = torch.rand(batch_size, input_sz, device=device)
def reset():
    """Finish a BPTT step, and update the `goal`."""
    global action, total_loss
    if isinstance(total_loss, torch.Tensor) and total_loss.requires_grad: total_loss.backward()
    total_loss = 0
    optim.step();  optim.zero_grad(True)
    action = action.detach().requires_grad_(True)
    reset_goal()
def pos_histogram(plt, label):
    """That replay buffer contains lots of past positions. This func plots those as a 2D histogram."""
    x, y = [], []
    for ch in replay_buffer:
        if ch is not None:
            prev_action, prev_state, action, state, next_action, next_state = ch
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
        next_action = torch.cat([c[4] for c in choices], 0)
        next_state = torch.cat([c[5] for c in choices], 0)
    prev_action.requires_grad_(True)

    goal = state[torch.randperm(state.shape[0], device=device)]
    randn = torch.randn(state.shape[0], action_sz, device=device)

    # Predict the next state.
    state2 = state_predictor(cat(prev_action, prev_state))
    next_state2 = state_predictor(cat(prev_action, prev_state))
    state_pred_loss = (state2 - state).square().sum() + (next_state2 - next_state).square().sum()

    # Learn `future_advantage` by bootstrapping, comparing replay_buffer's policy to our own.
    action2 = next(cat(prev_action, state, goal))
    next_action2 = next(cat(action, next_state, goal))
    prev_adv = future_advantage(cat(prev_action, goal, action, action2))
    micro_adv = (state2 - goal).abs() + (state - goal).abs()
    for p in future_advantage.parameters(): p.requires_grad_(False)
    adv = future_advantage(cat(action, goal, next_action, next_action2))
    for p in future_advantage.parameters(): p.requires_grad_(True)
    adv_loss = (prev_adv - (micro_adv + adv * bootstrap_discount).detach()).square().sum()

    # Grad-minimize the replay-sample's advantage, making our own policy better.
    goal_loss = adv.sum()

    # Synthetic gradient of actions.
    # TODO: …Wait, how would we know `prev_action`'s gradient if only `.backward()` fills in the gradient, which is *not here*?…
    #   Should we take control of the optimizer too?…
    #   (For prediction, first try just a multiplicative discount, and if that fails, limit the gradient's norm.)
    # TODO: Give `action_grad(cat(action, state, goal))` to both `action` and `next_action`: multiply that by its detached predicted gradient, and sum that to get the loss.
    synth_grad_loss = 0
    prev_action.requires_grad_(False)

    # Log them.
    log(0, False, pos = pos_histogram)
    log(1, False, state_pred_loss = to_np(state_pred_loss))
    log(2, False, adv_loss = to_np(adv_loss))
    log(3, False, goal_loss = to_np(goal_loss))

    return state_pred_loss + adv_loss + goal_loss



reset()
prev_data = None
for iter in range(500000):
    prev_action, prev_state = action, state
    state, hidden_state = env_step(state, hidden_state, prev_action)
    randn = torch.randn(batch_size, action_sz, device=device)
    action = step(prev_action, state, randn, goal)

    total_loss += replay()
    if iter == 1000: clear()

    if random.randint(1, 64) == 1: state, hidden_state = env_init(batch_size=batch_size) # TODO:
    if random.randint(1, 32) == 1: reset()

    if prev_data is not None:
        replay_buffer[iter % len(replay_buffer)] = (*prev_data, prev_action.detach(), prev_state.detach(), action.detach(), state.detach())
    prev_data = (prev_action.detach(), prev_state.detach())

    # TODO: Run. Ideally, also fix, but this solution is so ambitious that I don't know if it can possibly work.
    #   (Ended up merging RNNs with BYOL in the design, because it seemed so natural. With so much creativity, I fear that it won't work out, no matter how tight the fit is. …Pretty sure that it didn't work out, at least in the initial attempt.)
    #   TODO: At least find out why it's broken.




# TODO: Empirically verify (or contradict) that RL can really be replaced by pointwise minimization.
#   - TODO: Just doing what we want didn't work, so find out exactly what we *can* do:
#     - TODO: Possibly, input not just xy but its `x → 1-2*abs(x)` fractal filling.
#   - TODO: Tinker with goals: new goal per-step, or only on BPTT resets.
#   - *Learn* the loss to minimize:
#     - TODO: RL: learn `future_dist` which represents the (discounted) future sum of all L1 distances between RNN-states (actions) and goals. And minimize that by actions.
#       - (Could be good for us, actually, since our BPTT length is probably far too small to allow most goals to be achieved.)
#       - TODO: `log` not just `pos_histogram` but also how poorly the goals are reached, by preserving distance-estimations and weighing by that in `plt.plot2d`.
#       - TODO: Possibly, for more accuracy (since it'll be much closer to 0 most of the time), bootstrap/learn not the distance directly but its `future_advantage` (diff between 2 distances, possibly only between `next`-suggested and in-replay actions), and for each replayed transition, maximize not the distance but the advantage over in-replay action.
#     - TODO: Learn synthetic gradient (multiplied by `bootstrap_discount` each time, to downrate the future's effect on the past), and compare with RL?
#     - TODO: Possibly: generate the step's goal by a neural-net, which maximizes future-distance or something.
#   - Retain non-differentiably-reachable minima, via self-imitation learning:
#     - TODO: An extra loss on `next` of `prev_action`: `(next(prev_action) - action) * (dist(next(prev_action)) - dist(action)).detach()`.
#     - TODO: Make that best-past-action a `.detach()`ed input to `next` instead (`best_next: (prev_action, input_emb) → best_action`), to not explicitly collapse diversity.
#   - TODO: Instead of simple next-frame prediction, embed inputs once again.



# …Augmentating images in computer vision to make NN representations invariant to them, is equivalent to doing that for consecutive RNN steps with body/eye movement — but not exactly to full-RNN-state invariance/prediction…

# …With embedding-prediction, I'm pretty sure it's the same as prediction, but if some parts are too hard to predict for too long, the model just gives up on them. Interference is also a lot of the problem in learning the shortest path (very many paths are essentially the same); is there any way to combine the two?…

# …We only try to improve the reachability of one goal at a time, which is synonymous with "non-scalable". Is there no way to construct representations of exponentially-many goals, and update many goals at once… Can embedding-prediction make similar goals the same and distinct goals different?…



# TODO: Gotta get back, back to the past:
#   TODO: In `test.py`, implement self-targeting RL (with dist-bootstrapping and `next`-dist-min and self-imitation) and self-targeting BPTT (with `next`-dist-min and a skip connection), and try to not just explore one graph but *learn* to explore `minienv`'s graphs. (I don't think any RL exploration method can *learn* to explore, only explore. So if it works, it's cool.)