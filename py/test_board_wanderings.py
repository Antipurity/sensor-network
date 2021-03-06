"""
Pretraining for all possible goals is like learning a map of how to get from anywhere to anywhere. Here, we gradually build up to the full formulation in an easily-comprehensible and easily-visualizable environment.

We have a square board here, and 1 agent that can only walk it. If we can't learn a map here, then how can we expect to learn a map in more complex environments?

With as few env-specific tricks as we can manage.

---

Ultimately, to pre-train for all possible goals, we want to learn how to eventually get from anywhere to anywhere in `state`-space.

---

Let's define our transition model as `next: (prev_state, goal) → next_state` (which also includes input/output inside of it for simplicity of our analysis). We'd like to learn good `next_state` — but where could we possibly get a good loss to do that?

There exist many `state→state→…→state→state` trajectories, and we'd like to make sure that they all encounter `goal` as soon as possible. Moreover, we only really need to know one trajectory for each `(state, goal)` pair, the shortest one (which rules out the hard generative-modeling and allows using the easy prediction): if our `state`-space is a faithfully-compressed representation of inputs/outputs, then state-goals won't miss any details and thus won't have to intervene on trajectories to patch up blindspots.

We need to consider a trajectory's first transition to its `next` state, and for each possible action (`next_state`), measure/learn its future's distance to the `goal`, and minimize that (this problem is known as [optimal control](http://www.scholarpedia.org/article/Optimal_control)). Succeed at this everywhere, and our map is learned.

- What's the actual distance, on a trajectory? Note that our states/goals are continuous, so we can't just count transitions until equality, nor can we explicitly dream up individual trajectories to pick the shortest one (there are infinitely many, so that's either inaccurate or taking too long).
  - The less transitions until we're at `goal`, and the longer we stay there afterward, the better. The future distance being the *sum* of individual `dist`ances fits this quite nicely, and is easy to learn.
    - (With a sum, we don't have to materialize full trajectories to learn distances, we can just bootstrap from single transitions via `future_dist(prev, goal) = (future_dist(next, goal) + dist(prev, goal)).detach()`.)
  - `dist(x,y)`: for easy optimization, something linear, and maybe summed-up as late as possible for a richer learning signal. L1 AKA `(x-y).abs().sum()` fits the bill.

---

In summary, our 'loss' to minimize is: "sum over all goals `y`: sum over an unrolled trajectory of `x`s: `(x-y).abs().sum()`".

In practice, on this board-env with 4 actions, gradient descent cannot find the best action. So we use RL, where we learn the inner sum by bootstrapping the neural net `future_dist: (state, goal) → future_distance`, and at each step of an unroll, consider all 4 actions and pick the min-predicted-distance one. This way, we can just pick a step from a replay buffer, pick a random goal-state from the replay buffer, and learn the distance.

We've tried many tricks, but this seems to work best.

---

Our current impl's scalability problems:
- We have neither randomly-selected nor emergent goals, only predetermined board states.
- This env is perfect-info; might want to investigate what happens when we only provide the board at the first time step (should still be learnable, but much slower than BPTT).
- `future_dist` is only an approximation of the true (part of the) loss to minimize. BPTT would have been a different approximation, at least.
    - When targets are not just boards but their neural-net mappings: if mapping just a board, we can still learn fine, but when mapping board & action, we can't learn anything. RL here seems very… non-scalable to more degrees of freedom.

Problems & solutions:
- Grad-based min can't find global action-minima, because the future-distance surface is very non-smooth. Solution? Self-imitation learning: `loss = ((action2 - action).square() * (dist2 - dist).detach().max(zeros))`, where `action` is from the replay buffer and `action2` is from `next`, to make `next` predict its lowest-distance version. However, in optimal conditions, it's not as good as enumerating all actions at unroll-time.
- Low batch sizes (even 1) make no difference thanks to the replay buffer, as long as its size and `updates_per_unroll` are big enough.

---

Future work:
- Some env where grad-min can actually work. There, we can try to forego the RL machinery of "approximate the future loss" and try to use BPTT directly. (Our formulated loss has connections to prediction and SSL, but augments it to concern not just adjacent states but all states in a principled manner (the RNN-step knows the target that it will go to) which is useful for env-learning.)
- `minienv`, where we don't just explore, but have to *learn* to explore. Having the best possible optimization is a must with this requirement.
"""



from model.log import log, clear, finish



import torch
import torch.nn as nn
device = 'cuda' if torch.cuda.is_available() else 'cpu'



import random



# The environment: a square board, with 2D actions.
def env_init(N, batch_size=1): # → board
    board = torch.rand(batch_size, N*N, device=device)
    board = (board == board.max(-1, keepdim=True)[0]).float()
    # Collisions are technically possible, but unlikely.
    return board
def env_step(N, board, xy): # → board
    """Given a one-hot board encoding and a 2-number action, returns a new board state, cyclically-shifted in the correct direction."""
    board = board.reshape(*board.shape[:-1], N, N)
    sh = [*board.shape[:-2], N*N]
    board_px = torch.cat((board[..., :, -1:], board[..., :, :-1]), -1).reshape(sh)
    board_py = torch.cat((board[..., -1:, :], board[..., :-1, :]), -2).reshape(sh)
    board_mx = torch.cat((board[..., :, 1:], board[..., :, :1], ), -1).reshape(sh)
    board_my = torch.cat((board[..., 1:, :], board[..., :1, :], ), -2).reshape(sh)
    is_x = xy[..., :1].abs() > xy[..., 1:].abs()
    return torch.where(
        is_x,
        torch.where(xy[..., :1] > 0, board_px, board_mx),
        torch.where(xy[..., 1:] > 0, board_py, board_my),
    )



class SkipConnection(nn.Module):
    def __init__(self, *fn): super().__init__();  self.fn = nn.Sequential(*fn)
    def forward(self, x): return self.fn(x) + x
def to_np(x): return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
def cat(*a, dim=-1): return torch.cat(a, dim)



N, batch_size = 8, 100
action_sz = 64
overparameterized = 1

unroll_len = N
action_min = True # If `True`, we enum the 4 actions to pick the min-future-distance one at each step.
grad_min = False # With our 4 discrete actions, grad-min just doesn't work no matter what we try.
#   (Grad-min assumes a smooth landscape. Consequences of actions don't sound like a smooth function.)
#   (What else can we do? Learn a generative model of distinct-outcome actions with low distance, and still do action-min on enumerably-few actions?)
self_imitation = False # Makes `next` predict its min-dist version.
#   (Like [SIL](https://arxiv.org/abs/1806.05635) but without probabilities and with learned R.)
#   (Much better than `grad_min`, much worse than `action_min`.)

replay_buffer = [None] * 1024
updates_per_unroll = N

bootstrap_discount = torch.tensor([.95], device=device)
#   Bootstrapping is `f(next) = THIS * f(prev) + local_metric(next)`
#   (Predicting many discounts at once doesn't help.)



next = nn.Sequential( # (prev_board, prev_action, target, random) → action
    nn.Linear(N*N + N*N + action_sz + action_sz, overparameterized * action_sz),
    nn.ReLU(),
    nn.LayerNorm(overparameterized * action_sz),
    nn.Linear(overparameterized * action_sz, action_sz),
).to(device)
future_dist = nn.Sequential( # (prev_board, action, target) → future_distance_sum
    # (For picking an action that leads to getting to the target board the fastest.)
    nn.Linear(N*N + action_sz + N*N, overparameterized * action_sz),
    nn.ReLU(),
    nn.LayerNorm(overparameterized * action_sz),
    SkipConnection(
        nn.Linear(overparameterized * action_sz, overparameterized * action_sz),
        nn.ReLU(),
        nn.LayerNorm(overparameterized * action_sz),
    ),
    SkipConnection(
        nn.Linear(overparameterized * action_sz, overparameterized * action_sz),
        nn.ReLU(),
        nn.LayerNorm(overparameterized * action_sz),
    ),
    nn.Linear(overparameterized * action_sz, bootstrap_discount.shape[0]),
).to(device)
opt = torch.optim.Adam([*next.parameters(), *future_dist.parameters()], lr=1e-3)

for iters in range(50000):
    # Sample a batch of trajectories.
    dist_pred_loss, dist_min_loss, self_imitation_loss = 0, 0, 0
    action = torch.zeros(batch_size, action_sz, device=device)
    board = env_init(N, batch_size=batch_size)
    dist_mean = 0
    with torch.no_grad():
        # First pick the target to go to.
        #   (From tests, in this trivial env, performance is the same even if we re-pick the target at each step.)
        target = random.choice(replay_buffer)
        target = target[2] if target is not None else torch.zeros(batch_size, N*N, device=device)
        for u in range(unroll_len):
            # Do the RNN transition (and an environment step), `unroll_len` times.
            zeros = torch.zeros(batch_size, action_sz, device=device)
            rand = torch.randn(batch_size, action_sz, device=device)
            prev_board, prev_action = board, action
            # Minimize the future-distance-sum by considering all 4 possible actions right here.
            #   (Minimizing by gradient descent in this environment is no bueno.)
            action = next(cat(prev_board, prev_action, target, rand))
            #   Using `zeros` in place of `rand` here is 4× slower to converge.
            if action_min:
                a1, a2, arest = action.split((1, 1, action.shape[-1]-2), -1)
                action_candidates = [cat(a1, a2, arest), cat(a1, a2, arest), cat(-a1, -a2, arest), cat(-a2, -a1, arest)]
                min_action, min_dist = None, None
                with torch.no_grad():
                    for action in action_candidates:
                        dist = future_dist(cat(prev_board, action, target)).sum(-1, keepdim=True)
                        if min_dist is None: min_action, min_dist = action, dist
                        else:
                            mask = dist < min_dist
                            min_action = torch.where(mask, action, min_action)
                            min_dist = torch.where(mask, dist, min_dist)
                action = min_action

            board = env_step(N, prev_board, action[..., 0:2])

            micro_dist = (board - target).abs().detach()
            dist_mean += micro_dist.mean(0).sum() # Something to log.

            index = (iters*unroll_len + u) % len(replay_buffer)
            replay_buffer[index] = (prev_board, prev_action.detach(), board, action.detach())

    # Replay from the buffer. (Needs Python 3.6+ for convenience.)
    choices = [c for c in random.choices(replay_buffer, k=updates_per_unroll) if c is not None]
    if len(choices):
        prev_board = torch.cat([c[0] for c in choices], 0)
        prev_action = torch.cat([c[1] for c in choices], 0)
        board = torch.cat([c[2] for c in choices], 0)
        action = torch.cat([c[3] for c in choices], 0)

        target = torch.where( # This actually seems to improve convergence speed 2×.
            #   (`prev_board` does not. More/less than 50% of the time, does not.)
            #   (Possibly, by doing this we're teaching the net how to stay at the goal once reached, which can be done many times in expectation and thus have a disproportionate effect compared to random goals.)
            torch.rand(board.shape[0], 1, device=device) < .5,
            board,
            board[torch.randperm(board.shape[0], device=device)],
        )

        zeros = torch.zeros(board.shape[0], action_sz, device=device)
        rand = torch.randn(board.shape[0], action_sz, device=device)

        # Bootstrapping: `future_dist(prev) = future_dist(next)*p + micro_dist(next)`
        micro_dist = (board - target).abs().sum(-1, keepdim=True)
        prev_dist = future_dist(cat(prev_board, action, target.detach()))
        next_action = next(cat(board, action, target.detach(), rand))
        for p in future_dist.parameters(): p.requires_grad_(False)
        dist = future_dist(cat(board, next_action, target.detach()))
        for p in future_dist.parameters(): p.requires_grad_(True)
        prev_dist_targ = dist * bootstrap_discount + micro_dist
        dist_pred_loss = dist_pred_loss + (prev_dist - prev_dist_targ.detach()).square().mean(0).sum()

        # Min-dist self-imitation, by `next`.
        action2 = next(cat(prev_board, prev_action, target.detach(), rand))
        prev_dist2 = future_dist(cat(prev_board, action2, target.detach()))
        prev_dist, prev_dist2 = prev_dist.sum(-1, keepdim=True), prev_dist2.sum(-1, keepdim=True)
        self_imitation_loss = self_imitation_loss + ((action2 - action.detach()).square() * (prev_dist2 - prev_dist).detach().max(zeros)).sum()
        if not self_imitation: self_imitation_loss = self_imitation_loss.detach()

        # Grad-min of actions.
        dist_min_loss = dist_min_loss + dist.mean(0).sum()
        if not grad_min: dist_min_loss = dist_min_loss.detach()

    if iters == 500: clear()

    (dist_pred_loss + dist_min_loss + self_imitation_loss + torch.zeros(1, device=device, requires_grad=True)).backward()
    opt.step();  opt.zero_grad(True)
    with torch.no_grad():
        log(0, False, dist_pred_loss = to_np(dist_pred_loss))
        log(1, False, dist_min_loss = to_np(dist_min_loss))
        log(2, False, self_imitation_loss = to_np(self_imitation_loss))
        log(3, False, mean_distance = dist_mean / (2*N))
        #   (Reaching about .66 means that targets are reached about 100% of the time.)
        log(4, False, action_mean = to_np(action.mean()), action_std = to_np(action.std()))
finish()