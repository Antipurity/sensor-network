"""
Pretraining for all possible goals is like learning a map of how to get from anywhere to anywhere. Here, we gradually build up to the full formulation in an easily-comprehensible and easily-visualizable environment.

We have a square board here, and 1 agent that can only walk it. If we can't learn a map here, then how can we expect to learn a map in more complex environments?

NO RL machinery! Lessons must be transferrable all the way.

---

Ultimately, to pre-train for all possible goals, we want to learn how to eventually get from anywhere to anywhere in `state`-space.


Let's define our transition model as `next: (prev_state, goal) → next_state` (which also includes input/output inside of it for simplicity of our analysis). We'd like to learn good `next_state` — but where could we possibly get a good loss to do that?

There exist many `state→state→…→state→state` trajectories, and we'd like to make sure that they all encounter `goal` as soon as possible. Moreover, we only really need to know one trajectory for each `(state, goal)` pair, the shortest one (which rules out the hard generative-modeling and allows using the easy prediction): if our `state`-space is a faithfully-compressed representation of inputs/outputs, then state-goals won't miss any details and thus won't have to intervene on trajectories to patch up blindspots.

We need to consider a trajectory's first transition to its `next` state, and for each possible action (`next_state`), measure/learn its future's distance to the `goal`, and minimize that (this problem is known as [optimal control](http://www.scholarpedia.org/article/Optimal_control#:~:text=Optimal%20control%20is%20the%20process,to%20minimise%20a%20performance%20index.)). Succeed at this everywhere, and our map is learned.

1. A neural net that would learn the future-distance: `future_dist: (state, goal) → future_distance`. Should be learned like a GAN: learned without gradient to `state` & `goal` (only its params), and minimized with gradient only to `state` & `goal` (not to its params, to not undo learning).
2. The actual distance, on a trajectory? Note that our states/goals are continuous, so we can't just count transitions until equality, nor can we explicitly dream up individual trajectories to pick the shortest one (there are infinitely many, so that's either inaccurate or taking too long).
  - The less transitions until we're at `goal`, and the longer we stay there afterward, the better. The future distance being the *sum* of individual `dist`ances fits this quite nicely, and is easy to learn.
    - (With a sum, we don't have to materialize full trajectories to learn distances, we can just bootstrap from single transitions via `future_dist(prev, goal) = (future_dist(next, goal) + dist(prev, goal)).detach()`.)
  - `dist(x,y)`: for easy optimization, something linear, and summed-up as late as possible for a richer learning signal. L1 AKA `(x-y).abs().sum()` fits the bill.

Our current impl's scalability problems:
- Not using grad-based min, only min via considering all 4 actions at each step. The action space is really discontinuous; can only hope that compression (like Barlow twins) can help.
- When varying `N`: convergence speed seems to be roughly cubic; the further away the targets are, the harder it is to learn the proper distance to them.
- Low batch sizes don't really work, which is a big problem if we want to have only 1 thread of experience. (…Though, a bit irrelevant if using replay buffers.)
- Have neither randomly-selected nor emergent goals, only predetermined board states.
- This env is perfect-info; might want to investigate what happens when we only provide the board at the first time step (should still be learnable, but much slower than BPTT).
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



N, batch_size = 8, 100
state_sz = 64
overparameterized = 1

unroll_len = N
action_min = True # If `True`, we enum the 4 actions to pick the min-future-distance one at each step.
#   (With our 4 discrete actions, grad-min just doesn't work no matter what tricks we try.)

replay_buffer = [None] * 1024
updates_per_unroll = N
bootstrap_discount = torch.tensor([.95], device=device) # Bootstrapping is `f(next) = THIS * f(prev) + local_metric(next)`
#   TODO: …Would summing actually be helped if we had multipliers that made all horizons equally important?…
#     What's the formula, though?
#   (Predicting many discounts at once actually slows down convergence, doesn't help.)



next = nn.Sequential( # (prev_board, prev_target, prev_action, random) → action
    nn.Linear(N*N + N*N + state_sz + state_sz, overparameterized * state_sz),
    nn.ReLU(),
    nn.LayerNorm(overparameterized * state_sz),
    nn.Linear(overparameterized * state_sz, state_sz),
).to(device)
future_dist = nn.Sequential( # (prev_board, action, target) → future_distance_sum
    # (For picking an action that leads to getting to the target board the fastest.)
    nn.Linear(N*N + state_sz + N*N, overparameterized * state_sz),
    nn.ReLU(),
    nn.LayerNorm(overparameterized * state_sz),
    SkipConnection(
        nn.Linear(overparameterized * state_sz, overparameterized * state_sz),
        nn.ReLU(),
        nn.LayerNorm(overparameterized * state_sz),
    ),
    SkipConnection(
        nn.Linear(overparameterized * state_sz, overparameterized * state_sz),
        nn.ReLU(),
        nn.LayerNorm(overparameterized * state_sz),
    ),
    nn.Linear(overparameterized * state_sz, bootstrap_discount.shape[0]),
).to(device)
past_target = nn.Sequential( # (prev_board, prev_action, target) → prev_target
    nn.Linear(N*N + state_sz + N*N, overparameterized * state_sz),
    nn.ReLU(),
    nn.LayerNorm(overparameterized * state_sz),
    SkipConnection(
        nn.Linear(overparameterized * state_sz, overparameterized * state_sz),
        nn.ReLU(),
        nn.LayerNorm(overparameterized * state_sz),
    ),
    nn.Linear(overparameterized * state_sz, N*N),
).to(device)
predict_target = nn.Sequential( # (board, action) → target
    # TODO: WAIT THIS IS NOT ENOUGH INFO TO INFER THE TARGET, WHAT THE HELL
    nn.Linear(N*N + state_sz, overparameterized * state_sz),
    nn.ReLU(),
    nn.LayerNorm(overparameterized * state_sz),
    SkipConnection(
        nn.Linear(overparameterized * state_sz, overparameterized * state_sz),
        nn.ReLU(),
        nn.LayerNorm(overparameterized * state_sz),
    ),
    nn.Linear(overparameterized * state_sz, N*N),
).to(device)
opt = torch.optim.Adam([*next.parameters(), *future_dist.parameters(), *past_target.parameters(), *predict_target.parameters()], lr=1e-3)

for iters in range(50000):
    # TODO: …Try re-reading Go-Explore more carefully; because making that continuous-ish and neural-net-ish does sound like a very promising approach to making proper maps of environments.

    # TODO: …Self-decided goals?… Refining an initially locally inconsistent picture into more locally-consistent forms, eventually causing global consistency?… (Naïve Barlow twins kinda failed the last time we tried that.)
    #   …But why would our synthetic-targets be able to do that…

    # …Wait, our reward-formulation is "for all possible goals, minimize the distance over a full goal-conditioned trajectory", right? Why not model "all possible goals" in service to another goal? TODO: Have a neural net `past_target(board, action, next_target) → target` from future to past that learns the min-future-distance auxiliary target to condition `next` on; bootstrap it.
    #   (& to enable both acting and bootstrapping: have the past-dependent neural net `predict_target(board, action) → target` predict this future-ordained goal.)
    #   (*Might* improve stability by moving the responsibility of learning long-term dependencies from `next` to bootstrapping.)
    #   (If we don't actually fix a target, we *might* be able to learn imagined targets, eventually refining them to fixed points. Which makes this worth trying, because auto-goal-extraction from RNN state is what we wanted in the first place, and this method is actually motivated by RL stuff, not other-field vaguely-related stuff.)
    #   TODO: …Isn't this targeting business suggesting that we could unite all our neural nets into one, with `(board, action, target)` at both input and output (or maybe 2, one prev→next and the other next→prev), and train by just masking out inputs?…
    #     1. What's the difference between actions and targets in this formulation?… Can't we merge them, making this bigger neural net pretty pointless…
    #       (Maybe we could have a prev-action predictor given next-action, though.)
    #       We do need goals if we goal-condition, on any space that's not action-space. And if we have goals, then we could learn them.
    #     2. These targets are so indistinct that we won't be able to actually extract any self-decided goals, right?…



    # Sample a batch of trajectories (pre-deciding the target-board).
    dist_pred_loss, dist_min_loss = 0, 0
    state = torch.zeros(batch_size, state_sz, device=device)
    board = env_init(N, batch_size=batch_size)
    target = env_init(N, batch_size=batch_size)
    # target = (target - target.mean(-1, keepdim=True)) / (target.std(-1, keepdim=True) + 1e-5) # TODO:
    for p in future_dist.parameters(): p.requires_grad_(False)
    dist_sum = 0
    for u in range(unroll_len):
        # Do the RNN transition (and an environment step), `unroll_len` times.
        zeros = torch.zeros(batch_size, state_sz, device=device)
        rand = torch.randn(batch_size, state_sz, device=device)
        prev_board, prev_state = board, state
        # Minimize the future-distance-sum by considering all 4 possible actions right here.
        #   (Minimizing by gradient descent in this environment is no bueno.)
        # TODO: …Should use `predict_target(board, state)` to get `target` here… …Except, that's really not enough info…
        state = next(torch.cat((prev_board, target, prev_state, rand), -1))
        #   Using `zeros` in place of `rand` here is 4× slower to converge.
        if action_min:
            sx, sy, srest = state.split((1, 1, state.shape[-1]-2), -1)
            state_candidates = [
                torch.cat((sx, sy, srest), -1),
                torch.cat((sy, sx, srest), -1),
                torch.cat((-sx, -sy, srest), -1),
                torch.cat((-sy, -sx, srest), -1),
            ]
            min_state, min_dist = None, None
            with torch.no_grad():
                for state in state_candidates:
                    dist = future_dist(torch.cat((prev_board, state, target), -1)).sum(-1, keepdim=True)
                    if min_dist is None: min_state, min_dist = state, dist
                    else:
                        mask = dist < min_dist
                        min_state = torch.where(mask, state, min_state)
                        min_dist = torch.where(mask, dist, min_dist)
            state = min_state

        board = env_step(N, prev_board, state[..., 0:2])
        dist_sum += (board - target).abs().sum(-1, keepdim=True).detach() # Something to log.

        index = (iters*unroll_len + u) % len(replay_buffer)
        replay_buffer[index] = (prev_board, prev_state.detach(), board, state.detach())

        # Grad-minimize that sum-of-future-distances by actions.
        #   (In this env of 4 actions, this is very ineffective; use `action_min` instead.)
        #     (I suppose our representation-learning game is too weak.)
        dist_min_loss = dist_min_loss + future_dist(torch.cat((prev_board, state, target), -1)).mean(0).sum()

    for p in future_dist.parameters(): p.requires_grad_(True)

    for _ in range(updates_per_unroll):
        choice = random.choice(replay_buffer)
        if choice is None: continue
        prev_board, prev_action, board, action = choice

        if random.randint(1,2) == 1: # This actually seems to improve convergence speed 2×.
            target = board # (`prev_board` does not.)
            # (May be just a coincidence. Or, could be teaching the net how to stay at the target once reached, which can be done many times in expectation and thus have a disproportionate effect.)
        else:
            target = env_init(N, batch_size=batch_size)

        zeros = torch.zeros(batch_size, state_sz, device=device)
        rand = torch.randn(batch_size, state_sz, device=device)

        # Bootstrapping: `future_dist(prev) = future_dist(next)*p + micro_dist(next)`
        micro_dist = (board - target).abs().sum(-1, keepdim=True)
        #   TODO: `target` always being static is kinda a problem too, right? Not like it's env-given, we just decided it. Better to learn many targets at once, if we can.
        #     Bootstrapped-targets seem like a reasonable refining-of-local-consistency opportunity.
        prev_dist2 = future_dist(torch.cat((prev_board, action, target), -1))
        next_action = next(torch.cat((board, target, action, rand), -1))
        prev_dist_targ = future_dist(torch.cat((board, next_action, target), -1)) * bootstrap_discount + micro_dist
        dist_pred_loss = dist_pred_loss + (prev_dist2 - prev_dist_targ.detach()).square().mean(0).sum()

        # TODO: If `target is not None`, should use `predict_target(board, action)` twice, and get prediction-gradient with `past_target(board, action, target) → prev_target`.

    # if (iters+1) % 5000 == 0: # For debugging, visualize distances from anywhere to a target.
    #     # (Should, instead of relying on prediction-at-state=0, consider 4 states and display the min.)
    #     import matplotlib.pyplot as plt
    #     src = torch.eye(N*N, device=device)
    #     state = torch.zeros(N*N, state_sz, device=device)
    #     dst = torch.eye(1, N*N, device=device).expand(N*N, N*N)
    #     dist = future_dist(torch.cat((src, state, dst), -1)).sum(-1).reshape(N, N).detach().cpu().numpy()
    #     plt.clf()
    #     plt.imshow(dist)
    #     plt.pause(1)
    #     plt.show()
    if iters == 100: clear()

    (dist_pred_loss + dist_min_loss + torch.zeros(1, device=device, requires_grad=True)).backward()
    opt.step();  opt.zero_grad(True)
    with torch.no_grad():
        log(0, False, dist_pred_loss = to_np(dist_pred_loss))
        log(1, False, dist_min_loss = to_np(dist_min_loss))
        log(2, False, avg_distance = (dist_sum.sum(-1).mean() + .3) / (2*N))
        #   (Reaching about .66 means that targets are reached about 100% of the time.)
        log(3, False, state_mean = to_np(state.mean()), state_std = to_np(state.std()))
finish()
# TODO: Okay, what do we want to learn, building up to URL gradually?
#   - ✓ Learning transitions: from board and action (randomly-generated) to board — EASY.
#   - ✓ Learning plans: from board & target-board & extra-state & whole-output (or a random vector initially), to the next action & extra state: `next`. (Limiting the unroll-length, of course.)
#     - ❌ Average-plan makes no sense because everything is connected to everything, so we need to learn the *distribution* of plans that will lead us to the target, so either:
#       - ✓ [Good: 40%] DDPM-like but speedy (not about to do thousands of steps per RNN step): make `next` self-denoising: make it accept its output as an input, initially a random vector, then unroll its RNN and make all stages predict the final one (or a variation on that). (Eventually removed the impl here, and all its tricks.)
#       - ❌ [Bad: 30%, highly unstable] GAN-like: train a discriminator (from board & target-board & extra-state & 'whole-output'-randomness) of whether a trajectory will succeed (known after a whole rollout), and maximize the predicted success-probability by all `next`-steps (but not by the discriminator).
#     - ✓ [Good: 40% with partial impl] Abandon distributions, pick only their shortest paths; special-case our discrete case to see how viable this is. Very viable: didn't even implement prior-path-conditioning, so all shortest paths interfere, making 40% on 4×4 even more impressive. (Performance deterioration is because there are always many valid paths, so when they collide, predicted action goes toward 0 and causes instability before long.)
#     - ✓ [Perfect: 95%] Neural pathfinding: explicitly learn the sum of future L1 distances (here, this is twice the number of steps until target), and minimize that. (Grad-min doesn't work, but considering all 4 actions at each step does.)
#   - ⋯ Also learn targets, don't just decide them randomly…
#   - ⋯ Figure out how to use grad-min instead of action-min, because a simplification relies on this.
#   - ⋯ Don't use boards as states & goals directly, instead compress them with `ev` (such as via Barlow twins).