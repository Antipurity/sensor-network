"""
Pretraining for all possible goals is like learning a map of how to get from anywhere to anywhere. Here, we gradually build up to the full formulation in an easily-comprehensible and easily-visualizable environment.

We have a square board here, and 1 agent that can only walk it. If we can't learn a map here, then how can we expect to learn a map in more complex environments?

NO RL machinery! Lessons must be transferrable all the way.
"""



from model.momentum_copy import MomentumCopy
from model.log import log, clear



from itertools import chain



import torch
import torch.nn as nn
device = 'cuda' if torch.cuda.is_available() else 'cpu'



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



N, batch_size = 4, 100
state_sz = 8
overparameterized = 8

unroll_len = 1
# unroll_len = N-1 # TODO: Test our per-action minimizer with this.



next = nn.Sequential( # (board, target_board, state, output) → output_state
    # (`output` is for sampling from the action distribution on each step, initially a random vector in each iteration, denoised one or more times.)
    # (The actions are simply sliced from `output_state`.)
    nn.Linear(N*N + N*N + state_sz + state_sz, overparameterized * state_sz),
    nn.ReLU(),
    nn.LayerNorm(overparameterized * state_sz),
    SkipConnection(nn.Sequential(
        nn.Linear(overparameterized * state_sz, overparameterized * state_sz),
        nn.ReLU(),
        nn.LayerNorm(overparameterized * state_sz),
    )),
    nn.Linear(overparameterized * state_sz, state_sz),
).to(device)
next_board = nn.Sequential( # (board, state) → next_board (state AKA action)
    # (So that goals don't have to be non-human-comprehensible.)
    # TODO: …By passing `state` to `future_dist`, we've made this obsolete, haven't we?…
    nn.Linear(N*N + state_sz, overparameterized * state_sz),
    nn.ReLU(),
    nn.LayerNorm(overparameterized * state_sz),
    nn.Linear(overparameterized * state_sz, N*N),
).to(device)
future_dist = nn.Sequential( # (prev_board, state, target_board) → future_distance_sum
    # (For picking an action that leads to getting to the target board the fastest.)
    nn.Linear(N*N + state_sz + N*N, overparameterized * state_sz),
    nn.ReLU(),
    nn.LayerNorm(overparameterized * state_sz),
    SkipConnection(nn.Sequential(
        nn.Linear(overparameterized * state_sz, overparameterized * state_sz),
        nn.ReLU(),
        nn.LayerNorm(overparameterized * state_sz),
    )),
    nn.Linear(overparameterized * state_sz, N*N),
).to(device)
opt = torch.optim.Adam([*next.parameters(), *next_board.parameters(), *future_dist.parameters()], lr=1e-3)
for iters in range(50000):

    # Ultimately, to pre-train for all possible goals, we want to learn how to eventually get from anywhere to anywhere in `state`-space.
    #   Let's define our transition model as `next: (prev_state, goal) → next_state` (which also includes input/output inside of it for simplicity of our analysis). We'd like to learn good `next_state` — but where could we possibly get a good loss to do that?
    #   There exist many `state→state→…→state→state` trajectories, and we'd like to make sure that they all encounter `goal` as soon as possible. Moreover, we only really need to know one trajectory for each `(state, goal)` pair, the shortest one (which rules out the hard generative-modeling and allows using the easy prediction): if our `state`-space is a faithfully-compressed representation of inputs/outputs, then state-goals won't miss any details and thus won't have to intervene on trajectories to patch up blindspots.
    #   We need to consider a trajectory's first transition to its `next` state, and for each possible action (`next_state`), measure/learn its future's distance to the `goal`, and minimize that (this problem is known as [optimal control](http://www.scholarpedia.org/article/Optimal_control#:~:text=Optimal%20control%20is%20the%20process,to%20minimise%20a%20performance%20index.)). Succeed at this everywhere, and our map is learned.
    #   1. A neural net that would learn the future-distance: `future_dist: (state, goal) → future_distance`. Should be learned like a GAN: learned without gradient to `state` & `goal` (only its params), and minimized with gradient only to `state` & `goal` (not to its params, to not undo learning).
    #   2. The actual distance, on a trajectory? Note that our states/goals are continuous, so we can't just count transitions until equality, nor can we explicitly dream up individual trajectories to pick the shortest one (there are infinitely many, so that's either inaccurate or taking too long).
    #     - The less transitions until we're at `goal`, and the longer we stay there afterward, the better. The future distance being the *sum* of individual `dist`ances fits this quite nicely, and is easy to learn.
    #       - (With a sum, we don't have to materialize full trajectories to learn distances, we can just bootstrap from single transitions via `future_dist(prev, goal) = (future_dist(next, goal) + dist(prev, goal)).detach()`.)
    #     - `dist(x,y)`: for easy optimization, something linear, and summed-up as late as possible for a richer learning signal. L1 AKA `(x-y).abs().sum()` fits the bill.

    # TODO: Run & fix. (It looks like a neural-net reformulation of a dynamic programming problem, so it shouldn't really fail barring optimization difficulties, right?)



    # Sample a batch of trajectories (pre-deciding the target-board).
    next_board_loss, dist_pred_loss, dist_min_loss = 0, 0, 0
    state = torch.zeros(batch_size, state_sz, device=device)
    board = env_init(N, batch_size=batch_size)
    target_board = env_init(N, batch_size=batch_size)
    achieved_target = torch.full((batch_size,), False, device=device)
    distances = [] # For direct future-distance prediction, w/o bootstrapping.
    distances.append((future_dist(torch.cat((board, state, target_board), -1)), 0)) # TODO: (This is just so that we could effortlessly visualize any-action distances as an image, by setting `state=0`.)
    for u in range(unroll_len):
        # Do the RNN transition (and an environment step), `unroll_len` times.
        random = torch.randn(batch_size, state_sz, device=device)
        # Minimize the future-distance-sum by considering all 4 possible actions right here.
        #   (Minimizing by gradient descent in this environment is no bueno.)
        state = next(torch.cat((board, target_board, state, random), -1))
        sx, sy, srest = state.split((1, 1, state.shape[-1]-2), -1)
        state_candidates = [
            state,
            torch.cat((sy, sx, srest), -1),
            torch.cat((-sx, -sy, srest), -1),
            torch.cat((-sy, -sx, srest), -1),
        ]
        min_state, min_dist = None, None
        with torch.no_grad():
            for state in state_candidates:
                dist = future_dist(torch.cat((board, state, target_board), -1)).sum(-1, keepdim=True)
                if min_dist is None: min_state, min_dist = state, dist
                else:
                    mask = dist < min_dist
                    min_state = torch.where(mask, state, min_state)
                    min_dist = torch.where(mask, dist, min_dist)
        state = min_state

        prev_board, board = board, env_step(N, board, state[..., 0:2])
        achieved_target_now = (board == target_board).all(-1)
        prev_achieved_target = achieved_target
        achieved_target = achieved_target | achieved_target_now

        # Be able to predict the next board from prev & action.
        #   (Technically unnecessary here since we have `board` anyway, but why not?)
        next_board_loss = next_board_loss + (next_board(torch.cat((prev_board, state.detach()), -1)) - board).square().sum()
        # Be able to predict sum-of-future-distances from prev & target, to minimize.
        #   (Technically unnecessary here since we could just have a table from all-past-actions & current-board & target-board to distance, but at that point, why even have neural nets at all?)
        micro_dist = (board - target_board).abs()
        fut_dist_pred = future_dist(torch.cat((prev_board, state, target_board), -1))
        distances.append((fut_dist_pred, micro_dist))
        # fut_dist_targ = (future_dist(torch.cat((board, target_board), -1)) if u<unroll_len-1 else 0) + micro_dist # TODO: The next state in the middle, computed from the next board… (Bootstrap.)
        # dist_pred_loss = dist_pred_loss + (fut_dist_pred - fut_dist_targ.detach()).square().sum()
        # Minimize that sum-of-future-distances by actions.
        # for p in chain(future_dist.parameters(), next_board.parameters()): p.requires_grad_(False)
        # dist_min_loss = dist_min_loss + future_dist(torch.cat((prev_board, state, target_board), -1)).sum() # TODO:
        # for p in chain(future_dist.parameters(), next_board.parameters()): p.requires_grad_(True)

    # Predict future-distances directly. # TODO: Bootstrap instead.
    for i in reversed(range(1, len(distances))):
        distances[i-1] = (distances[i-1][0], distances[i-1][1] + distances[i][1])
    for pred, targ in distances:
        dist_pred_loss = dist_pred_loss + (pred - targ.detach()).square().sum()
    if (iters+1) % 5000 == 0: # For debugging, visualize distances from anywhere to a target.
        import matplotlib.pyplot as plt
        src = torch.eye(N*N, device=device)
        state = torch.zeros(N*N, state_sz, device=device)
        dst = torch.eye(1, N*N, device=device).expand(N*N, N*N)
        dist = future_dist(torch.cat((src, state, dst), -1)).sum(-1).reshape(N, N).detach().cpu().numpy()
        # TODO: Maybe, in this visualization, have 4 prior `state`s and pick the min between their future-distances? …Or just rely on prediction.
        plt.clf()
        plt.imshow(dist)
        plt.pause(1)
        plt.show()
    if iters == 100: clear()

    achieved_target = achieved_target.float()
    (next_board_loss + dist_pred_loss + dist_min_loss).backward()
    opt.step();  opt.zero_grad(True)
    with torch.no_grad():
        correct_frac = achieved_target.mean()
        log(0, False, next_board_loss = to_np(next_board_loss))
        log(1, False, dist_pred_loss = to_np(dist_pred_loss))
        log(2, False, dist_min_loss = to_np(dist_min_loss))
        log(3, False, correct_target_perc = to_np((correct_frac*100).round()))
        log(4, False, state_mean = to_np(state.mean()), state_std = to_np(state.std()))
        print(str(iters).rjust(6))
# TODO: Okay, what do we want to learn, building up to URL gradually?
#   - ✓ From board and action (randomly-generated) to board — EASY
#   - ✓ From board and neighboring-board (gotten via env_step with a random action) to action (the randomly-generated one) — will be so easy that it's pointless to implement.
#   - From board & target-board & extra-state & whole-output (or a random vector initially), to the next action & extra state: `next`.
#     - (Need to limit the unroll-length, or else practically everything will count as reachable.)
#     - Average-plan makes no sense because everything is connected to everything, so we need to learn the *distribution* of plans that will lead us to the target, so either:
#       - ✓ [Good: 40%] DDPM-like but speedy (not about to do thousands of steps per RNN step): make `next` self-denoising (accept its output as an input, initially a random vector), and wherever we have a loss (here, just: make less-denoised outputs predict more-denoised outputs, only in trajectories that reached the target), make predict-branches have less denoisings than stopgrad-branches to make denoising learned. Possibly, have completely separate RNN-states for different denoising levels. (Sounds quite trainable, but might just collapse diversity like in the initial experiments; maybe using CCL for prediction could help.) (Eventually removed the impl here.)
#       - ❌ [Bad: 30%, highly unstable] GAN-like: train a discriminator (from board & target-board & extra-state & 'whole-output'-randomness) of whether a trajectory will succeed (known after a whole rollout), and maximize the predicted success-probability by all `next`-steps (but not by the discriminator).
#       - ⋯ [Good: 40% with partial impl, very fast convergence] Abandon distributions, pick only their shortest paths; special-case our discrete case to see how viable this is. (Very viable: didn't implement prior-path-conditioning, so all shortest paths interfere, making 40% on 4×4 even more impressive.)
#         - (Performance deterioration is because there are always 2+ valid actions, so when they collide, predicted action goes toward 0 and causes instability before long. So, not worth caring about.) (unroll_len=1 is max-performance while the others are worse because targets are not path-dependent, which I think would be included in the final loss, but may take too much effort to manually engineer here.)
#       - ⋯ …Rethink whether we really couldn't enforce a sensible priority on which plans from a distribution we pick, because of impl and/or optim troubles…
#     - (Until we can reliably learn the next action, with a batch-size of 1, without non-transferrable tasks.)
#   - Almost-URL: learn the distribution of targets, along with distributions of plans to reach them (learning eventual-RNN-states would have been full URL).
#   - Full URL, where goal-of-state is learned too: goal:ev(state);  goal=ev(next(state, goal))