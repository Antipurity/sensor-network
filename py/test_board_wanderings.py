"""
Pretraining for all possible goals is like learning a map of how to get from anywhere to anywhere. Here, we gradually build up to the full formulation in an easily-comprehensible and easily-visualizable environment.

We have a square board here, and 1 agent that can only walk it. If we can't learn a map here, then how can we expect to learn a map in more complex environments?

NO RL machinery! Lessons must be transferrable all the way.
"""



from model.momentum_copy import MomentumCopy
from model.log import log



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
state_sz = 2
overparameterized = 64

unroll_len, denoising_levels = 2*N-2, 1



next = nn.Sequential( # (board, target_board, state, output) → output_state
    # (`output` is for sampling from the action distribution on each step, initially a random vector in each iteration, denoised one or more times.)
    # (The actions are simply sliced from `output_state`.)
    nn.Linear(N*N + N*N + state_sz + state_sz, overparameterized * state_sz),
    nn.ReLU(),
    nn.LayerNorm(overparameterized * state_sz),
    nn.Linear(overparameterized * state_sz, state_sz),
).to(device)
next_board = nn.Sequential( # (board, state) → next_board (state AKA action)
    # (So that goals don't have to be non-human-comprehensible.)
    nn.Linear(N*N + state_sz, overparameterized * state_sz),
    nn.ReLU(),
    nn.LayerNorm(overparameterized * state_sz),
    nn.Linear(overparameterized * state_sz, N*N),
).to(device)
future_dist = nn.Sequential( # (next_board, target_board) → future_distance_sum
    # (For picking an action that leads to getting to the target board the fastest.)
    nn.Linear(N*N + N*N, overparameterized * state_sz),
    nn.ReLU(),
    nn.LayerNorm(overparameterized * state_sz),
    nn.Linear(overparameterized * state_sz, N*N),
).to(device)
opt = torch.optim.Adam([*next.parameters(), *next_board.parameters(), *future_dist.parameters()], lr=1e-3)
for iters in range(50000):

    # Ultimately, to pre-train for all possible goals, we want to learn how to eventually get from anywhere to anywhere in `state`-space.
    #   Let's define our transition model as `next: (prev_state, goal) → next_state` (which also includes input/output inside of it for simplicity of our analysis). We'd like to learn good `next_state` — but where could we possibly get a good loss to do that?
    #   There exist many `state→state→…→state→state` trajectories, and we'd like to make sure that they all encounter `goal` as soon as possible. Moreover, we only really need to know one trajectory for each `(state, goal)` pair, the shortest one (which rules out the hard generative-modeling and allows using the easy prediction): if our `state`-space is a faithfully-compressed representation of inputs/outputs, then state-goals won't miss any details and thus won't have to intervene on trajectories to patch up blindspots.
    #   We need to consider a trajectory's first transition to its `next` state, and for each possible action (`next_state`), measure/learn its future's distance to the `goal`, and minimize that. Succeed at this everywhere, and our map is learned.
    #   1. A neural net that would learn the future-distance: `future_dist: (state, goal) → future_distance`. Should be learned like a GAN: learned without gradient to `state` & `goal` (only its params), and minimized with gradient only to `state` & `goal` (not to its params, to not undo learning).
    #   2. The actual distance, on a trajectory? Note that our states/goals are continuous, so we can't just count transitions until equality, nor can we explicitly dream up individual trajectories to pick the shortest one (there are infinitely many, so that's either inaccurate or taking too long).
    #     - The less transitions until we're at `goal`, and the longer we stay there afterward, the better. The future distance being the *sum* of individual `dist`ances fits this quite nicely, and is easy to learn.
    #       - (With a sum, we don't have to materialize full trajectories to learn distances, we can just bootstrap from single transitions via `future_dist(prev, goal) = (future_dist(next, goal) + dist(prev, goal)).detach()`.)
    #     - `dist(x,y)`: for easy optimization, something linear, and summed-up as late as possible for a richer learning signal. L1 AKA `(x-y).abs().sum()` fits the bill.

    # TODO: Run & fix. (It looks like a neural-net reformulation of a dynamic programming problem, so it shouldn't really fail barring optimization difficulties, right?)
    #   …Why is the distance-prediction loss still high?
    #   TODO: Figure out why it doesn't work:
    #     TODO: Try predicting the sum of future distances directly, without bootstrapping. (See whether we can get its loss to 0, which bootstrapping doesn't seem to be able to do.)
    #       TODO: If it fails, try visualizing fixed-target predicted distances.
    #       …Still doesn't converge, so maybe, the problem is in adjusting the picked action?
    #       …Okay, how to construct that image?
    #     TODO: Try printing a trajectory's distance predictions. (See whether they decrease over time as they should.)
    #     TODO: Try turning our actions into next-board-state proposals, where the one with the highest value in one of the 4 neighbors wins? (Then, we won't need `next_board`.)



    # Sample a batch of trajectories (pre-deciding the target-board), accumulating the denoising loss, and minimizing it wherever we've reached the target.
    denoising_loss, next_board_loss, dist_pred_loss, dist_min_loss = 0, 0, 0, 0
    state = torch.zeros(batch_size, state_sz, device=device)
    board = env_init(N, batch_size=batch_size)
    target_board = env_init(N, batch_size=batch_size)
    target_ind = N*N*board.argmax(-1) + target_board.argmax(-1) # For special-casing discrete targets (see below).
    states = [] # For special-casing discrete targets (see below).
    achieved_target = torch.full((batch_size,), False, device=device)
    path_len = torch.full((batch_size,), float(unroll_len), device=device) # For special-casing discrete targets (see below).
    distances = [] # For direct future-distance prediction, w/o bootstrapping.
    for u in range(unroll_len):
        # Do the RNN transition (and an environment step), `unroll_len` times.
        nexts = [torch.randn(batch_size, state_sz, device=device)]
        for lvl in range(denoising_levels):
            # Denoise the next-state, `denoising_levels` times.
            #   (Diffusion models learn to reverse gradual noising of samples, with thousands of denoising steps. That's too slow in an RNN, so we learn the reversing directly.)
            #   (Doesn't seem to work, though. Maybe a GAN would have better luck.)
            nexts.append(next(torch.cat((board, target_board, state, nexts[-1]), -1)))
            input = torch.cat((board, target_board, state, nexts[-1]), -1)
        first, last = nexts[0], nexts[-1]
        for lvl, noised in enumerate(nexts[1:-1]):
            # (Preserve the batch dimension, so that we could select which to minimize and thus model the distribution of.)
            target = last
            denoising_loss = denoising_loss + (noised - target.detach()).square().sum(-1)
        state = nexts[-1]
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
        fut_dist_pred = future_dist(torch.cat((prev_board, target_board), -1))
        distances.append((fut_dist_pred, micro_dist))
        # fut_dist_targ = (future_dist(torch.cat((board, target_board), -1)) if u<unroll_len-1 else 0) + micro_dist
        # dist_pred_loss = dist_pred_loss + (fut_dist_pred - fut_dist_targ.detach()).square().sum()
        # Minimize that sum-of-future-distances by actions.
        for p in chain(future_dist.parameters(), next_board.parameters()): p.requires_grad_(False)
        next_board_is = next_board(torch.cat((prev_board, state), -1))
        dist_min_loss = dist_min_loss + future_dist(torch.cat((next_board_is, target_board.detach()), -1)).sum()
        for p in chain(future_dist.parameters(), next_board.parameters()): p.requires_grad_(True)

        # Special-casing for this discrete-targets-only environment, to see how well directly predicting the shortest path works.
        #   (Ideally, should have a smooth loss on states/actions, able to handle continuous targets gracefully.)
        path_len = torch.where(~prev_achieved_target & achieved_target_now, torch.full_like(path_len, u), path_len)
        states.append(state)
    matching_path_len = torch.where(
        (target_ind.unsqueeze(-1) == target_ind) & achieved_target,
        path_len,
        torch.full((batch_size, batch_size), float(unroll_len+1), device=device) - torch.eye(batch_size, device=device),
    )
    nearest_target = matching_path_len.argmin(-1)
    for st in states: # Yes, finally we directly predict the shortest same-target actions.
        tt = torch.index_select(st, 0, nearest_target)
        # denoising_loss = denoising_loss + (st - tt.detach()).square().sum()
        #   Hopefully, this special-casing won't be necessary anymore, and we can delete it entirely.
    # Predict future-distances directly.
    for i in reversed(range(1, len(distances))):
        distances[i-1] = (distances[i-1][0], distances[i-1][1] + distances[i][1])
    for pred, targ in distances:
        dist_pred_loss = dist_pred_loss + (pred - targ.detach()).square().sum()
    if (iters+1) % 1000 == 0: # For debugging, visualize distances from anywhere to a target.
        # TODO: ...This ain't right... Why?...
        import matplotlib.pyplot as plt
        src = torch.eye(N*N, device=device)
        dst = env_init(N, 1).expand(N*N, N*N)
        dist = future_dist(torch.cat((src, dst), -1)).sum(-1).reshape(N, N).detach().cpu().numpy()
        plt.clf()
        plt.imshow(dist)
        plt.pause(1)
        plt.show()

    achieved_target = achieved_target.float()
    # denoising_loss = denoising_loss * achieved_target # (This is only for denoising.)
    if isinstance(denoising_loss, torch.Tensor): denoising_loss = denoising_loss.sum()
    (denoising_loss + next_board_loss + dist_pred_loss + dist_min_loss).backward()
    opt.step();  opt.zero_grad(True)
    with torch.no_grad():
        correct_frac = achieved_target.mean()
        log(0, False, denoising_loss = to_np(denoising_loss))
        log(1, False, next_board_loss = to_np(next_board_loss))
        log(2, False, dist_pred_loss = to_np(dist_pred_loss))
        log(3, False, dist_min_loss = to_np(dist_min_loss))
        log(4, False, correct_target_perc = to_np((correct_frac*100).round()))
        log(5, False, state_mean = to_np(state.mean()), state_std = to_np(state.std()))
        print(str(iters).rjust(6))
# TODO: Okay, what do we want to learn, building up to URL gradually?
#   - ✓ From board and action (randomly-generated) to board — EASY
#   - ✓ From board and neighboring-board (gotten via env_step with a random action) to action (the randomly-generated one) — will be so easy that it's pointless to implement.
#   - From board & target-board & extra-state & whole-output (or a random vector initially), to the next action & extra state: `next`.
#     - (Need to limit the unroll-length, or else practically everything will count as reachable.)
#     - Average-plan makes no sense because everything is connected to everything, so we need to learn the *distribution* of plans that will lead us to the target, so either:
#       - ✓ [Good: 40%] DDPM-like but speedy (not about to do thousands of steps per RNN step): make `next` self-denoising (accept its output as an input, initially a random vector), and wherever we have a loss (here, just: make less-denoised outputs predict more-denoised outputs, only in trajectories that reached the target), make predict-branches have less denoisings than stopgrad-branches to make denoising learned. Possibly, have completely separate RNN-states for different denoising levels. (Sounds quite trainable, but might just collapse diversity like in the initial experiments; maybe using CCL for prediction could help.)
#       - ❌ [Bad: 30%, highly unstable] GAN-like: train a discriminator (from board & target-board & extra-state & 'whole-output'-randomness) of whether a trajectory will succeed (known after a whole rollout), and maximize the predicted success-probability by all `next`-steps (but not by the discriminator).
#       - ⋯ [Good: 40% with partial impl, very fast convergence] Abandon distributions, pick only their shortest paths; special-case our discrete case to see how viable this is. (Very viable: didn't implement prior-path-conditioning, so all shortest paths interfere, making 40% on 4×4 even more impressive.)
#         - (Performance deterioration is because there are always 2+ valid actions, so when they collide, predicted action goes toward 0 and causes instability before long. So, not worth caring about.) (unroll_len=1 is max-performance while the others are worse because targets are not path-dependent, which I think would be included in the final loss, but may take too much effort to manually engineer here.)
#       - ⋯ …Rethink whether we really couldn't enforce a sensible priority on which plans from a distribution we pick, because of impl and/or optim troubles…
#     - (Until we can reliably learn the next action, with a batch-size of 1, without non-transferrable tasks.)
#   - Almost-URL: learn the distribution of targets, along with distributions of plans to reach them (learning eventual-RNN-states would have been full URL).
#   - Full URL, where goal-of-state is learned too: goal:ev(state);  goal=ev(next(state, goal))