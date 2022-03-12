"""
Pretraining for all possible goals is like learning a map of how to get from anywhere to anywhere. Here, we gradually build up to the full formulation in an easily-comprehensible and easily-visualizable environment.

We have a square board here, and 1 agent that can only walk it. If we can't learn a map here, then how can we expect to learn a map in more complex environments?

NO RL machinery! Lessons must be transferrable all the way.
"""



from model.momentum_copy import MomentumCopy
from model.log import log



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



N, batch_size = 4, 1000
state_sz = 2
overparameterized = 32

unroll_len, denoising_levels = 2*N-2, 1



next = nn.Sequential( # (board, target_board, state, output) → output_state
    # (`output` is for sampling from the action distribution on each step, initially a random vector in each iteration, denoised one or more times.)
    # (The actions are simply sliced from `output_state`.)
    nn.Linear(N*N + N*N + state_sz + state_sz, overparameterized * state_sz),
    nn.ReLU(),
    nn.LayerNorm(overparameterized * state_sz),
    nn.Linear(overparameterized * state_sz, state_sz),
).to(device)
opt = torch.optim.Adam([*next.parameters()], lr=1e-3)
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
    #  TODO: Neural nets for:
    #    next: (prev_board, target_board, _, random) → action (the same as before)
    #    next_board: (prev_board, action) → next_board (so that states & goals can be in the easily-tracable )
    #    future_dist: (next_board, target_board) → future_distance_sum
    #  TODO: Every time we call `env_step` to get `board`, add `(next_board(torch.cat((prev_board, state.detach()), -1)) - board).square().sum()` to the loss.
    #  TODO: On every `next` transition, add `(future_dist(cat(prev_board, target_board)) - ((future_dist(cat(next_board, target_board)) if u<unroll_len-1 else 0) + (prev_board - target_board).abs()).detach()).square().sum()` to the loss.
    #  TODO: On every `next` transition, freeze `future_dist`'s parameters and add `future_dist(cat(board, target_board.detach())).sum()` to the loss and unfreeze `future_dist`'s parameters.
    #  TODO: Log all parts of the loss separately, so that we can see which parts are failing.



    # Sample a batch of trajectories (pre-deciding the target-board), accumulating the denoising loss, and minimizing it wherever we've reached the target.
    L2 = 0
    state = torch.zeros(batch_size, state_sz, device=device)
    board = env_init(N, batch_size=batch_size)
    target_board = env_init(N, batch_size=batch_size)
    target_ind = N*N*board.argmax(-1) + target_board.argmax(-1) # For special-casing discrete targets (see below).
    states = [] # For special-casing discrete targets (see below).
    achieved_target = torch.full((batch_size,), False, device=device)
    path_len = torch.full((batch_size,), float(unroll_len), device=device) # For special-casing discrete targets (see below).
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
            L2 = L2 + (noised - target.detach()).square().sum(-1)
        state = nexts[-1]
        board = env_step(N, board, state[..., 0:2])
        achieved_target_now = (board == target_board).all(-1)
        prev_achieved_target = achieved_target
        achieved_target = achieved_target | achieved_target_now

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
        L2 = L2 + (st - tt.detach()).square().sum()

    achieved_target = achieved_target.float()
    # L2 = L2 * achieved_target # (This is only for denoising.)
    L2.sum().backward()
    opt.step();  opt.zero_grad(True)
    with torch.no_grad():
        correct_frac = achieved_target.mean()
        log(0, False, L2 = L2.detach().cpu().numpy())
        log(1, False, correct_target_perc = (correct_frac*100).round().cpu().numpy())
        log(2, False, state_mean = state.mean().cpu().numpy(), state_std = state.std().cpu().numpy())
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