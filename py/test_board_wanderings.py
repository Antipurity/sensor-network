"""
Pretraining for all possible goals is like learning a map of how to get from anywhere to anywhere. Here, we gradually build up to the full formulation in an easily-comprehensible and easily-visualizable environment.

We have a square board here, and 1 agent that can only walk it. If we can't learn a map here, then how can we expect to learn a map in more complex environments?

NO RL machinery! Lessons must be transferrable all the way.
"""



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
    def forward(self, x):
        y = self.fn(x)
        return y + x if x.shape == y.shape else y



N, batch_size = 8, 1000
state_sz = 64
overparameterized = 4

unroll_len, denoising_levels = N*2, 3



next = nn.Sequential( # (board, target_board, state, output) → output_state
    # (`output` is for sampling from the action distribution on each step, initially a random vector in each iteration, denoised one or more times.)
    # (The actions are simply sliced from `output_state`.)
    nn.Linear(N*N + N*N + state_sz + state_sz, overparameterized * state_sz),
    SkipConnection(nn.Sequential(
        nn.LayerNorm(overparameterized * state_sz),
        nn.ReLU(),
        nn.Linear(overparameterized * state_sz, overparameterized * state_sz),
    )),
    SkipConnection(nn.Sequential(
        nn.LayerNorm(overparameterized * state_sz),
        nn.ReLU(),
        nn.Linear(overparameterized * state_sz, overparameterized * state_sz),
    )),
    nn.LayerNorm(overparameterized * state_sz),
    nn.ReLU(),
    nn.Linear(overparameterized * state_sz, state_sz),
).to(device)
opt = torch.optim.Adam(next.parameters(), lr=1e-4)
for iters in range(50000):
    # TODO: Run & fix. (Maybe not as good as a GAN, but maybe we can actually get the correct-target-percentage to go up.)
    # Sample a batch of trajectories (pre-deciding the target-board), accumulating the denoising loss, and minimizing it wherever we've reached the target.
    L2 = 0
    state = torch.zeros(batch_size, state_sz, device=device) # For now, only use the max-denoised state.
    board = env_init(N, batch_size=batch_size)
    target_board = env_init(N, batch_size=batch_size)
    achieved_target = torch.full((batch_size,), False, device=device)
    for u in range(unroll_len):
        # Do the RNN transition (and an environment step), `unroll_len` times.
        nexts = [torch.randn(batch_size, state_sz, device=device)]
        for _ in range(denoising_levels):
            # Denoise the next-state, `denoising_levels` times.
            #   (Diffusion models learn to reverse gradual noising of samples, with thousands of denoising steps. That's too slow in an RNN, so we learn the reversing directly.)
            #   (…Would this just learn the fixed-points and then stop producing any diversity?… I think it would, since the only randomness is at the beginning, which we learn to ignore by design… Would diffusion models fare better…? Probably; but would a GAN? I think a GAN is pretty much the same, honestly, just with a learned target (changes in which might actually save us).)
            nexts.append(next(torch.cat((board, target_board, state, nexts[-1]), -1)))
        for noised in nexts[1:-1]:
            # (Preserve the batch dimension, so that we could select which to minimize.)
            L2 = L2 + (noised - nexts[1:-1]).square().sum(-1)
        state = nexts[-1]
        board = env_step(N, board, state[..., 0:2])
        achieved_target = achieved_target | (board == target_board).all(-1)
    L2 = (L2 * achieved_target.float()).sum()
    L2.backward()
    opt.step();  opt.zero_grad(True)
    with torch.no_grad():
        correct_frac = achieved_target.float().sum() / batch_size
        print(str(iters).rjust(6), 'L2', str(L2.detach().cpu().numpy()).ljust(11), 'correct target', str((correct_frac*100).cpu().numpy())+'%')
# TODO: Okay, what do we want to learn, building up to URL gradually?
#   - ✓ From board and action (randomly-generated) to board — EASY
#   - ✓ From board and neighboring-board (gotten via env_step with a random action) to action (the randomly-generated one) — will be so easy that it's pointless to implement.
#   - From board & target-board & extra-state & whole-output (or a random vector initially), to the next action & extra state: `next`.
#     - (Need to limit the unroll-length, or else practically everything will count as reachable.)
#     - Average-plan makes no sense because everything is connected to everything, so we need to learn the *distribution* of plans that will lead us to the target, so either:
#       - ⋯ DDPM-like but speedy (not about to do thousands of steps per RNN step): make `next` self-denoising (accept its output as an input, initially a random vector), and wherever we have a loss (here, just: make less-denoised outputs predict more-denoised outputs, only in trajectories that reached the target), make predict-branches have less denoisings than stopgrad-branches to make denoising learned. Possibly, have completely separate RNN-states for different denoising levels. (Sounds quite trainable, but might just collapse diversity like in the initial experiments; maybe using CCL for prediction could help.)
#       - ⋯ GAN-like: train a discriminator (from board & target-board & extra-state & 'whole-output'-randomness) of whether a trajectory will succeed (known after a whole rollout), and maximize the predicted success-probability by all `next`-steps (but not by the discriminator).
#   - Almost-URL: learn the distribution of targets, along with distributions of plans to reach them (learning eventual-RNN-states would have been full URL).
#   - Full URL, where goal-of-state is learned too: goal:ev(state);  goal=ev(next(state, goal))