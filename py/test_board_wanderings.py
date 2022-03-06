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
    next_board = torch.where(
        is_x,
        torch.where(xy[..., :1] > 0, board_px, board_mx),
        torch.where(xy[..., 1:] > 0, board_py, board_my),
    )
    return next_board



class SkipConnection(nn.Module):
    def __init__(self, *fn): super().__init__();  self.fn = nn.Sequential(*fn)
    def forward(self, x):
        y = self.fn(x)
        return y + x if x.shape == y.shape else y



N, batch_size = 8, 1000
overparameterized = 4
next_board = nn.Sequential(
    nn.Linear(N*N + 2, N*N*overparameterized),
    SkipConnection(nn.Sequential(
        nn.LayerNorm(N*N*overparameterized),
        nn.ReLU(),
        nn.Linear(N*N*overparameterized, N*N*overparameterized),
    )),
    nn.LayerNorm(N*N*overparameterized),
    nn.ReLU(),
    nn.Linear(N*N*overparameterized, N*N),
).to(device)
opt = torch.optim.Adam(next_board.parameters(), lr=1e-2)
for iters in range(50000):
    board = env_init(N, batch_size)
    action = torch.randn(batch_size, 2, device=device)
    real_next = env_step(N, board, action)
    pred_next = next_board(torch.cat((board, action), -1))
    loss = (real_next - pred_next).square().sum()
    loss.backward();  opt.step();  opt.zero_grad()
    with torch.no_grad():
        real_pos, pred_pos = real_next.argmax(-1), pred_next.argmax(-1)
        correct_perc = (real_pos == pred_pos).float().sum() / batch_size
        print(str(iters).rjust(6), 'L2', str(loss.detach().cpu().numpy()).ljust(11), 'correct position', str((correct_perc*100).cpu().numpy())+'%')
# TODO: Okay, what do we want to learn, building up to URL gradually?
#   - ✓ From board and action (randomly-generated) to board — EASY
#   - From board and neighboring-board (gotten via env_step with a random action) to action (the randomly-generated one)
#   - From board and target-board (& extra state) to the next action (& extra state) (can't average, need to learn the *distribution* of plans that will lead us to the target)
#     - Would conditioning on extra RNN-state help us disambiguate trajectories?… In effect, we'd be training a diffusion model, from initial state to a sample-of the distribution of correct trajectories… Sounds tricky, but also like the only approach that could work without RL-machinery.
#       - …But how do we actually learn this… What's the actual loss…
#   - URL, where target-of-target is learned too: goal:ev(state);  goal=ev(next(state, goal))
#   (…When we put it all like that, it all seems quite doable, if difficult. Distributions are key.)