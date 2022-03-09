"""
Pretraining for all possible goals is like learning a map of how to get from anywhere to anywhere. Here, we gradually build up to the full formulation in an easily-comprehensible and easily-visualizable environment.

We have a square board here, and 1 agent that can only walk it. If we can't learn a map here, then how can we expect to learn a map in more complex environments?

NO RL machinery! Lessons must be transferrable all the way.
"""



from model.momentum_copy import MomentumCopy



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



# (TODO: This only worked once, with super-simple hyperparams, can't reproduce, so should try a GAN.)
N, batch_size = 2, 100
state_sz = 2
overparameterized = 40

unroll_len, denoising_levels = 1, 1
# unroll_len, denoising_levels = 2*N-2, 2



next = nn.Sequential( # (board, target_board, state, output) → output_state
    # (`output` is for sampling from the action distribution on each step, initially a random vector in each iteration, denoised one or more times.)
    # (The actions are simply sliced from `output_state`.)
    nn.Linear(N*N + N*N + state_sz + state_sz, overparameterized * state_sz),
    # SkipConnection(nn.Sequential(
    #     nn.LayerNorm(overparameterized * state_sz),
    #     nn.ReLU(),
    #     nn.Linear(overparameterized * state_sz, overparameterized * state_sz),
    # )),
    nn.LayerNorm(overparameterized * state_sz),
    nn.ReLU(),
    nn.Linear(overparameterized * state_sz, state_sz),
).to(device)
next_discriminator = nn.Sequential( # (board, target_board, state, output) → will_reach_target
    nn.Linear(N*N + N*N + state_sz + state_sz, overparameterized * state_sz),
    SkipConnection(nn.Sequential(
        nn.LayerNorm(overparameterized * state_sz),
        nn.ReLU(),
        nn.Linear(overparameterized * state_sz, overparameterized * state_sz),
    )),
    nn.LayerNorm(overparameterized * state_sz),
    nn.ReLU(),
    nn.Linear(overparameterized * state_sz, 1, bias=False),
    nn.Softsign(),
).to(device)
next_discriminator_copy = MomentumCopy(next_discriminator, .99)
opt = torch.optim.Adam([*next.parameters(), *next_discriminator.parameters()], lr=1e-3)
frozen_state = torch.randn(batch_size, state_sz, device=device) # TODO: ...This is freaky: why can we not learn to be better than random chance even with frozen generator state?!
target_board = env_init(N, batch_size=batch_size) # TODO: ...What if there's only ever one target... Or, only ever one initial-state?... (WHERE DID WE GO WRONG)
for iters in range(50000):
    # TODO: Run & fix. (Maybe not as good as a GAN, but maybe we can actually get the correct-target-percentage to go up.)
    # Sample a batch of trajectories (pre-deciding the target-board), accumulating the denoising loss, and minimizing it wherever we've reached the target.
    L2 = 0
    state = torch.zeros(batch_size, state_sz, device=device)
    board = env_init(N, batch_size=batch_size)
    # target_board = env_init(N, batch_size=batch_size) # TODO: ...What if there's only ever one target... Or, only ever one initial-state?... (WHERE DID WE GO WRONG)
    achieved_target = torch.full((batch_size,), False, device=device)
    target_reachable = []
    for u in range(unroll_len):
        # Do the RNN transition (and an environment step), `unroll_len` times.
        nexts = [frozen_state]
        # nexts = [torch.randn(batch_size, state_sz, device=device)] # TODO:
        for lvl in range(denoising_levels):
            # Denoise the next-state, `denoising_levels` times.
            #   (Diffusion models learn to reverse gradual noising of samples, with thousands of denoising steps. That's too slow in an RNN, so we learn the reversing directly.)
            #   (Doesn't seem to work, though. Maybe a GAN would have better luck.)
            nexts.append(next(torch.cat((board, target_board, state, nexts[-1]), -1)))
            input = torch.cat((board, target_board, state, nexts[-1]), -1)
            # print(input[0].detach().cpu().numpy()) # TODO:
            target_reachable.append(next_discriminator(input.detach()).sum(-1))
            print(target_reachable[-1].mean().detach().cpu().numpy()) # TODO: ...Wait, why is it all negative? Quite suspicious, honestly...
            #   TODO: ...And why is this mean going toward `1` when the correct-target percentage is 25% and such?!
            # L2 = L2 + (next_discriminator_copy(input) - 1).square().sum() # TODO: ...Why is even a GAN not working... In fact, how are we even reaching 20K in discriminator loss?... WHAT DID WE IMPLEMENT WRONG?!
            #   ...It doesn't make any sense for the model to be completely unable to learn whether single steps will lead to success, right? SO WHERE ARE THE BUGS
        for lvl, noised in enumerate(nexts[1:-1]):
            # (Preserve the batch dimension, so that we could select which to minimize.)
            L2 = L2 + (noised - nexts[-1]).square().sum(-1)
        state = nexts[-1]
        board = env_step(N, board, state[..., 0:2])
        achieved_target = achieved_target | (board == target_board).all(-1)
    achieved_target = achieved_target.float()
    L2 = (L2 * achieved_target).sum()
    L22 = 0
    for reachable in target_reachable:
        print('                                                                 discriminator guessed correctly', str((((reachable > 0).float() == achieved_target).float().mean()*100).round().detach().cpu().numpy())+'%', '>0', str(((reachable > 0).float().mean()*100).round().detach().cpu().numpy())+'%') # TODO: ...How can it possibly be such a high percentage from the very start?! ...Probably all-OK or all-non-OK, since percentages match. ...But why is that even so...
        #   Why is nearly every run initialized to be discriminated as an extreme?! AND WHY CAN'T IT EVEN CONVERGE ON THIS TRIVIAL TASK, OF FITTING 4 FIXED INPUTS TO 4 FIXED OUTPUTS?!
        L22 = L22 + (reachable - (achieved_target*2-1)).square().sum() # TODO: ...Why can't we learn this, even still??
        #   TODO: ...Wait, something is still very wrong: why is the mean very decidedly going up, when `achieved_targets.mean()` is very clearly ~25%?! ...And why is putting `-` instead of `+` here not repel `reachable` from 0 but instead always make it negative even if it was positive at the beginning...
    (L2 + L22).backward()
    opt.step();  opt.zero_grad(True);  next_discriminator_copy.update()
    with torch.no_grad():
        correct_frac = achieved_target.mean()
        print(
            str(iters).rjust(6),
            'denoising L2', str(L2.detach().cpu().numpy()).ljust(13),
            'discriminator L2', str(L22.detach().cpu().numpy()).ljust(13),
            'correct target', str((correct_frac*100).round().cpu().numpy())+'%',
        )
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