"""
TODO: No more RL. We've *seen* where we've been, so it should be very trivial to learn how to get anywhere, right? (Though, might need RL to pick shortest paths anyway, but it shouldn't be the deciding factor.)
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
def env_step(N, board, action): # → board
    """Given a one-hot board encoding and a 1-number action, returns a new board state, cyclically-shifted in the correct direction. Batchable.

    (The action number represents one of 4 actions: -∞…-.5 for +x, -.5…0 for -x, 0….5 for +y, .5…+∞ for -y.)

    (The 2-number scheme that relied on the max-magnitude direction was abandoned, because it played poorly with action averaging.)"""
    board = board.reshape(*board.shape[:-1], N, N)
    sh = [*board.shape[:-2], N*N]
    board_px = torch.cat((board[..., :, -1:], board[..., :, :-1]), -1).reshape(sh)
    board_py = torch.cat((board[..., -1:, :], board[..., :-1, :]), -2).reshape(sh)
    board_mx = torch.cat((board[..., :, 1:], board[..., :, :1], ), -1).reshape(sh)
    board_my = torch.cat((board[..., 1:, :], board[..., :1, :], ), -2).reshape(sh)
    return torch.where(
        action < 0.,
        torch.where(action < -.5, board_px, board_mx),
        torch.where(action < .5, board_py, board_my),
    )



class SkipConnection(nn.Module):
    def __init__(self, *fn): super().__init__();  self.fn = nn.Sequential(*fn)
    def forward(self, x): return self.fn(x) + x
def to_np(x): return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
def cat(*a, dim=-1): return torch.cat(a, dim)



N, batch_size = 4, 100 # TODO: Even like this, 70% is quite reachable. But can we reach 100%, by extending whole trajectories back?
#   TODO: N=4… Wait, is increasing THIS the reason why we can't learn?…
action_sz = 64

unroll_len = N
grad_min = False # With our 4 discrete actions, grad-min just doesn't work no matter what we try.
#   (Grad-min assumes a smooth landscape. Discrete actions are not smooth.)

replay_buffer = [None] * 1024
updates_per_unroll = N

bootstrap_discount = torch.tensor([.99], device=device)
#   Bootstrapping is `f(next) = THIS * f(prev) + local_metric(next)`
#   (Predicting many discounts at once doesn't help.)



act = nn.Sequential( # (prev_board, target) → action
    nn.Linear(N*N + N*N, action_sz), nn.ReLU(), nn.LayerNorm(action_sz),
    SkipConnection(nn.Linear(action_sz, action_sz), nn.ReLU(), nn.LayerNorm(action_sz)),
    nn.Linear(action_sz, action_sz), nn.LayerNorm(action_sz),
).to(device)
future_dist = nn.Sequential( # (prev_board, action, target) → future_distance_sum
    # (For picking an action that leads to getting to the target board the fastest.)
    nn.Linear(N*N + action_sz + N*N, action_sz), nn.ReLU(), nn.LayerNorm(action_sz),
    SkipConnection(nn.Linear(action_sz, action_sz), nn.ReLU(), nn.LayerNorm(action_sz)),
    nn.Linear(action_sz, 1),
).to(device)
opt = torch.optim.Adam([*act.parameters(), *future_dist.parameters()], lr=1e-3)

def show_dist(plt, key):
    with torch.no_grad():
        board = torch.eye(N*N, N*N, device=device)
        target = torch.eye(1, N*N, device=device).expand(N*N, N*N)
        action = act(cat(board, target))
        dist = future_dist(cat(board, action, target)).reshape(N, N)
        plt.imshow(dist.cpu().numpy(), label=key)

for iters in range(50000):
    # Sample a batch of trajectories.
    action = torch.zeros(batch_size, action_sz, device=device)
    board = env_init(N, batch_size=batch_size)
    reached = torch.full((batch_size, 1), False, device=device)
    with torch.no_grad():
        # First pick the target to go to.
        #   (From tests, in this trivial env, performance is the same even if we re-pick the target at each step.)
        # target = random.choice(replay_buffer)
        # target = target[2] if target is not None else torch.zeros(batch_size, N*N, device=device)
        target = env_init(N, batch_size=batch_size) # TODO:
        for u in range(unroll_len):
            # Do the RNN transition (and an environment step), `unroll_len` times.
            zeros = torch.zeros(batch_size, action_sz, device=device)
            randn = torch.randn(batch_size, action_sz, device=device)
            prev_board, prev_action = board, action
            # Minimize the future-distance-sum by considering all 4 possible actions right here.
            #   (Minimizing by gradient descent in this environment is no bueno.)
            action = act(cat(prev_board, target))
            if iters % 100 < 50 and random.randint(1, 10) <= 3: action = torch.randn(batch_size, action_sz, device=device)
            #   (Having this is very much not ideal, but it does actually ensure that the RNN explores a lot.)

            board = env_step(N, prev_board, action[..., 0:1])

            reached |= (board == target).all(-1, keepdim=True)

            index = (iters*unroll_len + u) % len(replay_buffer)
            replay_buffer[index] = (prev_board, prev_action.detach(), board, action.detach(), randn, target)

    # Replay from the buffer. (Needs Python 3.6+ for convenience.)
    choices = [c for c in random.choices(replay_buffer, k=updates_per_unroll) if c is not None]
    if len(choices):
        prev_board = torch.cat([c[0] for c in choices], 0)
        prev_action = torch.cat([c[1] for c in choices], 0)
        board = torch.cat([c[2] for c in choices], 0)
        action = torch.cat([c[3] for c in choices], 0)
        randn = torch.cat([c[4] for c in choices], 0)
        target = torch.cat([c[5] for c in choices], 0)

        # target = torch.where( # This actually seems to improve convergence speed 2×.
        #     #   (`prev_board` does not. More/less than 50% of the time, does not.)
        #     #   (Possibly, by doing this we're teaching the net how to stay at the goal once reached, which can be done many times in expectation and thus have a disproportionate effect compared to random goals.)
        #     torch.rand(board.shape[0], 1, device=device) < .5,
        #     board,
        #     board[torch.randperm(board.shape[0], device=device)],
        # )
        target = board[torch.randperm(board.shape[0], device=device)]

        zeros = torch.zeros(board.shape[0], action_sz, device=device)

        # TODO: Run this.
        # TODO: Why isn't it working?
        #   TODO: Why does N=4 need 5k epochs to start making any progress and 10k epochs to get to 90%? It used to be much better.
        #   TODO: Why is N=8 unfeasibly slow?

        # TODO: …Should we try removing `action` from `future_dist`, to see whether this was fundamental?

        # TODO: …Would adding `S`-step returns (making the predicted-distance the min of single-step-dist and distant-dist, for each step in a trajectory, starting with the second-last one) help with convergence speed? It should make it `S` times more efficient, especially initially, right?



        # If `board` is what we wanted, well, we're in luck: `future_dist(prev_board, action, board) = 0`.
        ends_here_loss = future_dist(cat(prev_board, action, board)).square().mean(0).sum()

        # Bootstrapping: `future_dist(prev) = 1 + future_dist(next)*p`
        # TODO: Can we only do this if target!=board, for faster convergence?
        prev_dist = future_dist(cat(prev_board, action, target.detach()))
        next_action = act(cat(board, target))
        for p in future_dist.parameters(): p.requires_grad_(False)
        dist = future_dist(cat(board, next_action, target.detach())) # TODO: …What if the target was differentiable too?… No reason not to, since the distance is tethered at 0 at final states, right?
        for p in future_dist.parameters(): p.requires_grad_(True)
        prev_dist_targ = 1 + dist * bootstrap_discount
        prev_dist_targ = torch.where((target != board).any(-1, keepdim=True), prev_dist_targ, prev_dist)
        #   TODO: …This line makes no difference to convergence speed.
        dist_pred_loss = (prev_dist - prev_dist_targ.detach()).square().mean(0).sum()

        # Min-dist self-imitation, by `act`.
        #   (A lot like [SIL.](https://arxiv.org/abs/1806.05635))
        action2 = act(cat(prev_board, target))
        prev_dist2 = future_dist(cat(prev_board, action2, target))
        with torch.no_grad():
            # Gate by min-distance.
            target_action = torch.where(prev_dist < prev_dist2, action, action2) # TODO: +.5 maybe?…
            #   TODO: Is self-imitation interfering with itself, since there are always at least 2 same-distance actions to take?
            #     Why is the distance plot so non-smooth until late in convergence (10k epochs)?
        self_imitation_loss = (action2 - target_action).square().sum()

        # Grad-min of actions.
        # dist_min_loss = dist.mean(0).sum()
        # if not grad_min: dist_min_loss = dist_min_loss.detach()

        if iters == 1000: clear()

        (ends_here_loss + dist_pred_loss + self_imitation_loss + torch.zeros(1, device=device, requires_grad=True)).backward()
        opt.step();  opt.zero_grad(True)
        with torch.no_grad():
            log(0, False, dist = show_dist)
            log(1, False, ends_here_loss = to_np(ends_here_loss))
            log(2, False, dist_pred_loss = to_np(dist_pred_loss))
            log(3, False, self_imitation_loss = to_np(self_imitation_loss))
            log(4, False, reached = to_np(reached.float().mean()))
            #   (Reaching about .66 means that targets are reached about 100% of the time.)
            log(5, False, action_mean = to_np(action.mean()), action_std = to_np(action.std()))
finish()