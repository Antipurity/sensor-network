"""
TODO: No more RL. We've *seen* where we've been, so it should be very trivial to learn how to get anywhere, right? (Though, might need RL to pick shortest paths anyway, but it shouldn't be the deciding factor.)
"""



from model.log import log, clear, finish
from model.momentum_copy import MomentumCopy



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



N, batch_size = 4, 100
action_sz = 64

unroll_len = N
action_min = False # If `True`, we enum the 4 actions to pick the min-future-distance one at each step.
grad_min = False # With our 4 discrete actions, grad-min just doesn't work no matter what we try.
#   (Grad-min assumes a smooth landscape. Consequences of actions don't sound like a smooth function.)
#   (What else can we do? Learn a generative model of distinct-outcome actions with low distance, and still do action-min on enumerably-few actions?)
self_imitation = False # Makes `next` predict its min-dist version.
#   (Like [SIL](https://arxiv.org/abs/1806.05635) but without probabilities and with learned R.)
#   (Much better than `grad_min`, much worse than `action_min`.)

replay_buffer = [None] * 1024
updates_per_unroll = N

bootstrap_discount = torch.tensor([.99], device=device)
#   Bootstrapping is `f(next) = THIS * f(prev) + local_metric(next)`
#   (Predicting many discounts at once doesn't help.)



next = nn.Sequential( # future → action
    nn.Linear(action_sz, action_sz), nn.ReLU(), nn.LayerNorm(action_sz),
    nn.Linear(action_sz, action_sz),
).to(device)
future_dist = nn.Sequential( # (prev_board, action, target) → future_distance_sum
    # (For picking an action that leads to getting to the target board the fastest.)
    nn.Linear(N*N + action_sz + N*N, action_sz), nn.ReLU(), nn.LayerNorm(action_sz),
    SkipConnection(nn.Linear(action_sz, action_sz), nn.ReLU(), nn.LayerNorm(action_sz)),
    nn.Linear(action_sz, bootstrap_discount.shape[0]),
).to(device)
ev = nn.Sequential( # (board, target) → future
    # TODO: Also accept the target (which is not necessary in this full-info env).
    # (The `ev`entual future of a cell. …Though this name doesn't seem to match with what it's used for…)
    nn.Linear(N*N + N*N, action_sz), nn.ReLU(), nn.LayerNorm(action_sz),
    SkipConnection(nn.Linear(action_sz, action_sz), nn.ReLU(), nn.LayerNorm(action_sz)),
    SkipConnection(nn.Linear(action_sz, action_sz), nn.ReLU(), nn.LayerNorm(action_sz)),
    nn.Linear(action_sz, action_sz), nn.LayerNorm(action_sz),
).to(device)
ev_delayed = MomentumCopy(ev, .99)
ev_next = nn.Sequential( # prev_future → future
    # (For BYOL of `ev`.)
    nn.Linear(action_sz, action_sz), nn.ReLU(), nn.LayerNorm(action_sz),
    SkipConnection(nn.Linear(action_sz, action_sz), nn.ReLU(), nn.LayerNorm(action_sz)),
    SkipConnection(nn.Linear(action_sz, action_sz), nn.ReLU(), nn.LayerNorm(action_sz)),
    nn.Linear(action_sz, action_sz), nn.LayerNorm(action_sz),
).to(device)
opt = torch.optim.Adam([*next.parameters(), *future_dist.parameters(), *ev.parameters(), *ev_next.parameters()], lr=1e-3)

for iters in range(50000):
    # Sample a batch of trajectories.
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
            future = ev(cat(prev_board, target))
            #   Using `zeros` in place of `rand` here is 4× slower to converge. TODO: …Might want to make `ev` take `rand` just like `next` used to, huh…
            action = next(future)
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

        # target = torch.where( # This actually seems to improve convergence speed 2×.
        #     #   (`prev_board` does not. More/less than 50% of the time, does not.)
        #     #   (Possibly, by doing this we're teaching the net how to stay at the goal once reached, which can be done many times in expectation and thus have a disproportionate effect compared to random goals.)
        #     torch.rand(board.shape[0], 1, device=device) < .5,
        #     board,
        #     board[torch.randperm(board.shape[0], device=device)],
        # )
        target = board[torch.randperm(board.shape[0], device=device)]

        zeros = torch.zeros(board.shape[0], action_sz, device=device)
        rand = torch.randn(board.shape[0], action_sz, device=device)

        # TODO: Run, and see what happens.
        #   (Ideally, would see distance going down quickly because good actions get learned instantly, but…)
        #   TODO: Why isn't it working? Why does `trajectory_end_loss` go down to 3, but distance doesn't decrease even a little?…
        #     …Is it because `ev` has no `randn` arg…
        #     …Is it because our loss is wrong………

        # Remember ends of trajectories: `next(ev(goal=board)) = action`.
        trajectory_end_loss = (next(ev(cat(prev_board, board))) - action).square().sum()
        # Remember non-terminal actions of trajectories: `next(ev(goal)) = action`.
        prev_future = ev(cat(prev_board, target))
        trajectory_continuation_loss = (next(prev_future) - action).square().sum()
        # Crystallize trajectories, to not switch between them at runtime: `ev_next(ev(prev)) = ev(next)`.
        with torch.no_grad():
            future = ev_delayed(cat(action, board, target))
        trajectory_ev_loss = (ev_next(ev(cat(prev_board, target))) - future).square().sum()

        # Bootstrapping: `future_dist(prev) = future_dist(next)*p + micro_dist(next)`
        # micro_dist = (board - target).abs().sum(-1, keepdim=True)
        # prev_dist = future_dist(cat(prev_board, action, target.detach()))
        # next_action = next(future)
        # for p in future_dist.parameters(): p.requires_grad_(False)
        # dist = future_dist(cat(board, next_action, target.detach()))
        # for p in future_dist.parameters(): p.requires_grad_(True)
        # prev_dist_targ = dist * bootstrap_discount + micro_dist
        # dist_pred_loss = (prev_dist - prev_dist_targ.detach()).square().mean(0).sum()

        # Min-dist self-imitation, by `next`.
        # action2 = next(prev_future)
        # prev_dist2 = future_dist(cat(prev_board, action2, target.detach()))
        # prev_dist, prev_dist2 = prev_dist.sum(-1, keepdim=True), prev_dist2.sum(-1, keepdim=True)
        # self_imitation_loss = ((action2 - action.detach()).square() * (prev_dist2 - prev_dist).detach().max(zeros)).sum()
        # if not self_imitation: self_imitation_loss = self_imitation_loss.detach()

        # Grad-min of actions.
        # dist_min_loss = dist.mean(0).sum()
        # if not grad_min: dist_min_loss = dist_min_loss.detach()

        if iters == 500: clear()

        # (dist_pred_loss + dist_min_loss + self_imitation_loss + trajectory_end_loss + trajectory_continuation_loss + trajectory_ev_loss + torch.zeros(1, device=device, requires_grad=True)).backward()
        (trajectory_end_loss + trajectory_continuation_loss + trajectory_ev_loss + torch.zeros(1, device=device, requires_grad=True)).backward()
        opt.step();  opt.zero_grad(True);  ev_delayed.update()
        with torch.no_grad():
            # log(0, False, dist_pred_loss = to_np(dist_pred_loss))
            # log(1, False, dist_min_loss = to_np(dist_min_loss))
            # log(2, False, self_imitation_loss = to_np(self_imitation_loss))
            log(0, False, trajectory_end_loss = to_np(trajectory_end_loss))
            log(1, False, trajectory_continuation_loss = to_np(trajectory_continuation_loss))
            log(2, False, trajectory_ev_loss = to_np(trajectory_ev_loss))
            log(3, False, mean_distance = dist_mean / (2*N))
            #   (Reaching about .66 means that targets are reached about 100% of the time.)
            log(4, False, action_mean = to_np(action.mean()), action_std = to_np(action.std()))
finish()