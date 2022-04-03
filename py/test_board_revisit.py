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



N, batch_size = 4, 100 # TODO: N=16
action_sz = 64

unroll_len = N

replay_buffer = [None] * 1024
updates_per_unroll = 1 # Each replay-buffer's entry is `unroll_len` steps long and of width `batch_size`.

bootstrap_discount = torch.tensor([.99], device=device)
#   Bootstrapping is `f(next) = THIS * f(prev) + local_metric(next)`
#   (Predicting many discounts at once doesn't help.)



act = nn.Sequential( # (prev_board, target) → action
    nn.Linear(N*N + N*N, action_sz), nn.ReLU(), nn.LayerNorm(action_sz),
    SkipConnection(nn.Linear(action_sz, action_sz), nn.ReLU(), nn.LayerNorm(action_sz)),
    SkipConnection(nn.Linear(action_sz, action_sz), nn.ReLU(), nn.LayerNorm(action_sz)),
    nn.Linear(action_sz, action_sz), nn.LayerNorm(action_sz, elementwise_affine=False),
).to(device)
future_dist = nn.Sequential( # (prev_board, action, target) → future_distance_sum
    # (For picking an action that leads to getting to the target board the fastest.)
    nn.Linear(N*N + action_sz + N*N, action_sz), nn.ReLU(), nn.LayerNorm(action_sz),
    SkipConnection(nn.Linear(action_sz, action_sz), nn.ReLU(), nn.LayerNorm(action_sz)),
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
    boards, actions = [board], [action.detach()]
    with torch.no_grad():
        # First pick the target to go to.
        #   (From tests, in this trivial env, performance is the same even if we re-pick the target at each step.)
        target = env_init(N, batch_size=batch_size)
        for u in range(unroll_len):
            # Do the RNN transition (and an environment step), `unroll_len` times.
            randn = torch.randn(batch_size, action_sz, device=device)
            # Minimize the future-distance-sum by considering all 4 possible actions right here.
            #   (Minimizing by gradient descent in this environment is no bueno.)
            action = act(cat(board, target))
            if iters % 100 < 50 and random.randint(1, 10) <= 3: action = torch.randn(batch_size, action_sz, device=device)
            #   (Having this is very much not ideal, but it does actually ensure that the RNN explores a lot.)

            board = env_step(N, board, action[..., 0:1])

            reached |= (board == target).all(-1, keepdim=True)

            boards.append(board);  actions.append(action.detach())

        replay_buffer[iters % len(replay_buffer)] = (torch.stack(boards, 0), torch.stack(actions, 0))

    # Replay from the buffer. (Needs Python 3.6+ for convenience.)
    choices = [c for c in random.choices(replay_buffer, k=updates_per_unroll) if c is not None]
    if len(choices):
        boards = torch.cat([c[0] for c in choices], -2) # unroll_len+1 × B × N*N
        actions = torch.cat([c[1] for c in choices], -2) # unroll_len+1 × B × action_sz
        B = boards.shape[1]
        boards, actions = boards[-2:], actions[-2:] # TODO: …Why does adding this make all distances huge… It should behave EXACTLY the same as our old code, no excuses! What's happening?
        # targets = torch.where( # This improves convergence speed 5×. It's the key. # TODO: This version is for multi-step-returns ONLY.
        #     # (Possibly because it makes distances ≈5 instead of ≈65. And removes a factor of variation.)
        #     torch.rand(B, 1, device=device) < .5,
        #     boards[-1],
        #     boards[-1][torch.randperm(B, device=device)],
        # ).unsqueeze(0).expand(boards.shape[0], B, N*N)
        targets = torch.where( # This improves convergence speed 5×. It's the key.
            # (Possibly because it makes distances ≈5 instead of ≈65. And removes a factor of variation.)
            torch.rand(boards.shape[0], B, 1, device=device) < .5,
            torch.cat((boards[1:], boards[-1:]), 0),
            torch.cat((boards[1:], boards[-1:]), 0)[:, torch.randperm(B, device=device)],
        )

        next_actions2 = act(cat(boards, targets))
        next_actions = torch.cat((actions[1:], next_actions2[-1:]), 0).detach()
        dists2 = future_dist(cat(boards, next_actions2, targets))
        dists = future_dist(cat(boards, next_actions, targets))

        # Bootstrapping: `future_dist(prev) = |next - target| + future_dist(next)*p`
        #   Though if next==target, make the distance 0. (Which happens often due to our `targets`-sampling here.)
        with torch.no_grad():
            micro_dists = (boards[1:] - targets[1:]).abs().sum(-1, keepdim=True) / 2
            dists_are = micro_dists + dists2[1:] * bootstrap_discount # 1-step-in-the-future returns.
            dists_are = torch.where(micro_dists < 1e-5, torch.tensor(0., device=device), dists_are)
        # TODO: …How to compute multi-step returns?…
        dist_pred_loss = (dists[1:] - dists_are.detach()).square().sum(-1).mean()
        # TODO: …What did we do wrong?…
        #   Should we literally compare values? …Why are they exactly the same, yet we behave very differently anyway…
        #   …Maybe, the selection of `targets` is only good for the last thing. Elsewhere, it's in high-bias regime… …Did we screw up fixing it, or is the problem not in there?…
        # Self-imitation gated by min-dist, by `act`.
        #   (A lot like [SIL.](https://arxiv.org/abs/1806.05635))
        #   (Actions can interfere if there are many equally-good paths, but it's not too much of a problem.)
        with torch.no_grad():
            next_actions_are2 = torch.where(dists < dists2, next_actions, next_actions2)
        self_imitation_loss = (next_actions2 - next_actions_are2).square().sum(-1).mean()

        # TODO: Our scalability here is atrocious. Get N=16 not up to 15% but up to ≈100%, THEN we can do cont-control.
        #   (It's probably that quadratic-difficulty of propagating distances back, isn't it… And the need to remember distances for *all* actions, not just min-distance actions…)

        # TODO: …Would adding `S`-step returns (making the predicted-distance the min of single-step-dist and distant-dist, for each step in a trajectory, starting with the second-last one) help with convergence speed? It should make it `S` times more efficient, especially initially, right?
        #   (A synergy is with `future_dist`: its target-estimation would be more accurate initially with a non-predicted action.)
        #   (…Wouldn't it be biased due to relying on the predicted-distance of predicted-actions? Which is initially near-0, by the way, meaning that most multi-step returns would get discarded. Overoptimism is slow to correct, only new paths are quicker to propagate.)
        #   (Is pretty much tree-backup with a greedy policy, pretty sure. Which is [non-convergent](https://arxiv.org/abs/1705.09322) but [good enough](https://arxiv.org/pdf/2007.06700v1.pdf).)
        #   …All we can do is try it. TODO: Try it.
        #   TODO: …How do we implement this, exactly?
        #     TODO: Maybe, first write down at least the Python-side algorithm?

        # TODO: …If we gate the distance, is it possible to make the too-high-dist branch's prediction not exact but like "a bit more than the taken-branch distance"?…
        #   (No need to over-learn the learned loss in places where we don't use it.)
        #   (In the one-step limit, this means `prev_dist_targ = torch.where(prev_dist < prev_dist2, prev_dist_targ, prev_dist)`. Which takes forever to converge.)

        # TODO: …If we have gating-info, then we can do BPTT wherever the old trajectory is preferred: unroll `act` and give `action`'s gradient to `action2` via `action + action2 - action2.detach()`… Try that… (Won't help with directly minimizing the quadratically-full `micro_dist` sum, though.)
        #   (After all, without gradient descent, we'll suffer from the curse of dimensionality.)
        #   (…If we did input-embedding and BYOL-on-RNN, the non-learned distance would have been optimized too…)



        prev_board = torch.cat([c[0][-2] for c in choices], 0) # TODO: Remove.
        prev_action = torch.cat([c[1][-2] for c in choices], 0) # TODO: Remove.
        board = torch.cat([c[0][-1] for c in choices], 0) # TODO: Remove.
        action = torch.cat([c[1][-1] for c in choices], 0) # TODO: Remove.

        target = targets[0]

        # Bootstrapping: `future_dist(prev) = |next - target| + future_dist(next)*p`
        #   Though if next==target, make the distance 0.
        #     (Which happens often due to our `target`-sampling here.)
        prev_dist = future_dist(cat(prev_board, action, target))
        action2 = act(cat(prev_board, target))
        prev_dist2 = future_dist(cat(prev_board, action2, target))
        print(0, (prev_dist2 - dists2[-2]).abs().sum()) # TODO: …Same…
        with torch.no_grad():
            next_action2 = act(cat(board, target))
            dist2 = future_dist(cat(board, next_action2, target))
            micro_dist = (board - target).abs().sum(-1, keepdim=True) / 2
            prev_dist_targ = micro_dist + dist2 * bootstrap_discount
            prev_dist_targ = torch.where(micro_dist < 1e-5, torch.tensor(0., device=device), prev_dist_targ)
        print(1, (prev_dist_targ - dists_are[-1]).abs().sum()) # TODO: Compare prev_dist_targ. …They're the same… …Wait, why do they differ *now*? What changed? Anyway, FOUND THE BUG
        dist_pred_loss = (prev_dist - prev_dist_targ.detach()).square().mean(0).sum()

        # Self-imitation gated by min-dist, by `act`.
        #   (A lot like [SIL.](https://arxiv.org/abs/1806.05635))
        #   (Actions can interfere if there are many equally-good paths, but it's not too much of a problem.)
        with torch.no_grad():
            target_action = torch.where(prev_dist < prev_dist2, action, action2)
        self_imitation_loss = (action2 - target_action).square().mean(0).sum()

        if iters == 1000: clear()

        (dist_pred_loss + self_imitation_loss + torch.zeros(1, device=device, requires_grad=True)).backward()
        opt.step();  opt.zero_grad(True)
        with torch.no_grad():
            log(0, False, dist = show_dist)
            log(1, False, dist_pred_loss = to_np(dist_pred_loss))
            log(2, False, self_imitation_loss = to_np(self_imitation_loss))
            log(3, False, reached = to_np(reached.float().mean()))
            #   (Reaching about .66 means that targets are reached about 100% of the time.)
            log(4, False, action_mean = to_np(action.mean()), action_std = to_np(action.std()))
finish()