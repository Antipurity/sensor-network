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



N, batch_size = 8, 100 # TODO: Even like this, 70% is quite reachable. But can we reach 100%, by extending whole trajectories back?
#   TODO: N=4
action_sz = 64

unroll_len = N
action_min = False # If `True`, we enum the 4 actions to pick the min-future-distance one at each step.
grad_min = False # With our 4 discrete actions, grad-min just doesn't work no matter what we try.
#   (Grad-min assumes a smooth landscape. Consequences of actions don't sound like a smooth function.)
#   (What else can we do? Learn a generative model of distinct-outcome actions with low distance, and still do action-min on enumerably-few actions?)
self_imitation = False # Makes `act` predict its min-dist version.
#   (Like [SIL](https://arxiv.org/abs/1806.05635) but without probabilities and with learned R.)
#   (Much better than `grad_min`, much worse than `action_min`.)

replay_buffer = [None] * 1024
updates_per_unroll = N

bootstrap_discount = torch.tensor([.99], device=device)
#   Bootstrapping is `f(next) = THIS * f(prev) + local_metric(next)`
#   (Predicting many discounts at once doesn't help.)



act = nn.Sequential( # (prev_board, future) → action
    # (TODO: …Also, wouldn't `act` want to know not which ring it's in, but which ring it's going to? And, always go to the closer-to-the-goal ring, which a loss should ensure…)
    nn.Linear(N*N + action_sz, action_sz), nn.ReLU(), nn.LayerNorm(action_sz),
    SkipConnection(nn.Linear(action_sz, action_sz), nn.ReLU(), nn.LayerNorm(action_sz)),
    nn.Linear(action_sz, action_sz), nn.LayerNorm(action_sz),
).to(device)
future_dist = nn.Sequential( # (prev_board, action, target) → future_distance_sum
    # (For picking an action that leads to getting to the target board the fastest.)
    nn.Linear(N*N + action_sz + N*N, action_sz), nn.ReLU(), nn.LayerNorm(action_sz),
    SkipConnection(nn.Linear(action_sz, action_sz), nn.ReLU(), nn.LayerNorm(action_sz)),
    nn.Linear(action_sz, bootstrap_discount.shape[0]),
).to(device)
ev = nn.Sequential( # (board, target) → future
    # (The `ev`entual future of a cell. …Though this name doesn't seem to match with what it's used for…)
    nn.Linear(N*N + N*N, action_sz), nn.ReLU(), nn.LayerNorm(action_sz),
    SkipConnection(nn.Linear(action_sz, action_sz), nn.ReLU(), nn.LayerNorm(action_sz)),
    nn.Linear(action_sz, action_sz), nn.LayerNorm(action_sz),
).to(device)
ev_delayed = MomentumCopy(ev, .999)
ev_act = nn.Sequential( # (prev_board, prev_future, action) → future
    # (For BYOL of `ev`.)
    nn.Linear(N*N + action_sz + action_sz, action_sz), nn.ReLU(), nn.LayerNorm(action_sz),
    SkipConnection(nn.Linear(action_sz, action_sz), nn.ReLU(), nn.LayerNorm(action_sz)),
    SkipConnection(nn.Linear(action_sz, action_sz), nn.ReLU(), nn.LayerNorm(action_sz)),
    nn.Linear(action_sz, action_sz), nn.LayerNorm(action_sz),
).to(device)
opt = torch.optim.Adam([*act.parameters(), *future_dist.parameters(), *ev.parameters(), *ev_act.parameters()], lr=1e-3)

def show_ev(plt, key):
    with torch.no_grad():
        board = torch.eye(N*N, N*N, device=device)
        target = torch.eye(1, N*N, device=device).expand(N*N, N*N)
        future = ev(cat(board, target))
        v = future[..., :3].reshape(N, N, 3)
        colors = (v - v.min()) / (v.max() - v.min())
        plt.imshow(colors.cpu().numpy(), label=key)

for iters in range(50000):
    # Sample a batch of trajectories.
    action = torch.zeros(batch_size, action_sz, device=device)
    board = env_init(N, batch_size=batch_size)
    dist_mean = 0
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
            action = act(cat(prev_board, ev(cat(prev_board, target))))
            if iters % 100 < 50 and random.randint(1, 10) <= 3: action = torch.randn(batch_size, action_sz, device=device)
            #   (Having this is very much not ideal, but it does actually ensure that the RNN explores a lot.)
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

            board = env_step(N, prev_board, action[..., 0:1])

            reached |= (board == target).all(-1, keepdim=True)
            micro_dist = (board - target).abs().detach()
            dist_mean += micro_dist.mean(0).sum() # Something to log.

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
        # target = board[torch.randperm(board.shape[0], device=device)]

        zeros = torch.zeros(board.shape[0], action_sz, device=device)

        # TODO: Why isn't it working?


        # Remember ends of trajectories: `act(ev(goal=board)) = action`.
        future_here = ev(cat(prev_board, board))
        action2 = act(cat(prev_board, future_here).detach()) # Detaching doesn't hurt performance at all.
        trajectory_end_loss = (action2 - action).square().sum()
        # Debug.
        # def which(a):
        #     return torch.where(
        #         a[..., 0] < 0.,
        #         torch.where(a[..., 0] < -.5, 0, 1),
        #         torch.where(a[..., 0] < .5, 2, 3),
        #     )
        # A = torch.stack((action[...,0]<-.5, (action[...,0]>=-.5) & (action[...,0]<0.), (action[...,0]>=0.) & (action[...,0]<.5), action[...,0]>=.5), 0)
        # B = torch.stack((action2[...,0]<-.5, (action2[...,0]>=-.5) & (action2[...,0]<0.), (action2[...,0]>=0.) & (action2[...,0]<.5), action2[...,0]>=.5), 0)
        # print(A.float().sum(-1).cpu().numpy(), ((A == B) & (A | B)).float().mean(-1).cpu().numpy()) # TODO: …Why is the actual action matched 100% of the time, but actual reachability is like 55%?…
        # Remember non-terminal actions of trajectories: `act(ev(goal)) = action`.
        prev_future = ev(cat(prev_board, target))
        trajectory_continuation_loss = 0 # (act(cat(prev_board, prev_future)) - action).square().sum()
        #   TODO: This loss is exactly the same as `action2` but with a random `target`, which will ABSOLUTELY average over `target` and make it completely meaningless…
        #   …Should we use `next_future` here *maybe*?… But past/future meanings don't match up then…
        # Crystallize trajectories, to not switch between them at runtime: `ev_act(ev(prev)) = ev(next)`.
        # with torch.no_grad():
        #     future = ev(cat(board, target)).detach()
        trajectory_ev_loss = 0 # (ev_act(cat(prev_board, prev_future, action)) - future).square().sum()
        #   (Can't help but notice that this loss-component alone removes 20% of reachability.)
        #   TODO: …Those random actions, though good for exploration, make this part of the loss useless, right?… What do we do now?
        #   TODO: COME UP WITH SOMETHING
        #     …We need a trajectory-description, and to make the prev-trajectory the same as (or transforms-into) the next-trajectory…
        #     …But we also need to make sure that same-trajectory-description actions are still followed…
        #       …Should we preserve the trajectory-destination in the replay-buffer, and train same-trajectory `act` with `action`?…
        #       …Should we try only learning the rest after 3k/5k iterations are reached?…
        #       TODO: …Should `act` accept not only the future but also all its args (*possibly* without the goal), to make futures kind of optional?…
        #         …Should we make `ev`s of destinations the same known quantities, such as all-0s?…
        #           (This one might be a very good idea, because with our current `ev` loss, it just varies like it's nothing. In fact, `ev_act` will probably end up counting transitions AKA distance to the end, and if 2 paths merge, their representations would have pressure to become the same… which might or might not prioritize shorter paths somehow, maybe if they have more neighbors and thus more pressure…)
        #             (This should create equal-embedding rings around targets. Meaning that `act` won't have enough information to decide on the action. …Would it really create rings? The farther thing will become the average of closer things that it can reach with an action, but the closer things would get momentum-updated, without any averaging done… …Starting to think that since `ev_act` depends on the action, there would be no rings, and we'd need to explicitly learn the distance… But if that's the case, then why do we need `ev` at all, we can just pin down zero-distance at ends and increment it by 1 elsewhere…)
        #               …Should we try replacing momentum-updating with separate `ev_prev` and `ev`?…
        #                 But how to ensure that the next step's ev_prev is the prev step's ev without momentum-updating? Just `ev_prev(board, action) = ev(prev_board)`, possibly with *another* neural net to go in reverse to `ev_act`?
        #                   `ev_act(cat(prev_board, ev_prev(prev_board), action)) = ev(board).detach();  ev(prev_board) = ev_prev(board).detach()`…
        #                 (Can we use `ev_act` anywhere?…)
        #     TODO: …Intermediate-action-prediction is actually good to have (can't really act otherwise), but we're stifled by the target being averaged… Is there really no way to remember which action goes exactly to the next `ev`-future?…
        #       …And, how would we know which action is *best* and which are just superfluous… Don't we really need that distance?… Is distance the devil, or just a natural side-effect of finding the shortest path, like in Dijkstra's algo (in which case, well, it can be easily slotted into the bigger framework)?

        # Pin down the result of `ev` at trajectory ends.
        # with torch.no_grad():
        #     last_future = torch.eye(1, action_sz, device=device).expand(board.shape[0], action_sz)
        #     last_future = (last_future - last_future.mean(-1, keepdim=True)) / (last_future.std(-1, keepdim=True) + 1e-5)
        # trajectory_ev_loss = (future_here - last_future).square().sum()
        # TODO: Just having this results in all `ev`s becoming exactly the same… How to extend the trajectory? …Actually, even with extension, results are nearly-indistinguishable…
        # Preserve the `ev` trajectory.
        with torch.no_grad():
            future = ev_delayed(cat(board, target)).detach()
        trajectory_continuation_loss = (ev_act(cat(prev_board, ev(cat(prev_board, target)), action)) - future).square().sum()
        # trajectory_continuation_loss = (ev(cat(prev_board, target)) - future).square().sum()
        # Remember actions that lead to the same result (…probably).
        #   …But we have no way of ensuring that actions actually go to the goal, only that they continue where they were going…
        #   …If only we had a way to ensure that next_action ended up at the same place due to our action or something…
        future = ev(cat(prev_board, target)) # …Oh no, this averages over `target`s, and so it cannot be done.
        action3 = act(cat(prev_board, future).detach())
        trajectory_ev_loss = (action3 - action).square().sum()
        #   2 1 2
        # 2 1 0 1 2
        #   2 1 2
        #     2
        # Ensuring this with loss requires that the past predicts the future (so that the future becomes smudged into one color), if we think of actions as always pointing to 0… However, if we also have actions that point away from 0, then such a loss would jumble everything into a constant.
        #   Trajectory-preservation is just good for nothing.
        #   …If we did preserve the original target, would things be different? (It *is* too bad to be used, but just want to see what happens. …Doesn't even look different, can't even remember its own trajectories.)


        # …We can also kinda turn this loss into Go-Explore by making `ev` output a particular state once the target is actually reached (and ensuring that once we reach such a state, we're in "exploratory mode" where we stay in that mode and do actions randomly)…

        # …A novelty-reward for proposing goals could be the misprediction of next-future by prev-future, very similarly to Random Network Distillation, so that we automatically prioritize areas that we're not proficient in…
        #   (If nothing else, this gives an incentive to construct a world model beyond distance-prediction. …Unless distance-misprediction can also be the novelty-reward, maybe even given as an input.)



        # Bootstrapping: `future_dist(prev) = future_dist(next)*p + micro_dist(next)`
        # micro_dist = (board - target).abs().sum(-1, keepdim=True)
        # prev_dist = future_dist(cat(prev_board, action, target.detach()))
        # next_action = act(cat(board, future))
        # for p in future_dist.parameters(): p.requires_grad_(False)
        # dist = future_dist(cat(board, next_action, target.detach()))
        # for p in future_dist.parameters(): p.requires_grad_(True)
        # prev_dist_targ = dist * bootstrap_discount + micro_dist
        # dist_pred_loss = (prev_dist - prev_dist_targ.detach()).square().mean(0).sum()

        # Min-dist self-imitation, by `act`.
        # action2 = act(cat(prev_board, prev_future))
        # prev_dist2 = future_dist(cat(prev_board, action2, target.detach()))
        # prev_dist, prev_dist2 = prev_dist.sum(-1, keepdim=True), prev_dist2.sum(-1, keepdim=True)
        # self_imitation_loss = ((action2 - action.detach()).square() * (prev_dist2 - prev_dist).detach().max(zeros)).sum()
        # if not self_imitation: self_imitation_loss = self_imitation_loss.detach()

        # Grad-min of actions.
        # dist_min_loss = dist.mean(0).sum()
        # if not grad_min: dist_min_loss = dist_min_loss.detach()

        if iters == 1000: clear()

        # (dist_pred_loss + dist_min_loss + self_imitation_loss + trajectory_end_loss + trajectory_continuation_loss + trajectory_ev_loss + torch.zeros(1, device=device, requires_grad=True)).backward()
        (trajectory_end_loss + trajectory_continuation_loss + trajectory_ev_loss + torch.zeros(1, device=device, requires_grad=True)).backward()
        opt.step();  opt.zero_grad(True);  ev_delayed.update()
        with torch.no_grad():
            # log(0, False, dist_pred_loss = to_np(dist_pred_loss))
            # log(1, False, dist_min_loss = to_np(dist_min_loss))
            # log(2, False, self_imitation_loss = to_np(self_imitation_loss))
            log(0, False, ev = show_ev)
            log(1, False, trajectory_end_loss = to_np(trajectory_end_loss))
            log(2, False, trajectory_continuation_loss = to_np(trajectory_continuation_loss))
            log(3, False, trajectory_ev_loss = to_np(trajectory_ev_loss))
            log(4, False, mean_distance = dist_mean / (2*N))
            log(5, False, reached = to_np(reached.float().mean()))
            #   (Reaching about .66 means that targets are reached about 100% of the time.)
            log(6, False, action_mean = to_np(action.mean()), action_std = to_np(action.std()))
finish()