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



N, batch_size = 8, 100 # TODO: N=16
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
        K, B = boards.shape[0], boards.shape[1]
        targets = torch.where( # This improves convergence speed 5×. It's the key.
            # (Possibly because it makes distances ≈5 instead of ≈65. And removes a factor of variation.)
            torch.rand(1, B, 1, device=device) < .5,
            torch.cat((boards[1:], boards[-1:]), 0), # This is somehow better than `boards[-1:]`.
            torch.cat((boards[1:], boards[-1:]), 0)[:, torch.randperm(B, device=device)],
        )

        next_actions2 = act(cat(boards, targets))
        next_actions = torch.cat((actions[1:], next_actions2[-1:]), 0).detach()
        dists2 = future_dist(cat(boards, next_actions2, targets))
        dists = future_dist(cat(boards, next_actions, targets))

        # Bootstrapping: `future_dist(prev) = |next - target| + future_dist(next)*p`
        #   Though if next==target, make the distance 0. (Which happens often due to our `targets`-sampling here.)
        with torch.no_grad():
            # Compute (K-1)-step returns in Python, no care about efficiency.
            micro_dists = (boards[1:] - targets[1:]).abs().sum(-1, keepdim=True) / 2
            # dists_are = micro_dists + dists2[1:] * bootstrap_discount # 1-step-in-the-future returns.
            # dists_are = torch.where(micro_dists < 1e-5, torch.tensor(0., device=device), dists_are)
            dists_are, dist_so_far = [], dists2[-1]
            for k in reversed(range(K-1)):
                dist_so_far = dist_so_far.min(dists2[k+1]) # Don't follow the old trajectory if we have a better option.
                dist_so_far = micro_dists[k] + dist_so_far * bootstrap_discount
                dist_so_far = torch.where(micro_dists[k] < 1e-5, torch.tensor(0., device=device), dist_so_far) # Ground `target`-reaching steps to have 0 distance.
                dists_are.append(dist_so_far)
            dists_are = torch.stack(list(reversed(dists_are)), 0)
        dist_pred_loss = (dists[:-1] - dists_are.detach()).square().sum(-1).mean()
        # Self-imitation gated by min-dist, by `act`.
        #   (A lot like [SIL.](https://arxiv.org/abs/1806.05635))
        #   (Actions can interfere if there are many equally-good paths, but it's not too much of a problem.)
        with torch.no_grad():
            next_actions_are2 = torch.where(dists < dists2, next_actions, next_actions2)
        self_imitation_loss = (next_actions2 - next_actions_are2).square().sum(-1).mean()

        # TODO: Our scalability here is atrocious. Get N=16 not up to 15% but up to ≈100%, THEN we can do cont-control.
        #   …Mission failed…
        #   …Is distance too hard to learn after all?…
        #   …Is our only remaining option REALLY to re-examine BYOL for clustering trajectories, where `act` is the input embedding…








        # TODO: Already assume that this will fail, and gain an understanding of why:
        #   1. TODO: Ensure that trajectories with the same future do end up there: `ev(act(prev, goal)) = act(next, goal)`
        #   2. TODO: Ensure that the final-future is as the replay buffer says: `act(prev, next) = action`


        # TODO: On a graph A|B|C|D  |  A→D, B→A, B→C, C→B, C→D.
        #   TODO: What are the optimal conditions for `act(prev, goal)` to end up pointing to goals? (No distances.)
        #     First, the near-goal conditions:
        #       INIT
        #       act(A,A) = ???, act(A,B) = ???, act(A,C) = ???, act(A,D) = A→D
        #       act(B,A) = B→A, act(B,B) = ???, act(B,C) = B→C, act(B,D) = ???
        #       act(C,A) = ???, act(C,B) = C→B, act(C,C) = ???, act(C,D) = C→D
        #     Second, spread them back, depth-first: given `(prev, action, next, goal)`, if `act(prev, goal) == ???` (don't overwrite) but `act(next, goal) != ???` (spread backwards), do `act(prev, goal) = action` (for each next & goal, find an action that ends in next with a non-taken prev):
        #       TRUE
        #       act(A,A) = ···, act(A,B) = ···, act(A,C) = ···, act(A,D) = A→D
        #       act(B,A) = B→A, act(B,B) = B→C, act(B,C) = B→C, act(B,D) = B→A
        #       act(C,A) = C→B, act(C,B) = C→B, act(C,C) = C→B, act(C,D) = C→D
        #       (…"Find a `prev` with the same action as `next` & `goal` (the actual next action is irrelevant)" or "given `prev` and `goal`, find an `action: prev→next` with a filled `next`" *could* mean "make all such actions' embeddings the same"…)
        #       (…"Don't overwrite" *could* mean "use `ev` somewhere here"…)
        #       TODO: …Try writing down the exact equation-producing equations that we could use to solve this, using `ev action` and `act(prev,goal)`…
        #         …Didn't we try essentially both configurations like this, and found that neither works?…
        #         …If we didn't have to worry about overwriting, we could have just done `prev→next:  act(prev, goal) = prev→next = act(prev, next)`… What equation would possibly imply this one?…
        #           …To worry about overwriting, we do need to add a construction on either side of the equation. But which side, and which construction?…
        #           …`ev` with only the action has failed either way… Can we fix it by also conditioning it on state and/or goal (thereby reducing how many equations are produced)?…
        #           …Is this equation fundamentally about combining goals, like `act(prev, goal) = act(prev, next) + act(next, goal)`?… But how to write this down as a real equation?… Do we maybe want `ev` to act as the `+` here — but then, how to make prev-action be the same as current-action iff next goes to goal?…
        #             TODO: …Try writing down the actual, non-simplified equation, first in terms of distances (…how to do even that if we don't have state?), then hierarchy levels?…
        #               dist(prev, next) = 1
        #               × dist(a,c) = min(b, dist(a,b) + dist(b,c))   (NOT CONSTRUCTIVE)
        #               × dist(a,c) = min(dist(a,c), dist(a,b) + dist(b,c))   (Bootstrap; numerically problematic.)   (NOT CONSTRUCTIVE)
        #               dist(a, c, a→b) = 1 + dist(b, c, act(b,c))   (The only constructive solution)
        #               act(a,c) if a→b and dist(a,c,a→b) < dist(a,c,act(a,c)) = a→b (Construct the min spanning tree) (NON CONSTRUCTIVE)
        #               …I really don't see how we'd come up with a scheme that can actually do that search without writing "search pls"…


        #   TODO: Try that 'algorithm' above (where "apply a loss" means "write down an equation"); see whether it can possibly converge (meaning that solving the equations gives us a superset of the real solution, but non-equal things are still non-equal (didn't collapse)). If not, think of the difference in how we've assigned optimal actions manually and automatically, and make a new algorithm more like our manual thinking.
        #       (…Oh, neither `ev act(prev,·)=act(next,·)` nor `act(prev,·)=ev act(next,·)` allow changing goals, so they fundamentally can't construct the map that we want…)
        #         …Can we allow that switching, with nothing but formal constructs, *not* distances?… GAH, HOW

        #       ==============================================================
        #       act(A,A) = ???, act(A,B) = ???, act(A,C) = ???, act(A,D) = A→D
        #       act(B,A) = B→A, act(B,B) = ???, act(B,C) = B→C, act(B,D) = ???
        #       act(C,A) = ???, act(C,B) = C→B, act(C,C) = ???, act(C,D) = C→D
        #       ==============================================================
        #       A→D, B→A, B→C, C→B, C→D
        #         (Applying `ev(act(prev, ·)) = act(next, ·)` for all actions `prev→next`:)
        #       act(A,·) = ev(act(B,·))
        #       act(B,·) = ev(act(C,·))
        #       act(C,·) = ev(act(B,·))
        #       act(D,·) = ev(act(A,·)) = ev(act(C,·))
        #         (Definitely not enough to infer any paths by itself. Really need some separate way to ensure that actions transform in the right way…)
        #       …Inconclusive, though definitely not enough by itself…
        #       …The fact that we're kinda adjusting the next action in a trajectory rubs me the wrong way now, given how we did the algorithm… Maybe, try `act(prev, ·) = ev act(next, ·)` now?
        #       A→D, B→A, B→C, C→B, C→D
        #       act(A,·) = ev act(D,·)
        #       act(B,·) = ev act(C,·) = ev act(B,·)
        #       act(C,·) = ev act(B,·) = ev act(D,·)
        #       …Inconclusive…



        # (…An intuitively-appealing idea is: given an RNN that reaches fixed-points at (prev, goal=next), and for which each prev→next transition does one more step toward the fixed-point for any goal (RNN(ev(prev,·))=ev(next,·), probably), find the actions (`ev` here? Does it need to know the action, or is its result the action? Or, is the action inferred from the result such that the goal here is minimized…) that get the RNN closer to the fixed-point, somehow…)
        # (…Also, if we do end up learning distance YET AGAIN, maybe we could try ensuring linear combinations instead of one-step summation: `dist(prev, next) = |prev-next|` and `dist(a,c) = min(dist(a,c), dist(a,b) + dist(b,c))`? With this, we'll be able to compose sequences much more quickly, though we do need to pick `b` intelligently. …And, to reduce how much we need to learn, don't condition the distance on the min-dist action, instead find that min-dist already — though, if the policy is optimal anyway, it shouldn't matter…)
        #   (Dijkstra's has a set of soon-to-be-visited nodes, sampled in a min-dist order. In ML, this means a generative model. Very much like open-ended algorithms such as AMIGo and POET: always suggest tasks that are neither too hard nor too easy.)
        #   (What we kinda want is Prim's algorithm but for ML… Though Prim's algo uses distances, connecting the action with the min distance…)
        # (…If we switch from sum-of-distances to min-distance (and forego min-path-finding), then we could train a discriminator of whether an action would reach the goal (by copying the next action's future-probability, or making it 100% if we're at goal), and train actions to maximize reachability… The discrimination would collapse to 100% eventually, though…)
        # (…I think classical pathfinding algorithms are already well-used in ML… Have to understand hierarchy contractions to have even a chance…)












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