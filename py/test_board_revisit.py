"""
Here, we revisit the 2D board env, since continuous-control with our prior RL-understanding refused to be solved, a few times already.

This time, to remove prediction-interference: the env's action needs 4 numbers, and is decided by which is higher.

Open-ended algorithms, to prepare for any goal, learn min-distance paths between *all* source and goal states. (Reward-maximizing agents can be rewritten as agents that go to high-reward states.)

In theory, we've *seen* where we've been, so it should be trivial to revisit them, right? Learn min-dist trees from everywhere to everywhere.

What we want is:

- Without the quadratic bottleneck of learning precise all-to-all distances, like reinforcement-learning algorithms always do.
  - (Besides, the ability to distinguish between 5312 and 5314 steps to judge plans isn't very biologically plausible.)

- Without the linear-time to backprop the info, since that's not transferrable to very long horizons (like millions of steps), especially with function approximation that can compound errors.

---

# Past attempts to overcome the quadratic bottleneck

---

(As tedious as this may be to read through, this research trajectory was far more tedious to unroll.)

For explicitly discriminating which action comes earlier (for self-imitation), could learn distances or distance-surrogates (given a 'future', possibly BYOL-learned, by notation here):

- Learn all the actual pairwise distances. (Quadratically-many numbers to learn: too much.)
  - (But compared to any of the options below, it's amazing.)
- Learn a goal-dependent embedding `ev` for which `ev(prev,g) = prev_ev(ev(next,g))` and `ev(g,g) = OK` (make it smoothly transform into goals), and compare func-call-depths when needed:
  - Learn the actual distance given `ev`.
  - Learn a comparator of 2 `ev`s, `cmp(fut1, fut2)→-1…0…1`: `cmp(RNN(f),f)=-1`, `cmp(f,f)=0`, `cmp(f,RNN(f))=1`, `cmp(RNN(f), RNN(g)) = cmp(f,g)`, `cmp(RNN(f),g) where cmp(f,g)<0 = -1`, `cmp(f,RNN(g)) where cmp(f,g)>0 = 1`.
  - (…Though, learning a goal-dependent vortex is as info-heavy as learning all pairwise distances if not more, so it's just as bad as that.)
- Continuous contraction: given src & dst & max-dist, alter src within those confines to get closer to dst. Then, we can decompose all problems at unroll-time: go halfway, then the rest of the way.
  - Have a GAN `alter(src, dst, max_dist)→src` & `alter_dist`, where the 'discriminator' is the distance from output to dst. Losses:
    - `alter_dist(prev, next, |prev-next| or more) = |prev-next|`
    - `alter_dist(prev, goal, max_dist) = min(max_dist, |prev-next| + .99 * alter_dist(next, goal, dist - |prev-next|))`
    - (…Wow, this is even worse than all-to-all distance learning.)
- Discrete contraction hierarchies: ground *tasks* in one-step transitions, and combine 2 consecutive *tasks* on each new hierarchy level (contracting a node by combining 2 actions that go through it in 1 meta-action/shortcut: similar to classical contraction hierarchies). Effectively, the level (an extra input to many neural nets) is floor of log-2 of dist.
  - Good: the 'distance' has much less precision and thus takes less space in neural nets. Bad: with N levels, we need N× more compute.
  - To always pick the shortest action on each level (especially when we have `A→B  B→C  A→C`) instead of smudging (and trying to connect all to all), we need to learn a "which level does this action come from" net, and use that for gated action-prediction. Which is pretty much the old dist.
  - Nodes are learned with BYOL (`next(ev(A)) = sg ev(B)` if A→B is a real meta/transition) to be differentiable, and to make same-endpoints paths have the same embeddings, but if we want this to not collapse or smudge, then we need every single meta-action with distinct endpoints to be both distinct and perfectly-learned. Since there are as many total meta-actions as there are node pairs, we're back to the non-scalable "need quadratic neural-net capacity" but in a worse way.
    - (…Wow, this is way worse than all-to-all distance learning. But aren't at least some of its ideas salvageable?)
- Dijkstra-algo-like filling from src and/or dst: given src, some dsts are "settled" (we know the action), some are "unknown" (we haven't reached it yet), others are "considered" (reached but learning how to re-reach). Model the "considered" set via generative models, one way or another.
  - (Need horizons of tasks that are neither too easy (visited & settled) nor too hard (never neighboring any visited state), like in AMIGo or POET or [iterative deepening](https://en.wikipedia.org/wiki/Iterative_deepening_depth-first_search).)
  - (We don't need all-to-all distances, we only need to know whether some distances are definitely worse than others.)
  - A particularly easy impl is: with a `(src, dist) → dst` generative model, sample from it twice (with a random `dist`ance) and update the double-stepped distance if it's shorter than what we had in mind.
    - The `dist`-to-learn could be `floor(log2(actual_dist))`, which reduces the required NN-output precision dramatically.
  - (The problem is that generative models, of non-stationary sparse distributions, are very hard to actually learn.)

# Final/current solution

- Combining whole plans, shortcutting through midpoints. Replace the `dist(src, dst, [action?])` net with `embed(src)` and measure distances (`1+log(steps)`) in embedding space: so, learn a locally-isometric map of the env's topology (a torus, in this 2D env). Preserve faraway states, along with inputs and first-action and distance. Then, we can just learn the min-dist `act(src, dst)`.
    - (The dead-simple "learn the `steps=1` actions" without dist-map-learning performs worse than with that, so representation-learning is important, and ensures (local) directional alignment.)
    - (While performing simple self-imitation weighed by the difference of in-replay and in-embedding dists works, for which storing even simple faraway pairs suffice, it doesn't scale.)
    - With `A→B→C` faraway double-stepping, we can do exponential improvement by reusing a task's result: `act(A,C) = act(A,B).detach()` — in addition to single-step `act(prev,next) = prev→next` grounding, and to dist-learning. (GANs are replaced by faraway sampling.)

# Potential good things

(No good-enough reason to try, here.)

- Pick unroll-goals to maximize distance-misprediction, similarly to [AdaGoal](https://arxiv.org/abs/2111.12045).

- Problem: `embed`-space distances are symmetric. Solution: different nets for source-embedding and destination-embedding.

- Ablate whether in-`embed`-space distances are really better than `dist`s. (Though, real closeness-learning in `embed`-space allows `act` to easily generalize, so the test should be in a complex env.)

- Stabilize dist-learning by conditioning it on the level, where level 0 always predicts the sampled distance, and each next level predicts prev-level dists but lower-dists get a higher loss-multiplier.

- Train an ensemble of `act`s, and gate meta-action-prediction by how certain the closer-destination action is (such as `act_loss / (-(act1-act2).abs().sum(-1, keepdim=True)).exp().detach()`). (Gives no benefit in this trivial env.)

# This is pretraining for RL

Goal-state-reaching can actually learn to reach not only goal-states, but arbitrary funcs of them. All we need is to add fake `state→that` transitions to the replay buffer. This way, we can erase as much info as we want.

Examples of goal-funcs:

- RL (reward-maximization) can be written as reaching any state that gets a high sum-of-past-rewards. If we have goal-state-reaching trained, then a user can come along and mark states with reward; and the moment the reward-goals get nearly the same `embed`ding due to dist-learning, the agent would instantly know how to reach high reward from anywhere (though, situating the new closeness might take a while).

- Another model's view, for example, [CLIP](https://github.com/openai/CLIP) embeddings of images and/or text. Explore semantically-meaningful states, not just distinct states — without huge computational resources.
    - (If evaluating at every unroll-step is too expensive, then just don't.)

- Arbitrary human input. Prompt distinct reactions from a human. (This might play with AI safety a bit too closely though.)

- All of the above, with a header to distinguish which goal we're at.
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
    """Given a one-hot board encoding and a 4-number action (the max value wins), returns a new board state, cyclically-shifted in the correct direction. Batchable.

    (The 2-number scheme that relied on the max-magnitude direction was abandoned, because it played poorly with action averaging.)"""
    board = board.reshape(*board.shape[:-1], N, N)
    sh = [*board.shape[:-2], N*N]
    board_px = torch.cat((board[..., :, -1:], board[..., :, :-1]), -1).reshape(sh)
    board_py = torch.cat((board[..., -1:, :], board[..., :-1, :]), -2).reshape(sh)
    board_mx = torch.cat((board[..., :, 1:], board[..., :, :1], ), -1).reshape(sh)
    board_my = torch.cat((board[..., 1:, :], board[..., :1, :], ), -2).reshape(sh)
    action = action.argmax(-1, keepdim=True)
    return torch.where(
        action < 2,
        torch.where(action == 0, board_px, board_mx),
        torch.where(action == 2, board_py, board_my),
    )



class SkipConnection(nn.Module):
    def __init__(self, *fn): super().__init__();  self.fn = nn.Sequential(*fn)
    def forward(self, x): return self.fn(x) + x
def to_np(x): return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
def cat(*a, dim=-1): return torch.cat(a, dim)



N, batch_size = 16, 100
action_sz = 64
emb_sz = 64

unroll_len = N

rb, replay_buffer = 0, [None] * 1024
updates_per_unroll = 1 # Each replay-buffer's entry is `unroll_len` steps long and of width `batch_size`.


def dist(x,y):
    """Calculates the distance in embedding-space, which by training, should be made into `1+log(path_length)` between board-inputs; for prev→next transitions, the path length is 2 (for simplicity of math).

    By not learning a neural net to output the distance, we learn an actual topological model of the environment, in which shortest-path actions are the ones closest to straight lines to where we want to go (so, little info for `act` to remember)."""
    return (x - y).square().mean(-1, keepdim=True)
def combine(D1, D2):
    """Combines two consecutive distance-levels (1 plus log of linear dist) into one."""
    return torch.logaddexp(D1, D2)



def net(ins, outs):
    return nn.Sequential(
        nn.Linear(ins, action_sz), nn.ReLU(), nn.LayerNorm(action_sz),
        SkipConnection(nn.Linear(action_sz, action_sz), nn.ReLU(), nn.LayerNorm(action_sz)),
        SkipConnection(nn.Linear(action_sz, action_sz), nn.ReLU(), nn.LayerNorm(action_sz)),
        nn.Linear(action_sz, outs),
    ).to(device)
embed = net(N*N, emb_sz) # An locally-isometric map of the environment, learned by preserving distances.
#   (Not only single-step dists but all dists: otherwise, a coloring (like a checkerboard pattern in this 2D board env) would have been enough.)
#   (Downside: all distances are assumed to be symmetric.)
act = nn.Sequential( # (prev_emb, dst_emb) → action
    # (With an `embed`ded world model, `act`'s per-src decision boundaries may be simplified a lot: to the action-to-Nearest-Neighbor-of-dst.)
    #   (Evidence: we can reach 80% reachability for N=8 when we learn *all* distances, but only 30% when we only learn single-transition distances to be 1.)
    net(emb_sz + emb_sz, action_sz),
    nn.LayerNorm(action_sz, elementwise_affine=False),
).to(device)
opt = torch.optim.Adam([*embed.parameters(), *act.parameters()], lr=1e-3)



def show_dist_and_act(plt, key):
    """An image of the *learned* distance (0…+∞) and actions (0|1|2|3)."""
    with torch.no_grad():
        src = torch.eye(N*N, N*N, device=device)
        dst = torch.eye(1, N*N, device=device).expand(N*N, N*N)
        emb1, emb2 = embed(src), embed(dst)
        dists = dist(emb1, emb2).reshape(N, N)
        zeros = torch.zeros_like(dists)
        maxes = torch.ones_like(dists) * dists.max()
        acts = act(cat(emb1, emb2))[..., 0:4].argmax(-1).reshape(N, N)+1
        plt.imshow(torch.cat((dists, zeros, maxes), -1).cpu().numpy(), label=key, cmap='nipy_spectral')
        x, y = torch.arange(2*N, 3*N), torch.arange(0, N)
        u = torch.where(acts==1, 1, torch.where(acts==2, -1, 0)).reshape(N,N)
        v = torch.where(acts==3, 1, torch.where(acts==4, -1, 0)).reshape(N,N)
        plt.quiver(x, y, u.cpu(), v.cpu(), scale=1, scale_units='xy', angles='xy', units='xy')
def get_act(src_emb, dst_emb):
    if iters % 100 < 50 and random.randint(1, 10) <= 3:
        action = torch.randn(batch_size, action_sz, device=device)
    else:
        action = act(cat(src_emb, dst_emb))
    return action



# The main unrolling + training loop.
for iters in range(50000):
    # Sample a batch of trajectories.
    board = env_init(N, batch_size=batch_size)
    reached = torch.full((batch_size, 1), False, device=device)
    with torch.no_grad():
        # First pick the target to go to.
        target = env_init(N, batch_size=batch_size)
        dst_emb = embed(target)
        action = get_act(embed(board), dst_emb)

        unroll = [(0, board, action.detach())]
        for u in range(unroll_len):
            # Do the RNN transition (and an environment step), `unroll_len` times.
            board = env_step(N, board, action[..., 0:4])
            action = get_act(embed(board), dst_emb)

            reached |= (board == target).all(-1, keepdim=True)
            unroll.append((u+1, board, action))

        # Save random faraway A → … → B pairs in the replay buffer.
        for _ in range(unroll_len):
            i = random.randint(0, len(unroll)-3)
            j = random.randint(i+1, len(unroll)-2) if random.randint(1,2)!=1 else i+1
            k = random.randint(j+1, len(unroll)-1)
            A, B, C = unroll[i], unroll[j], unroll[k] # (D, board, next_action)
            D12 = torch.full((batch_size, 1), float(B[0]-A[0]), device=device)
            D23 = torch.full((batch_size, 1), float(C[0]-B[0]), device=device)

            rb = (rb+1) % len(replay_buffer)
            replay_buffer[rb] = (A[1], A[2], D12, B[1], B[2], D23, C[1])
            #   (s1, action1, D12, s2, action2, D23, s3)

    # Replay from the buffer. (Needs Python 3.6+ for our convenience.)
    choices = [c for c in random.choices(replay_buffer, k=updates_per_unroll) if c is not None]
    if len(choices):
        s1      = torch.cat([c[0] for c in choices], -2)
        action1 = torch.cat([c[1] for c in choices], -2)
        D12     = torch.cat([c[2] for c in choices], -2)
        s2      = torch.cat([c[3] for c in choices], -2)
        action2 = torch.cat([c[4] for c in choices], -2)
        D23     = torch.cat([c[5] for c in choices], -2)
        s3      = torch.cat([c[6] for c in choices], -2)

        e1, e2, e3 = embed(s1), embed(s2), embed(s3)
        d12, d23, d13 = dist(e1, e2), dist(e2, e3), dist(e1, e3)
        # D12, D23, D13 = D12.log()+1, D23.log()+1, (D12 + D23).log()+1 # (Log-space. Often performs worse.)
        D12, D23, D13 = D12, D23, D12+D23 # (Lin-space. Ends up very cleanly arranged.)

        a12, a13 = act(cat(e1,e2)), act(cat(e1,e3))

        def loss_dist(d,D):
            # Always nonzero, but fades if dist is too high; prefers lower dists.
            mult = (d.detach() - D) + 1
            mult = torch.where( D>1.1, 1 * torch.where(mult>0, mult+1, mult.exp()).clamp(0,15), torch.tensor(1., device=device) )
            return (mult * (d - D).square())
        def loss_act(d,D, a,A):
            # Tries to cut off anything not-min-dist, if in lin-space.
            mult = torch.where( D>1.1, (d.detach() - D + 1).clamp(0,15), torch.tensor(1., device=device) )
            return (mult * (a - A).square())

        # Learn shortest distances, and shortest-actions and combined-plans.
        dist_cond = D13 < (d12+d23)
        dist_target = torch.where(dist_cond, D13, (d12+d23).detach())
        act_target = torch.where(dist_cond, action1, a12.detach())
        l_dist = loss_dist(d12, D12) + loss_dist(d23, D23) + loss_dist(d13, dist_target)
        l_act = 0
        l_act = l_act + torch.where(D12<1.1,1.,0.)*loss_act(d12, D12, a12, action1)
        l_act = l_act + (1/16) * loss_act(d13, dist_target, a13, act_target)
        #   (Not weighing is a bit worse, but not too terrible.)
        l_act = l_act*3



        # N=16: 60% at 14k, 90% at 20k; 85% at 12k and no more; 85% at 14k and no more. (Flatline for 5k.)



        # Base-only actions, N=12:
        #   Logarithmic-space: 75% at 10k. OR WORSE.
        #        Linear-space: 80% at 5k, 85% at 6k. 2× the efficiency of full-action-plans.
        # Weighing meta-action-loss by 1/16, N=16:
        #   Not: 8% at 6k, 45% at 15k, 50% at 20k. (…With later code: 70% at 10k; 60% at 10k.)
        #   Yes: 35% at 6k, 50% at 8k, 60% at 10k. Smooth improvement. (But disconnected arrow regions are still a problem.) (…With later code: ≈80% at 10k.)
        # `action1` as `act_target`, N=16:
        #   Yes: 60% at 10k; 70% at 10k. (Doesn't allow reusing solved subtasks.)
        #   Not: 80% at 10k; 50% at 10k. (Exp-improvement, but this board env is probably way too small.)
        # Min-dist between replay and predicted action as `act_target`:
        #        85% at 10k; 15% at 8k, 75% at 10k; 75% at 10k. (Seems a bit better.)
        # (Our code-configuration is the best we could find, though not by much.)
        # (After those runs, there were some code changes, so final performance may differ.)
        # (Distance seems to converge faster than actions, so if we allowed ourselves to enumerate all actions at each cell instead of learning `act`, convergence would have been 2× or 3× faster.)

        # Broken (swapped) dist-gating for meta-actions:
        #   60% at 10k; 20% at 10k, 50% at 12k; 40% at 10k. (Worse.)
        # With meta-action gating that only lets in strictly-lower distances (still with a multiplier of 1):
        #   75% at 10k; 45% at 10k; 65% at 10k; 65% at 10k. (Func approx likely screws this up.)



        # Optimize.
        (l_dist + l_act).sum().backward()
        opt.step();  opt.zero_grad(True)
        with torch.no_grad(): # Print metrics.
            log(0, False, dist = show_dist_and_act)
            log(1, False, l_dist = to_np(l_dist.sum()))
            log(2, False, l_act = to_np(l_act.sum()))
            log(3, False, reached = to_np(reached.float().mean()))



finish()