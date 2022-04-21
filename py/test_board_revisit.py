"""
Here, we revisit the 2D board env, since continuous-control failed.

Open-ended algorithms, to prepare for any goal, learn min-distance paths between *all* source and goal states. In theory, we've *seen* where we've been, so it should be trivial to revisit them, right? In practice, we haven't been able to break away from learning dist-sums, so far, which is probably non-scalable to real-world environments, because it needs to learn quadratically-many numbers — not to mention, not very biologically plausible due to magnitude & precision.

This time:

- To remove prediction-interference: the env's action needs 4 numbers, and is decided by which is higher.

- REMOVED everything below for now:

- For grounding, remembered to ensure that `act(prev, goal=next) = action` for all transitions, by setting distance to 0 in these cases.

- Self-imitation now *gates* by old-distance-is-better, doesn't multiply by the distance-differential.

- Now have tree-backup (basically multi-step returns so that dist-sum bootstrapping has more accurate targets), which does improve dist-learning speed.

Unimplemented:

- Classical pathfinding but for ML: Dijkstra-algo-like horizons of tasks that are neither too easy (visited & settled) nor too hard (never neighboring any visited state), like in AMIGo or POET or [iterative deepening](https://en.wikipedia.org/wiki/Iterative_deepening_depth-first_search). Goals are just randomly selected.
    - (For state-visiting, apply "neighbor's goal can be our own goal too" via an src→goal GAN and an (src,goal)→action net: could initialize the GAN to always output the state as the goal, and to update, make the neighbor generate its goal-suggestion, and make the original-state discriminator judge that as real and make `act` with the generated goal predict the action that's in the picked replay-buffer step; like a GAN, its own generated goals should be judged as fake (*possibly* only if the action-learning loss is sufficiently low). Doesn't even learn the distance, instead just tries to crystallize closest-actions first.)
        - (…Actually sounds unexpectedly simple and plausible…)
        - (…Possibly, to revisit less, learn another GAN of what we've generated, and only allow a goal-suggestion if it's judged as secondhand-fake. Or even reuse the original GAN's discriminator to gate suggestions, but with a momentum-delayed copy.)
        - (…Possibly, use not only goal-of-neighbor with a replayed action (which doesn't have to play nice with our goals), but also goal-of-goal with the same action (which kinda increases our reach, exponentially quickly in fact).)
        - TODO: …Try it then, in this board env?
            - TODO: What are the impl details of our first potentially-plausible algorithm here?
                - TODO: At least map the goal-states through a neural net, right? (So that GANs can feasibly output them.)
                - TODO: . . .
            - …Or should we think about joint embeddings more…

- Contraction hierarchies but for ML: unknown joint-embedding schemes, with which we might be able to *not* learn the distance. Ideally, would learn contraction hierarchies, but *how* is unknown.
    - We need to either try to come up with some, or revisit continuous-control with task-horizons.
    - (Isn't the "dst's dst is our dst too" idea related to node contraction's "remove a graph node and join the path; at query-time, only go up the query-hierarchy (probably measuring embedding similarity)"?)
    - (Joint-embedding is probably the only real option to scale up to realistic environments, because the neural nets will no longer have to learn every single quadratically-numerous dist & action, and very slowly update them when they discover unexpected paths; many implicit datapoints can be updated at once if we use joint-embedding.)
    - TODO: Our latest exponential space-filling method is quite reminiscent of contraction hierarchies, isn't it? So, replace this with that, right?

---

# Past attempts to overcome the quadratic bottleneck 

---

- For explicitly discriminating which action comes earlier (for self-imitation), could learn distances or distance-surrogates (given a 'future', possibly BYOL-learned, by notation here):
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
    - (We don't need all-to-all distances, we only need to know whether some distances are definitely worse than others.)
    - A particularly easy impl is: with a `(src, dist) → dst` generative model, sample from it twice (with a random `dist`ance) and update the double-stepped distance if it's shorter than what we had in mind.
      - The `dist`-to-learn could be `floor(log2(actual_dist))`, which reduces the required NN-output precision dramatically.
    - (The problem is that generative models, of non-stationary sparse distributions, are very hard to actually learn.)
  - Replace the `dist(src, dst)` net with `embed(src)` and measure distances (`1+log(steps)`) in embedding space: so, learn a locally-isometric map of the env's topology. Preserve faraway states, along with inputs and first-action and distance. Then, we can just learn the min-dist `act(src, dst)`.
    - (The dead-simple "learn the actions" performs worse than also learning the dist map, so representation-learning is important.)
    - Seems to perform best with `A→B→C` double-stepping: `act(A,C) = act(A,B).detach()` (in addition to single-step `act(prev,next) = prev→next` grounding).
"""



from model.log import log, clear, finish
from model.gan import GAN



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

unroll_len = N

rb, replay_buffer = 0, [None] * 1024
updates_per_unroll = 1 # Each replay-buffer's entry is `unroll_len` steps long and of width `batch_size`.

noise_sz = 16
dist_levels = 8 # …Wait, if we won't have `dst`, then this will no longer be required…
emb_sz = 64

def dist(x,y):
    """Calculates the distance in embedding-space, which by training, should be made into `1+log(path_length)` between board-inputs; for prev→next transitions, the path length is 2 (for simplicity of math).

    By not learning a neural net to output the distance, we learn an actual topological model of the environment, in which shortest-path actions are the ones closest to straight lines to where we want to go (so, little info for `act` to remember)."""
    return (x - y).square().mean(-1, keepdim=True)
def combine(D1, D2):
    """Combines two consecutive distance-levels (1 plus log of linear dist) into one."""
    return torch.logaddexp(D1, D2)

# TODO: Eventually, also try inputting not one-hot embeddings of boards, but their xy coords. Should be easy to learn, right?



# (The below is old news, to compare the future impl to…)
#   Bootstrapping, `f(next) = p * f(prev) + local_metric(next)`:
#   (N=8: 40% at 5k, 53% at 10k, 53% at 20k. BIAS)
perfect_distance_targets = True # Replaces dist-bootstrapping with `distance` calls.
#   (N=4: 90% at 5k, 99% at 10k.)
#   (N=8: 55% at 5k, 90% at 10k.)
#   (N=16: 25% at 10k, 80% at 20k, 95% at 30k.)
#   (Actually pretty good behavior. Though we do use ridiculously-large batch sizes.)
#   (Shows that bad dist-bootstrapping is the reason for our poor performance.)
perfect_distance = True # Makes self-imitation use perfect distance, not bootstrapped.
#   (N=4: 99% at 5k.)
#   (N=8: 65% at 10k, 90% at 20k, 95% at 25k.)
#   (N=16: 10% at 5k, 15% at 20k.)
#   (Adding more layers to `act` doesn't help. Neither does more hidden-layer-size.)
#   (Worse than imperfect-distance, somehow.)



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
mid = net(emb_sz + emb_sz, emb_sz) # TODO:
# mid = GAN(net(emb_sz + emb_sz + noise_sz, emb_sz), net(emb_sz + emb_sz + emb_sz, 1), noise_sz=noise_sz) # (src, dst) → mid
#   Returns a midpoint halfway through.
#     Necessary to ever increase `dist`, by generating & comparing candidate midpoints.
#     (A future candidate for a non-GAN solution, since we really don't need many midpoints? May want BYOL first though.)
dst = GAN(net(emb_sz + 1 + noise_sz, emb_sz), net(emb_sz + 1 + emb_sz, 1), noise_sz=noise_sz)
dst_encode2 = net(emb_sz + 1 + emb_sz, noise_sz) # (src, dist, dst) → noise
dst_encode = net(emb_sz + 1 + emb_sz, 2 * noise_sz) # (src, dist, dst) → mean_and_stdev
dst_decode = net(emb_sz + 1 + noise_sz, emb_sz) # (src, dist, noise) → dst
#   A conditioned VAE.
#   Sample two same-distance destinations, and the middle one is the midpoint to compare.
#     (Good for scalability: in high-dimensional spaces, the probability of double-step revisiting single-step's territory vanishes, so useful learning occurs more often.)
#     (If using `future`s, it's a tiny bit like the BYOL loss for faraway targets, but a GAN instead of being conditioned on random-goal actions/plans.)
opt = torch.optim.Adam([*embed.parameters(), *act.parameters(), *mid.parameters(), *dst.parameters(), *dst_encode.parameters(), *dst_decode.parameters()], lr=1e-3)



# These funcs use domain knowledge; used for debugging.
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
def xy(board):
    ind = board.argmax(-1, keepdim=True)
    x = torch.div(ind, N, rounding_mode='floor')
    return x, ind - x*N
def from_xy(x,y): # → board
    return nn.functional.one_hot((x.long().squeeze(-1) % N)*N + (y.long() % N).squeeze(-1), N*N).float()
def perfect_dist(b1, b2):
    """Analytic distance between boards. Used for seeing what the theoretical max performance is."""
    (x1, y1), (x2, y2) = xy(b1), xy(b2)
    d = (x1-x2).abs() + (y1-y2).abs()
    for ox in range(-1, 2):
        for oy in range(-1, 2):
            (x3, y3) = (x1+ox*N, y1+oy*N)
            d = d.min((x3-x2).abs() + (y3-y2).abs())
    return d
def perfect_dst(src, level): # → dst
    """What `dst` should converge to: concentric ever-thicker rings."""
    x,y = xy(src)
    min_dist = 2**level # 2**level … 2**(level+1)-1
    r = torch.rand(level.shape, device=device)
    D = torch.floor(min_dist + r * min_dist) # |x₁-x₂|+|y₁-y₂|
    DX = torch.floor(torch.rand_like(D) * (D+1))
    DY = D - DX
    DX = torch.where(torch.rand_like(D) < .5, DX, -DX)
    DY = torch.where(torch.rand_like(D) < .5, DY, -DY)
    # (The probability of drawing DX==0|DY==0 is twice as high as any other point on the ring, but it shouldn't matter.)
    return from_xy(x+DX, y+DY)
def perfect_mid(src, dst): # → mid
    """What `mid` should converge to: some midpoint that's on the path from source to destination."""
    # Avg the coords, but also handle wrapping by picking the closest-avg for a reflected point.
    (x1,y1), (x2,y2) = xy(src), xy(dst)
    mx, my, mD = None, None, None
    for dx in range(2):
        for dy in range(2):
            x,y = torch.floor((x1+x2)/2 + dx*N/2) % N, torch.floor((y1+y2)/2 + dy*N/2) % N
            D = torch.min((x-x1).abs() + (y-y1).abs(), (x-x2).abs() + (y-y2).abs())
            if mD is None:
                mx, my, mD = x, y, D
            else:
                cond = D < mD
                mx, my, mD = torch.where(cond, x, mx), torch.where(cond, y, my), torch.where(cond, D, mD)
    return from_xy(mx, my)
def get_act(src_emb, dst_emb):
    if iters % 100 < 50 and random.randint(1, 10) <= 3: action = torch.randn(batch_size, action_sz, device=device) # TODO:
    else: action = act(cat(src_emb, dst_emb))
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
            j = random.randint(i+1, len(unroll)-2)
            k = random.randint(j+1, len(unroll)-1) # TODO: Maybe also try saving consecutive things in 50% of cases, to better ensure grounding?
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
        # D12, D23, D13 = D12.log()+1, D23.log()+1, (D12 + D23).log()+1
        D12, D23, D13 = D12, D23, (D12 + D23) # TODO: (Lin-space. Very cleanly arranged.)

        a12, a23, a13 = act(cat(e1,e2)), act(cat(e2,e3)), act(cat(e1,e3))

        # TODO:
        def loss_dist(d,D):
            # Always nonzero, but fades if dist is too high; prefers lower dists.
            mult = (d.detach() - D)
            mult = torch.where( D>1.1, 1 * torch.where(mult>0, mult+1, mult.exp()).clamp(0,15), torch.tensor(1., device=device) )
            return (mult * (d - D).square())
        def loss_act(d,D, a,A):
            # Tries to cut off anything not-min-dist, if in lin-space.
            mult = torch.where( D>1.1, 1 * (d.detach() - D + 1).clamp(0,15), torch.tensor(1., device=device) )
            return (mult * (a - A).square())

        # Learn shortest paths and shortest distances.
        l_dist = loss_dist(d12, D12) + loss_dist(d23, D23) + loss_dist(d13, D13)
        l_act = torch.where(D12<1.1,1.,0.)*loss_act(d12, D12, a12, action1) + torch.where(D23<1.1,1.,0.)*loss_act(d23, D23, a23, action2) + (1/16) * loss_act(d13, D13, a13, a12.detach())
        #   …What we're really missing here is the ability to action-condition the distance, so that we don't have to rely on sampled dist (which is random beyond the immediate horizon of what we've solved, effectively removing our ability to exp-combine)… …But isn't using the measured distance, in fact, the better form of action-cond, namely, task-cond?
        l_act = l_act*3



        # …The code above, for N=12:
        #   With N=8 and just-as-old-code (including the closer-predicted-action as a target), 90% at 8k, same as pairs. But the same code with N=12? 8% at 6k, 45% at 15k, 50% at 20k.
        #     Why is this so much worse than only-base actions?
        #       Linear-space: …80% at 5k, 85% at 6k. 2× the efficiency of full-action-plans.
        #       Logarithmic-space: 75% at 10k. OR WORSE.
        #   …What about just weighing the non-base part by 1/16? 35% at 6k, 50% at 8k, 60% at 10k. Smooth improvement.
        #     …But, disconnected arrow regions are still a huge problem…
        #   …Weighing the non-base part by 1/16 (now with (+1) in losses): 80% at 10k. 2nd run: 50% at 10k.
        #     Linear-space with second-order-dist-weighing:  85% at 7k; no more.
        #       (`D13.min(combine(d12-1,d23-1).detach()+1)`)
        #     Linear-space with first-order-dist-weighing:   85% at 8k, 90% at 9k, 95% at 12k.
        #     Linear-space with min-of-orders-dist-weighing: 90% at 10k, 95% at 11k; no more.
        #       (I like this one the most, for now: initially-unstable, but bubbles would get punctured.)
        #     Linear-space with max-of-orders-dist-weighing: 90% at 7k; no more.
        #     …Same performance everywhere… Starting to think that this env is too trivial, again…
        #   …What about using `action1` as the meta-target, but mult by 1/16? 60% at 10k; 70% at 10k.
        #     (Doesn't allow exp-combining of bringing in a solved subtask. Needs linear-time to expand.)



        # …What if we had not `act(src,dst)` but `act(src,dst,max_dist)`? And, `act(A,B, D1+D2) = act(A,C, D1+D2) = act(A,B, D1)`, dist-diff-gated? Would this solve our (hypothesized) interference issues?
        #   …Can it be, one target is predicted-act, one actual? What would that correspond to?
        # …Or, `act(src,dst, imitate_until_dist)`, `act(A,C, D1+D2) = action` and `act(A,C,D1) = act(A,B,D1)`? …The latter loss would ideally be taking the dist-min among all actions… If our dist isn't action-conditioned, then we can't compare on/off-policy actions… Except by dist-diff…



        # …What if we do have two `act` nets, and gate task-combining by how well the first task is learned?



        # TODO: …Does distance even matter?… What if mult_act was 1?…
        #   …It still works: 95% at 14k… Unflattening at 4k…
        #   …This env is hilariously simple, then.
        #   90% at 9k, 95% at 10k, though it's high-variance (possibly because isolated regions can form and are hard to break).
        #   TODO: Try torch.where(D < 1.1, 1., 0.). (Ground-only.)
        #     85% at 9k.
        #   Apparently, distance matters, insofar as it provides representations.
        # TODO: Try weighing actions by reverse-distance, so that shorter plans are learned first?…
        #   95% at 11k. A bit better I guess?
        #   90% at 7k; high-variance.
        #   TODO: Try 1/D**2.
        #     …Encountered a 50% local minimum, where there was an isolated region of arrows… (I guess "longer paths are less likely and thus won't get reinforced" is only *likely*, not guaranteed.)
        #     90% at 8k, 95% at 9k.
        #   TODO: Try 1/D**3.
        #     90% at 8k, 95% at 10k.
        #   TODO: Try torch.where(D < 1.1, 1., 1/16), like it was in the prior loss.
        #     90% at 8k.
        #     No significant advantages.
        #   TODO: Try torch.where(D < 1.1, 1., 0.) * ×. (Ground-only.)
        #     80% at 9k.
        # TODO: Re-run dist-diff-weighted act-learning.
        #   95% at 9k. No significant advantages.
        #   80% at 10k.
        #   TODO: Try 1/D**2.
        #     90% at 8k.
        # TODO: …Try distance-learning-less.
        #   Complete failure, at 10k.
        #   TODO: Try 1/D**2.
        #     Almost complete failure: 20% at 10k.
        #   TODO: Try torch.where(D < 1.1, 1., 1/16).
        #     Almost complete failure: 20% at 11k.
        # TODO: …Try with the actual boards as inputs to `act`.
        #   60% at 10k, 75% at 12k, 90% at 16k.
        #   TODO: Try 1/D**2.
        #     50% at 10k, 70% at 16k.
        #   TODO: Try torch.where(D < 1.1, 1., 1/16).
        #     40% at 16k, 45% at 20k.
        #   TODO: Try torch.where(D < 1.1, 1., 0.). (AKA grounding-only. If it can reach >25% performance, then we have no ground to stand on when saying "dist-diff-weighting gives a significantly nicer representation".)
        #     Failing with 25% at 20k, as expected.
        #   Always worse than with dist-learning, so I guess that does do something useful.
        # TODO: Try dist-diff-weighting, but without detaching the inputs of `act`.
        #   80% at 9k, 95% at 14k.
        #   TODO: Try 1.
        #     95% at 10k.
        #   TODO: Try 1 * ×.
        #     95% at 10k. (Dist-diff-weighting doesn't make a difference then? Or, only matters to reach near-99%? I guess it would matter a lot more in very long paths, especially in envs with many episodes in one, where we'd need to discard different-episode paths via learned-dist in order to not smudge everything.)
        #   TODO: Try torch.where(D < 1.1, 1., 0.). (Ground-only.)
        #     65% at 10k, 75% at 20k.
        #   TODO: Try (+1) instead of (+2) in actions.
        #     85% at 10k, 95% at 11k.
        #   TODO: Try 1/D**2.
        #     ✓ 90% at 8k, 95% at 9k.
        # TODO: …Run with N=12…
        #   TODO: (1/D**2)×: 65% at 14k. No more. Pretty bad, compared to the old code.
        #   TODO: 1×: 10% at 2k, 10% at 12k, 45% at 20k. Really bad.
        #     TODO: With `act`-arg detaching? 15% at 10k, 45% at 20k. Just as bad.
        #   (Seeing lots of swathes of whole isolated arrow-rings. And distance is misshapen, as if those rings are closer.)
        #   Doesn't feel solved, anymore.
        #   …So are triplets essential to good performance (and exponential-combining, not via hoping that NNs generalize to nearby destinations, but via *actually* using midpoint destinations)?…
        # TODO: …Run with N=16…
        #   TODO: (1/D**2)×: complete failure: 7% at 20k.
        #   TODO: 1×: total failure: 7% at 20k, then quickly unflattens to still-35%-at-50k.
        #     (Seeing *3* isolated arrow-rings, which I guess explains 33% reachability.)
        # mult_dist = 1 * (d12-D12 + 2).detach().clamp(0,15)
        # mult_act  = 1 * (d12-D12 + 2).detach().clamp(0,15)
        # l_dist = (mult_dist * (d12 - D12).square()).sum()
        # l_act = (mult_act * (act(cat(e1, e2)) - action1).square()).sum()
        # TODO: …Is the meager code above truly able to learn as much as all the code below?…
        #   (And if not, why not?)
        #   …80% at 10k and 95% at 13k for N=8, unflattens at 4k… Such a slowdown… Why?
        #     We're able to get to 95% at 9k in the best case, so we're only 2× slower. Acceptable, really. (The only difference from old code is that we use `action` instead of a predicted-`act`ion as the target.)

        # TODO: …Also make a note about the "only learning ground-actions: pretty good with full distance learning, trash with only ground-dist-learning" phenomenon (particularly, about the need to re-test it in this setting), then remove the old code.

        # Optimize.
        (l_dist + l_act).sum().backward()
        opt.step();  opt.zero_grad(True)
        with torch.no_grad(): # Print metrics.
            log(0, False, dist = show_dist_and_act)
            log(1, False, l_dist = to_np(l_dist.sum()))
            log(2, False, l_act = to_np(l_act.sum()))
            log(3, False, reached = to_np(reached.float().mean()))


        # …Possibly, log [NAS-WithOut-Training](https://arxiv.org/pdf/2006.04647.pdf) score, where in a batch and in a NN layer, we compute per-example binary codes of ReLU activations (1 when input>0, 0 when input<0), then compute the sum of pairwise abs-differences?

        # …If `embed`-dist-learning is kinda like multiscale BYOL without the predictor, should we add that predictor (possibly conditioned on the dist, because we don't care if it's unusable directly post-training), so that a post-step repr can end up shifted wherever it wants?… Or is that too much…












        # # TODO: action1/board1/…, and dist1/…, no semantic names.
        # #   TODO: …Actually, should just have `src action D dst`.
        # prev_action = torch.cat([c[0][2] for c in choices], -2)
        # prev_board = torch.cat([c[0][1] for c in choices], -2)
        # action = torch.cat([c[1][2] for c in choices], -2)
        # board = torch.cat([c[1][1] for c in choices], -2)
        # # TODO: Also load distances, now. (And use them. …With `perfect_dst`, we used to use `dist` for estimation; now we can use *actual* distances, right? Wouldn't this mean faster convergence? …But should we adjust both small-distances too, or only their sum?)
        # B = board.shape[-2]

        # prev_emb, emb = embed(prev_board), embed(board)

        # # Ground.
        # #   TODO: Don't "ground" since board1 does not necessarily go directly to board2; instead, should do the same thing with this as with meta, namely, weighing prediction by how much the stored-dist is worse than the predicted-dist, right?…
        # #     …Wait a second: how is it different from the meta-loss, then?…
        # #       Can we combine ground & meta losses into one (2, technically)? And if so, then maybe we don't need triplets, only faraway-pairs?…
        # l_ground_act = (act(cat(prev_emb, emb)) - action).square().sum()
        # l_ground_dist = (dist(prev_emb, emb) - 1).square().sum()

        # # Meta/combination: sample dst-of-dst to learn faraway dists & min-dist actions.
        # D = torch.randint(0, dist_levels, (B,1), device=device).float() # Distance.
        # B_board = perfect_dst(prev_board, D)
        # C_board = perfect_dst(B_board, D)
        # A = prev_emb;  B = embed(B_board);  C = embed(C_board)
        # DB, DC = combine(dist(A,B), dist(B,C)), dist(A,C)
        # TARG = (2*2**D).log() + 1 # TODO: To bootstrap, use `DB` here. (Seems to offer neither advantages nor disadvantages, though.)
        # dist_mult = ((DC+2-TARG).detach().clamp(0,15)/1) # TODO:
        # act_mult = dist_mult # ((DC-DB+0).detach().clamp(0,15)/1)+1 # TODO:
        # #   TODO: Re-run with +.5. …Complete failure: 35% at 9k.
        # #   TODO: Re-run with +1. High-variance: 60%|60%|92%|90% at 5k, 75%|90%|95%|95% at 9k.
        # #     ❌ N=16: 17% at 7k, 21% at 10k, 28% at 20k, 30% at 23k, 45% at 30k, 60% at 40k, 70% at 50k
        # #       …The dist map looks very spotty, very slowly converging to the correct picture. Is this a case of high-bias low-variance, since by cutting off high-dist, we're making propagation of changes very slow?
        # #   TODO: Re-run with +2. Pretty good: 85% at 5k, 90% at 9k.
        # #     TODO:✓With act_mult combine(·, 1). Good: 92% at 5k, 95% at 9k.
        # #       ✓ N=12: 80% at 6k
        # #     TODO: With act_mult +1. 85% at 5k, 85% at 9k.
        # #     TODO: With act_mult (+0)+1. Pretty good: 92% at 5k, 85% at 9k.
        # #     TODO: With act_mult (+.1)+1.
        # #       N=16: 60% at 8k, 70% at 11k, 75% at 12k (plateauing for 7k epochs, like everywhere else)
        # #     TODO:✓With act_mult (+1)+1. 85% at 5k, 95% at 9k.
        # #       N=12: 80% at 7k, 90% at 9k
        # #       ✓ N=16: 70% at 9k
        # #     Non-bootstrapped targets: 85% at 5k, 95% at 9k
        # #       N=16: 65% at 9k, 75% at 15k, 80% at 20k, 85% at 25k, 90% at 35k
        # #       N=16: 85% at 9k, 70% at 11k, 85% at 14k (second run; high-variance?)
        # #   TODO: Re-run with +3. Pretty good: 85% at 5k, 85% at 9k.
        # #   TODO: Re-run with +4. Pretty good: 85% at 5k, 88% at 9k.
        # #   TODO: Re-run with (+1)**2. …Bad: only 70% at 9k.
        # #   TODO: Re-run with (+2)**2. Not terrible: 70% at 5k, 80% at 9k.
        # #   TODO: Re-run with (+1)**3. …Complete failure: 40%|45% at 9k.
        # #   TODO: Re-run with (+2)**3. Surprisingly good: 80% at 5k, 90% at 9k.
        # #   TODO: Re-run with ((+2)/2)**4. Not terrible: 85% at 9k.
        # #   TODO:✓Re-run with (+1).exp()-1. Good: 90% at 5k, 97% at 9k.
        # #     ❌ N=12: 35% at 8k
        # #   TODO: Re-run with (+2).exp()-1. Not bad: 80% at 5k, 90% at 9k.
        # #   TODO:✓Re-run with (+2)/3. Good: 90% at 5k, 97% at 9k.
        # #     N=12: 70% at 6k
        # #     TODO: With act_mult +0. 90% at 5k, 95% at 9k.
        # #     TODO: With act_mult +1. 80% at 5k, 85% at 9k.
        # #     TODO: With act_mult (+0)+1. 90% at 5k, 95% at 9k.
        # #     TODO: With act_mult (+1)+1. 90% at 5k, 95% at 9k.
        # #     TODO: With act_mult (+.1)+.9. 85% at 5k, 85% at 9k.
        # #   TODO: Re-run with (+1)+1. 70% at 5k, 80% at 9k.
        # #   TODO: Re-run with combine(DC,1)-DB. …Complete failure: 35% at 5k, 35% at 9k.
        # #   TODO: Re-run with combine(DC,2)-DB. …Failure: 60% at 5k, 60% at 9k.
        # #     TODO: With act_mult combine(·, 1). …Failure: 45% at 5k, 55% at 9k.
        # #   TODO: Re-run with combine(DC,3)-DB. 70% at 5k, 85% at 9k.
        # #   TODO: Re-run with combine(DC,4)-DB. 80% at 5k, 90% at 9k.
        # #   TODO: Re-run with combine(DC,5)-DB. 80% at 5k, 85% at 9k.
        # #   TODO: Re-run with DC*1.5-DB.
        # #     TODO: With act_mult combine(·, 1). Not bad: 65% at 5k, 95% at 9k.
        # #     TODO: With act_mult +1. Not bad: 80% at 5k, 90% at 9k.
        # #     TODO: With act_mult +0. Not bad: 75% at 5k, 90% at 9k.
        # #   TODO: Re-run with DC*2-DB.
        # #     TODO: With act_mult combine(·, 1). Bad: 75% at 5k, 80% at 9k.
        # #     TODO:✓With act_mult +1. 95% at 5k, 95% at 9k.
        # #     TODO: With act_mult +0. 92% at 5k, 90% at 9k.
        # #     (I feel like the speed here is just because we effectively have a higher multiplier of loss.)
        # l_meta_act = (1/16) * (act_mult * (act(cat(A,C).detach()) - act(cat(A,B)).detach()).square()).sum() # TODO:
        # #   (TODO: …With faraway-sampling, we'd be able to use not just a predicted action but the actual taken action here, eliminating a source of instability/bootstrapping…)
        # l_meta_dist = (dist_mult * (DC - TARG.detach()).square()).sum() # TODO:

        # TODO: Run & fix.
        #   TODO: How to fix the only failing component: generative models?
        #     GANs failed.
        #     VAEs failed.
        #     SWAEs failed.
        #     GANs failed again.
        #     So where is the truth in this world?…
        # …Perhaps we ought to preserve far-past samples and distances to them, instead of generating.

        # `embed` with correct distances (AKA learn a topological transformation), instead of learning `dist`:
        #   N=4: 90% at 1.2k, 99% at 2k.
        #   N=8: 90% at 5k, 95% at 6k.
        #   N=16: 90% at 12k, 95% at 14k.
        #   …These use perfect `dst` and `mid`, though.
        #   `mid`point-less:
        #   N=8: 95% at 5k
        #   N=12: 80% at 6k OR 90% at 9k
        #   N=16: 70% at 9k OR 75% at 12k
        #   (Can't seem to get any better. We're kind of in the off-policy regime due to `perfect_dst`; sampling from real close-to-min-dist trajectories might be able to achieve better results.)
        #     (…Though, after removing dist-bootstrapping and using the actual `D`, improvement to 90% for N=16 seems quite possible, even if slow (at 35k).)
        #   (If perfect reachability is needed, could always learn to generate midpoints.)











        # TODO: Our scalability here is atrocious. Be able to scale to the real world: make the goal of our goal *our* goal, AKA compute the transitive closure via embeddings, but only on first visits (otherwise everything will end up connected to everything). Ideally, we'd like to solve all most-trivial tasks (ground), then do nothing but *combine* the tasks — which should give us the exponential-improvement that dist-learning RL is too bogged-down by quadratic-info to have…
        #   …But now, with `embed`, scalability seems to be quite a bit better. There's at least a chance that it's good enough. (A chance which got lower when we forewent `mid`points, but the chance still exists.)


        # Usable for GANs, and for synth grad (least-magnitude): …DDPG's trick of "take the min of 2 nets" is really quite clever, since ReLU-nets are piecewise linear functions, so in non-trained regions, the linear pieces would be getting further and further away from data. Should we use it somewhere, like in `dst` and/or `mid`?





        # TODO: Implement faraway-sampling for once, by saving not just states, but also distances (at arbitrary offsets, preferably exponentially far away), and at least 3 per replay-sample.

        # TODO: Try doing a 3D t-SNE on `embed`dings. (Ideally, we'd see a torus.)



        # best control, for `sn`: max sensitivity to outcomes, *given possible outcomes* (otherwise, involuntary movements would take up bandwidth).
        #   …the ideal control-by-human method is an RNN that doesn't change when the outcome is by far the most likely, and otherwise, changes in a way most distinct from other outcomes… does `leads_to(ev(prev))=sg ev(next)` BYOL-on-RNNs really fit beyond a superficial analysis?… do we need to do separate research on this?
        #   Mathematically, it's maximizing [mutual info](https://en.wikipedia.org/wiki/Mutual_information): `sum(x&y, p(x,y) * log(p(x,y) / (p(x)*p(y))))`: sensitivity to least-probable states & actions, and most-probable state-action pairs.
        #     (Or [pointwise MI](https://en.wikipedia.org/wiki/Pointwise_mutual_information): `log(p(y|x) / p(y))`.)











        # Optimize.
        # (l_ground_act + l_ground_dist + l_meta_act + l_meta_dist).backward()
        # opt.step();  opt.zero_grad(True)
        # with torch.no_grad(): # Print metrics.
        #     log(0, False, dist = show_dist_and_act)
        #     log(1, False, ground_act = to_np(l_ground_act))
        #     log(2, False, ground_dist = to_np(l_ground_dist))
        #     log(3, False, meta_act = to_np(l_meta_act))
        #     log(4, False, meta_dist = to_np(l_meta_dist))
        #     log(5, False, reached = to_np(reached.float().mean()))


        # if iters == 1000: clear() # TODO: Does the reachability plot look like it plateaus after learning 1-step actions (2k…4k updates, 25% with N=8)? …No, it's way slower to reach 25%… Not sure if exponential-growth or stupid…
finish()