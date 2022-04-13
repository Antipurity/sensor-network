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



N, batch_size = 8, 100
action_sz = 64

unroll_len = N

rb, replay_buffer = 0, [None] * 1024
updates_per_unroll = 1 # Each replay-buffer's entry is `unroll_len` steps long and of width `batch_size`.

noise_sz = 16
dist_levels = 8

def combine(D1, D2):
    """Combines two consecutive distance-levels (floor of log-2 of linear dist) into one."""
    d1, d2 = D1.round(), D2.round()
    return torch.where((d1-d2).abs() < 1, d1+1, d1.max(d2))



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
act = nn.Sequential( # (prev_board, target) → action
    net(N*N + N*N, action_sz),
    nn.LayerNorm(action_sz, elementwise_affine=False),
).to(device)
dist = net(N*N + N*N, 1) # (prev_board, target) → floor(log2(future_distance_sum))
#   (Could one day be made a probability distribution too.) (Really minimizing how much we need to remember: action-independent, and quantized. Info-to-remember is kinda linear: way way less separation boundaries than full-dist.)
mid = GAN(net(N*N + N*N + noise_sz, N*N), net(N*N + N*N + N*N, 1), noise_sz=noise_sz) # (src, dst) → mid
#   Returns a midpoint halfway through.
#     Necessary to ever increase `dist`, by generating & comparing candidate midpoints.
#     (A future candidate for a non-GAN solution, since we really don't need many midpoints? May want BYOL first though.)
dst = GAN(net(N*N + 1 + noise_sz, N*N), net(N*N + 1 + N*N, 1), noise_sz=noise_sz) # (src, dist) → dst
#   Sample two same-distance destinations, and the middle one is the midpoint to compare.
#     (Good for scalability: in high-dimensional spaces, the probability of double-step revisiting single-step's territory vanishes, so useful learning occurs more often.)
#     (If using `future`s, it's a tiny bit like the BYOL loss for faraway targets, but a GAN instead of being conditioned on random-goal actions/plans.)
opt = torch.optim.Adam([*act.parameters(), *dist.parameters(), *mid.parameters(), *dst.parameters()], lr=1e-3)

def show_dist(plt, key):
    """An image of the *learned* distance."""
    with torch.no_grad():
        board = torch.eye(N*N, N*N, device=device)
        target = torch.eye(1, N*N, device=device).expand(N*N, N*N)
        action = act(cat(board, target))
        plt.imshow(dist(cat(board, action, target)).reshape(N, N).cpu().numpy(), label=key)
def distance(b1, b2):
    """Analytic distance between boards. Used for seeing what the theoretical max performance is."""
    def xy(board):
        ind = board.argmax(-1, keepdim=True)
        x = torch.div(ind, N, rounding_mode='floor')
        return x, ind - x*N
    (x1, y1), (x2, y2) = xy(b1), xy(b2)
    d = (x1-x2).abs() + (y1-y2).abs()
    for ox in range(-1, 2):
        for oy in range(-1, 2):
            (x3, y3) = (x1+ox*N, y1+oy*N)
            d = d.min((x3-x2).abs() + (y3-y2).abs())
    return d



# The main unrolling + training loop.
for iters in range(50000):
    # Sample a batch of trajectories.
    action = torch.zeros(batch_size, action_sz, device=device)
    board = env_init(N, batch_size=batch_size)
    reached = torch.full((batch_size, 1), False, device=device)
    prev_action, prev_board = action, board
    boards, actions = [board], [action.detach()]
    with torch.no_grad():
        # First pick the target to go to.
        target = env_init(N, batch_size=batch_size)
        for u in range(unroll_len):
            # Do the RNN transition (and an environment step), `unroll_len` times.
            prev_action, prev_board = action, board
            action = act(cat(board, target))
            if iters % 100 < 50 and random.randint(1, 10) <= 3: action = torch.randn(batch_size, action_sz, device=device) # TODO:
            board = env_step(N, board, action[..., 0:4])

            reached |= (board == target).all(-1, keepdim=True)
            rb = (rb+1) % len(replay_buffer)
            replay_buffer[rb] = (prev_action, prev_board, action, board)

    # Replay from the buffer. (Needs Python 3.6+ for convenience.)
    choices = [c for c in random.choices(replay_buffer, k=updates_per_unroll) if c is not None]
    if len(choices):
        prev_action = torch.cat([c[0] for c in choices], -2)
        prev_board = torch.cat([c[1] for c in choices], -2)
        action = torch.cat([c[2] for c in choices], -2)
        board = torch.cat([c[3] for c in choices], -2)
        B = board.shape[-2]

        # Ground.
        l_ground_act = (act(cat(prev_board, board)) - action).square().sum()
        l_ground_dist = dist(cat(prev_board, board)).square().sum()
        z = torch.zeros(B,1, device=device)
        l_ground_dst_g = dst.goal(prev_board, z, board, goal=0)
        l_ground_dst_d = dst.pred(prev_board, z, dst(cat(prev_board, z)), goal=1)
        #   (This GAN likely fails to converge, because the distributions hardly overlap… How to fix it?…)

        # Meta/combination: sample dst-of-dst to try to update midpoints, and learn our farther-dist-dst.
        D = torch.randint(0, dist_levels, (B,1), device=device) # Distance.
        A = prev_board;  B = dst(cat(A,D));  C = dst(cat(B,D));  M = mid(cat(A,C))
        DAM, DMC = dist(cat(A,M)), dist(cat(M,C))
        DB, DM, DC = combine(dist(cat(A,B)), dist(cat(B,C))), combine(DAM, DMC), dist(cat(A,C))
        l_meta_act = (act(cat(A,C)) - torch.where(DB < DM-1, act(cat(A,B)), act(cat(A,M))).detach()).square().sum()
        l_meta_dist = (DC - DB.min(DM).detach()).square().sum()

        # Learn generative models of faraway places.
        A0, C0, M0 = A.detach(), C.detach(), M.detach()
        l_dst_g = dst.goal(A0, D+1, C, goal = D+1) # Get that distance right.
        l_dst_d = dst.pred(A0, D+1, C0, goal = DC)
        l_mid_g = mid.goal(A0, C0, M0, goal = 0) # Minimize non-middle-ness.
        l_mid_d = mid.pred(A0, C0, M0, goal = (DC-1-DAM).abs() + (DC-1-DMC).abs())

        # TODO: Run & fix.
        #   TODO: How to find out what goes wrong with combining plans?
        #     ground_dst_d goes to 0, which is a sign of failure in GANs. As was feared, the distribution of neighboring states is too particular/small to be learned by a GAN… How can we overcome that?…

        # …What about this generative model: `gen` that's the usual prediction-model but also takes randn noise, and `gen_noise` to predict the noise given inputs&output (learned whenever we call `act2`); and to 'add a sample' to the generative distribution, use `act2` with the predicted noise (its action should be what-we-want-to-predict). Needs 2 losses: for `gen` (predict T given noise-for-T) and for `gen_noise` (predict randn given gen-for-randn).
        #   …Why have I never heard of this model? Is it that bad?
        #   …It has many downsides and potential holes; but is it worse than GANs?…
        #   TODO: Try this model.
        #     …But should we try it here, or in `model/gan.py`?… With the latter, we can at least try to somehow measure diversity, right?…

        #   TODO: …If we fail to make progress, then we could simplify: replace the `dst` and `mid` GANs with literal dicts-of-sets (from a tuple of all args to all possible outputs) (both are added-to or removed-from based on the predicted distance), and go through CPU… If everything else works well, then GANs are the problem.











        # TODO: Our scalability here is atrocious. Be able to scale to the real world: make the goal of our goal *our* goal, AKA compute the transitive closure via embeddings, but only on first visits (otherwise everything will end up connected to everything). Ideally, we'd like to solve all most-trivial tasks (ground), then do nothing but *combine* the tasks — which should give us the exponential-improvement that dist-learning RL is too bogged-down by quadratic-info to have…


        # TODO: The filling:
        #   (For each src/dst, we'd like to partition the dst/src space into actions that *first* lead to it, which implies linear capacity (nodes×actions) despite the nodes×nodes input space, not the quadratic capacity of dist-learning (nodes×nodes×actions).)
        #   TODO: BYOL loss, to have a good differentiable proxy for observations (`future`):
        #     TODO: `leads_to(future(prev)) = sg future(next)`
        #   TODO: …How to weigh `dst`'s discriminator-preference by loss, if we're already specializing to distance and non-middle-ness? Should we, even, especially in this most-simple env?
        #   TODO: Also the `src(dst, dist)→src` GAN: exactly the same as src→dst but reversed.
        #     (May be a good idea to fill from both ends, since volumes of hyperspheres in D dims grow by `K**D` times when radii increase by `K` times. Especially good for RL, which is very interested in goal states but may want to explore starting states initially.)
        #   (…DDPG's trick of "take the min of 2 nets" is really quite clever, since ReLU-nets are piecewise linear functions, so in non-trained regions, the linear pieces would be getting further and further away from data. Usable for GANs, and for synth grad (least-magnitude). Should we use it somewhere, like in `dst` and/or `mid`?)



        # best control, for `sn`: max sensitivity to outcomes, *given possible outcomes* (otherwise, involuntary movements would take up bandwidth).
        #   …the ideal control-by-human method is an RNN that doesn't change when the outcome is by far the most likely, and otherwise, changes in a way most distinct from other outcomes… does `leads_to(ev(prev))=sg ev(next)` BYOL-on-RNNs really fit beyond a superficial analysis?… do we need to do separate research on this?











        # Optimize.
        (l_ground_act + l_ground_dist + l_ground_dst_d + l_ground_dst_g + l_meta_act + l_meta_dist + l_dst_d + l_dst_g + l_mid_d + l_mid_g).backward()
        opt.step();  opt.zero_grad(True)
        with torch.no_grad(): # Print metrics.
            log(0, False, dist = show_dist)
            log(1, False, ground_act = to_np(l_ground_act))
            log(2, False, ground_dist = to_np(l_ground_dist))
            log(3, False, ground_dst_d = to_np(l_ground_dst_d), ground_dst_g = to_np(l_ground_dst_g))
            log(4, False, meta_act = to_np(l_meta_act))
            log(5, False, meta_dist = to_np(l_meta_dist))
            log(6, False, dst_d = to_np(l_dst_d), dst_g = to_np(l_dst_g))
            log(7, False, mid_d = to_np(l_mid_d), mid_g = to_np(l_mid_g))
            log(8, False, reached = to_np(reached.float().mean()))


        if iters == 1000: clear()
finish()