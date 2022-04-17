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
dst = GAN(net(N*N + 1 + noise_sz, N*N), net(N*N + 1 + N*N, 1), noise_sz=noise_sz)
dst_encode = net(N*N + 1 + N*N, 2 * noise_sz) # (src, dist, dst) → mean_and_stdev
#   TODO: …Wait, don't we want this to accept `dst` too?…
dst_decode = net(N*N + 1 + noise_sz, N*N) # (src, dist, noise) → dst
#   A conditioned VAE.
#   Sample two same-distance destinations, and the middle one is the midpoint to compare.
#     (Good for scalability: in high-dimensional spaces, the probability of double-step revisiting single-step's territory vanishes, so useful learning occurs more often.)
#     (If using `future`s, it's a tiny bit like the BYOL loss for faraway targets, but a GAN instead of being conditioned on random-goal actions/plans.)
opt = torch.optim.Adam([*act.parameters(), *dist.parameters(), *mid.parameters(), *dst.parameters(), *dst_encode.parameters(), *dst_decode.parameters()], lr=1e-3)



# These funcs use domain knowledge; used for debugging.
def show_dist(plt, key):
    """An image of the *learned* distance."""
    with torch.no_grad():
        board = torch.eye(N*N, N*N, device=device)
        target = torch.eye(1, N*N, device=device).expand(N*N, N*N)
        plt.imshow(dist(cat(board, target)).reshape(N, N).cpu().numpy(), label=key)
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

    # Replay from the buffer. (Needs Python 3.6+ for our convenience.)
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
        ground_dst = dst(cat(prev_board, z))
        l_ground_dst_g = dst.goal(prev_board, z, ground_dst, goal=0) # TODO:
        l_ground_dst_d = dst.pred(prev_board, z, ground_dst, goal=1) + dst.pred(prev_board, z, board, goal=0) # TODO:
        #   (This GAN likely fails to converge, because the distributions hardly overlap… How to fix it?…)
        #   TODO: Reject VAEs, return to GANs.
        #     TODO: Normalize `noise = dst_encode(cat(prev_board, z, board))`, to 0-mean 1-stdev. TODO: …Don't we want to encode `board`?… Should we condition on both boards?… Yeah, should have done that in the first place.
        #     TODO: Loss: `dst(prev_board, z, noise=noise.detach()) = board` (ensure that what we model is *always* in the embedding space, somewhere)
        #       `l_ground_dst_direct`

        # TODO: Learn to generate neighboring boards via `dst_encode` & `dst_decode`, at least.
        #   Why is this so difficult?
        # dst_noise_mean, dst_noise_stdev = dst_encode(cat(prev_board, z, board)).chunk(2, -1) # (The noise helps cover the latent-space with our few samples, but diversity suffers, and accuracy is still bad. Not to mention, still too slow to learn, especially given that mistakes in lower levels compound in higher levels.)
        # dst_noise_stdev = torch.nn.functional.softplus(dst_noise_stdev)
        # dst_noise = dst_noise_mean + torch.randn_like(dst_noise_stdev) * dst_noise_stdev
        # l_ground_dst_g = (dst_decode(cat(prev_board, z, dst_noise)) - board).square().sum()
        # def normal_log_prob(mean, std, z):
        #     var2 = 2 * std.square()
        #     return -.5 * (3.14159265359 * var2) - (z - mean).square() / var2
        # # theta = torch.randn(B, noise_sz, device=device)
        # # theta = theta / theta.square().sum(-1, keepdim=True).sqrt()
        # # theta = theta.t()
        # # l_ground_dst_d = 1 * ((dst_noise @ theta).sort(0).values - (torch.randn_like(dst_noise_stdev) @ theta).sort(0).values).square().sum() # SWAE
        # # TODO: …Okay, what about VAEs then? …Why is this impl so bad.
        # #   (Adapted from https://github.com/altosaar/variational-autoencoder/blob/master/train_variational_autoencoder_pytorch.py)
        # log_q_z = normal_log_prob(dst_noise_mean.detach(), dst_noise_stdev.detach(), dst_noise).sum(-1, keepdim=True) # TODO:
        # log_p_z = normal_log_prob(torch.zeros_like(dst_noise), torch.ones_like(dst_noise), dst_noise).sum(-1, keepdim=True) # TODO: (Penalize divergence from the normal distribution.)
        # l_ground_dst_d = (log_q_z - log_p_z).sum() # VAE
        # #   TODO: …Why are we failing to learn this?… Did we fail at implementing it? (Including log_q_z makes loss diverge, so it seems likely.)
        # # l_ground_dst_d = .1 * dst_noise.mean(0).square().sum() + (dst_noise.std(0) - 1).square().sum() # It's no KL divergence, but eh, so much simpler.
        if iters % 1000 == 0: # TODO: …None of this looks correct at all… Do we need proper KL-divergence after all?…
            BOARD = env_init(N, 1)
            print(*[c.detach().cpu().numpy() for c in xy(BOARD)], 'neighbors:')
            for i in range(6):
                # NEIGH = dst_decode(cat(BOARD, torch.zeros(1,1,device=device), torch.randn(1,noise_sz,device=device)))
                NEIGH = dst(BOARD, torch.zeros(1,1,device=device))
                print(' ', i, *[c.detach().cpu().numpy() for c in xy(NEIGH)], NEIGH)
                # TODO: …Why does this generative model seemingly converge to just one output…
                #   …Unusable…
                # TODO: …What if we sort generated-noise and random-noise along -1, and simply minimize L2 diff? …Wait, this is literally SWAE, isn't it. …Without random projections though.
                #   …Still unusable…

        # Meta/combination: sample dst-of-dst to try to update midpoints, and learn our farther-dist-dst.
        D = torch.randint(0, dist_levels, (B,1), device=device) # Distance.
        A = prev_board;  B = perfect_dst(A,D);  C = perfect_dst(B,D);  M = perfect_mid(A,C)
        DAM, DMC = dist(cat(A,M)), dist(cat(M,C))
        DB, DM, DC = combine(dist(cat(A,B)), dist(cat(B,C))), combine(DAM, DMC), dist(cat(A,C))
        l_meta_act = (act(cat(A,C)) - torch.where(DB < DM-.5, act(cat(A,B)), act(cat(A,M))).detach()).square().sum()
        l_meta_dist = (DC - DB.min(DM).detach()).square().sum()
        # l_meta_act, l_meta_dist = 0,0 # TODO: …Okay, how long does it really take to fully learn single-step transitions? I feel like we *should* be needing less than 10k epochs, right?… Are we held back by still having to learn quadratically-many distances?
        #   1-step-actions learning seems to only take about 2k…4k epochs. So combining is 6k…8k. Isn't this too much?

        # Learn generative models of faraway places.
        A0, C0, M0 = A.detach(), C.detach(), M.detach()
        l_dst_g = 0 # dst.goal(A0, D+1, C, goal = D+1) # Get that distance right. # TODO:
        l_dst_d = 0 # dst.pred(A0, D+1, C0, goal = DC) # TODO:
        l_mid_g = 0 # mid.goal(A0, C0, M, goal = 0) # Minimize non-middle-ness. # TODO:
        l_mid_d = 0 # mid.pred(A0, C0, M0, goal = (DC-1-DAM).abs() + (DC-1-DMC).abs()) # TODO:

        # TODO: Run & fix.
        #   TODO: How to fix the only failing component: generative models?

        # TODO: Try VAEs, since GANs kinda need rich distributions, not ≈3 distinct samples per class/input?
        #   (Need to condition both encoder & decoder on src&dist for `dst` and on src&dst for `mid`; only the output is autoencoded.)
        #   TODO: First do the dst VAE, and see whether we can run & fix it.
        #     Do we need a measure of sharpness (diff from a one-hot encoding) on generated samples? …If we fail, yes.
        #   TODO: Then do mid.

        # TODO: …Possibly try [SWAEs](https://arxiv.org/abs/1804.01947)?
        #   …I think this is literally replacing the KL divergence of VAEs with the L2 loss between random projections of sorted generated (`dst_noise`) & random (`torch.randn`) noises.











        # TODO: Our scalability here is atrocious. Be able to scale to the real world: make the goal of our goal *our* goal, AKA compute the transitive closure via embeddings, but only on first visits (otherwise everything will end up connected to everything). Ideally, we'd like to solve all most-trivial tasks (ground), then do nothing but *combine* the tasks — which should give us the exponential-improvement that dist-learning RL is too bogged-down by quadratic-info to have…


        # TODO: The filling:
        #   (For each src/dst, we'd like to partition the dst/src space into actions that *first* lead to it, which implies linear capacity (nodes×actions) despite the nodes×nodes input space, not the quadratic capacity of dist-learning (nodes×nodes×actions).)
        #   TODO: BYOL loss, to have a good differentiable proxy for observations (`future`):
        #     TODO: `leads_to(future(prev)) = sg future(next)`
        #   TODO: …How to weigh `dst`'s discriminator-preference by loss, if we're already specializing to distance and non-middle-ness? Should we, even, especially in this most-simple env?
        #   TODO: Also the `src(dst, dist)→src` GAN: exactly the same as src→dst but reversed.
        #     (May be a good idea to fill from both ends, since volumes of hyperspheres in D dims grow by `K**D` times when radii increase by `K` times. Especially good for RL, which is very interested in goal states but may want to explore starting states initially.)
        #   (…DDPG's trick of "take the min of 2 nets" is really quite clever, since ReLU-nets are piecewise linear functions, so in non-trained regions, the linear pieces would be getting further and further away from data. Usable for GANs, and for synth grad (least-magnitude). Should we use it somewhere, like in `dst` and/or `mid`?)



        # TODO: Have `embed(board)→emb`, `action_sz`-size.
        # TODO: Make all neural nets (except for `embed`) take embeddings instead of boards.
        # TODO: When replaying transitions, replace all `prev_board` and all `board` uses with `prev_emb` and `emb` respectively.
        # TODO: Remove `dist`, instead consider the L2 distance to be that.
        #   (`embed` would become a proper world model, which preserves topology and shortest-path distances. In 3D t-SNE, a 2D board should become a torus; a 1D board should become an actual circle.)
        #     (Single-step BYOL wouldn't have been enough, since it would have just done coloring, which is counterproductive for having `act`. We need to be multi-scale.)
        #     (With such a world model, `act`'s per-src decision boundaries should become trivial: literally action-to-Nearest-Neighbor-of-dst.)
        #   (With perfect `dst` and `mid`, there should be no change in performance.)

        # TODO: …Should we return to GANs but this time with an output→noise encoder, so that we could ensure that at least one GAN-sample is definitely correct, and random-noise samples at least have a chance at a good gradient?…
        #   Maybe even, the encoder isn't trained, and its output is normalized (and so is the noise)?
        #   (Also, `dst` in particular doesn't need to make its discriminator predict the distance; the discr can just predict whether a value is reachable, and we can/should apply the loss that ensures that the (`dist` or L2) distance is as we requested.)
        #   …I think it's better to do THIS next, because this way we'll still have the argmax visualization.

        # TODO: …Also, maybe we really should make `mid` a simple predictor, learned whenever a new midpoint's distance is better?



        # best control, for `sn`: max sensitivity to outcomes, *given possible outcomes* (otherwise, involuntary movements would take up bandwidth).
        #   …the ideal control-by-human method is an RNN that doesn't change when the outcome is by far the most likely, and otherwise, changes in a way most distinct from other outcomes… does `leads_to(ev(prev))=sg ev(next)` BYOL-on-RNNs really fit beyond a superficial analysis?… do we need to do separate research on this?
        #   Mathematically, it's maximizing [mutual info](https://en.wikipedia.org/wiki/Mutual_information): `sum(x&y, p(x,y) * log(p(x,y) / (p(x)*p(y))))`: sensitivity to least-probable states & actions, and most-probable state-action pairs.
        #     (Or [pointwise MI](https://en.wikipedia.org/wiki/Pointwise_mutual_information): `log(p(y|x) / p(y))`.)











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


        # if iters == 1000: clear() # TODO: Does the reachability plot look like it plateaus after learning 1-step actions (2k…4k updates, 25% with N=8)? …No, it's way slower to reach 25%… Not sure if exponential-growth or stupid…
finish()