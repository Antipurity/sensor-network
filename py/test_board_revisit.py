"""
Here, we revisit the 2D board env, since continuous-control failed.

Open-ended algorithms, to prepare for any goal, learn min-distance paths between *all* source and goal states. In theory, we've *seen* where we've been, so it should be trivial to revisit them, right? In practice, we haven't been able to break away from learning dist-sums, so far, which is probably non-scalable to real-world environments, because it needs to learn quadratically-many numbers — not to mention, not very biologically plausible due to magnitude & precision.

This time:

- To remove prediction-interference: the env's action needs 4 numbers, and is decided by which is higher.

- For grounding, remembered to ensure that `act(prev, goal=next) = action` for all transitions, by setting distance to 0 in these cases.

- Self-imitation now *gates* by old-distance-is-better, doesn't multiply by the distance-differential.

- Now have tree-backup (basically multi-step returns so that dist-sum bootstrapping has more accurate targets), which does improve dist-learning speed, though due to implementation details, .

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



N, batch_size = 8, 100
action_sz = 64

unroll_len = N

replay_buffer = [None] * 1024
updates_per_unroll = 1 # Each replay-buffer's entry is `unroll_len` steps long and of width `batch_size`.

bootstrap_discount = torch.tensor([.99], device=device)
#   Bootstrapping is `f(next) = THIS * f(prev) + local_metric(next)`
#   (Predicting many discounts at once doesn't help.)
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



act = nn.Sequential( # (prev_board, target) → action
    nn.Linear(N*N + N*N, action_sz), nn.ReLU(), nn.LayerNorm(action_sz),
    SkipConnection(nn.Linear(action_sz, action_sz), nn.ReLU(), nn.LayerNorm(action_sz)),
    SkipConnection(nn.Linear(action_sz, action_sz), nn.ReLU(), nn.LayerNorm(action_sz)),
    nn.Linear(action_sz, action_sz),
    nn.LayerNorm(action_sz, elementwise_affine=False),
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
    """An image of the *learned* distance."""
    with torch.no_grad():
        board = torch.eye(N*N, N*N, device=device)
        target = torch.eye(1, N*N, device=device).expand(N*N, N*N)
        action = act(cat(board, target))
        dist = future_dist(cat(board, action, target)).reshape(N, N)
        plt.imshow(dist.cpu().numpy(), label=key)
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

            board = env_step(N, board, action[..., 0:4])

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
        #   (Random actions contain a lot of noise, so distance for them may not transfer to distance for `act`'s actions.)
        with torch.no_grad():
            if not perfect_distance_targets:
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
            else:
                dists_are = distance(boards[1:], targets[1:])
        dist_pred_loss = (dists[:-1] - dists_are.detach()).square().sum(-1).mean()
        # Self-imitation gated by min-dist, by `act`.
        #   (A lot like [SIL.](https://arxiv.org/abs/1806.05635))
        with torch.no_grad():
            if not perfect_distance:
                cond = dists < dists2
            else:
                next_boards = torch.cat((boards[1:], boards[-1:]), 0).detach()
                cond = distance(next_boards, targets) < distance(boards, targets)
            next_actions_are2 = torch.where(cond, next_actions, next_actions2)
        self_imitation_loss = (next_actions2 - next_actions_are2).square().sum(-1).mean()










        # TODO: Our scalability here is atrocious. Be able to scale to the real world: make the goal of our goal *our* goal, AKA compute the transitive closure via embeddings, but only on first visits (otherwise everything will end up connected to everything).


        # TODO: Organize the possible ideas on how contraction hierarchies could be done.

        # Contraction hierarchies:
        #   - Classically: first preprocess using node contraction: in the order of node importance (always a heuristic, but important to query-time performance), create a new hierarchy level: remove a node/state, and add all its incoming+outcoming edges/actions as shortcuts. And at query-time, meet at the least-abstract level: always go up a node-importance level, go from both ends, and pick the min-dist-sum meeting node.



        # For explicitly discriminating which action comes earlier (for self-imitation), could learn distances or distance-surrogates (given a 'future', by notation here):
        #   - Learn all the actual pairwise distances. (Quadratically-many numbers to learn: too much.)
        #     - (But compared to any of the options below, it's amazing.)
        #   - Learn a goal-dependent embedding `ev` for which `ev(prev,g) = prev_ev(ev(next,g))` and `ev(g,g) = OK` (make it smoothly transform into goals), and compare func-call-depths when needed:
        #     - Learn the actual distance given `ev`.
        #     - Learn a comparator of 2 `ev`s, `cmp(fut1, fut2)→-1…0…1`: `cmp(RNN(f),f)=-1`, `cmp(f,f)=0`, `cmp(f,RNN(f))=1`, `cmp(RNN(f), RNN(g)) = cmp(f,g)`, `cmp(RNN(f),g) where cmp(f,g)<0 = -1`, `cmp(f,RNN(g)) where cmp(f,g)>0 = 1`.
        #     - (…Though, learning a goal-dependent vortex is as info-heavy as learning all pairwise distances if not more, so it's just as bad as that.)
        #   - Continuous contraction: given src & dst & max-dist, alter src within those confines to get closer to dst. Then, we can decompose all problems at unroll-time: go halfway, then the rest of the way.
        #     - Have a GAN `alter(src, dst, max_dist)→src` & `alter_dist`, where the 'discriminator' is the distance from output to dst. Losses:
        #       - `alter_dist(prev, next, |prev-next| or more) = |prev-next|`
        #       - `alter_dist(prev, goal, max_dist) = min(max_dist, |prev-next| + .99 * alter_dist(next, goal, dist - |prev-next|))`
        #       - (…Wow, this is even worse than all-to-all distance learning.)



        # - 1-step futures:
        #   - We'd like to know where each action takes us, and form a coherent picture, so: for each a→b action, we need to ensure that this action from a leads to the same future as b: `leads_to(future(a), a→b) = sg future(b)` (BYOL) where `act(future(a), future(goal)) = a→b` (which has to learn the *shortest-path* action, otherwise everything is interconnected anyway).
        # - n-step futures:
        #   - Possibly: the `lvl` arg to `future` and `leads_to` and `act` and `up` and `down`: a one-hot embedding of the `n` in `2**n`-len steps.
        #   - `up(future)→metafuture`, `down(metafuture)→future`.
        #     - (Possibly, identity functions.)
        #   - Loss, "each higher level encompasses 2 options of its lower level": `leads_to(up(x)) = sg up(leads_to(leads_to(x))) x:future(a)` (with appropriate-level `act`ions in `leads_to`).
        #     - Possibly, loss, "either 2 or 1 options on higher levels": `leads_to(up(x)) = sg up(leads_to(x))` (with *lower-level* `act`ions in `leads_to`).
        #   - Loss, "autoencoder": `down(up(x)) = up(down(x)) = sg x x:future(a)`.
        #   - Loss, "higher actions copy lower actions": `act(up(a), up(goal)) = sg act(a, goal)`. (If `up` is the identity function, we get this for free.)
        #   - Loss, "shorter (lower) paths take precedence over longer (higher) paths": `act(f, goal) = sg act(down(f), goal) f:leads_to(up(a))` (with higher-level `act`ions).
        #   - A basic loss: `act(prev, next) = action`.
        #   - (…Aren't we pretty much performing ever-less-precise clustering via this hierarchy, so that src & dst will definitely have a level where they do match…)
        #   - TODO: …THINK: will all this structure really *always* converge to low-distance actions?

        # …For contraction, we need to consider a→b→c trajectories, where traj(a, act1, act2)→meta adds consecutive actions to produce a meta-action, and have act1(a, meta) and act2(a, meta); to contract, we need to replace the trajectory with one action such that the distance is summed: 
        #   `traj(lvl,a,act1,act2)→meta`, `act1(lvl,a,meta)→act1`, `act2(lvl,a,meta)→act2`: products.
        #   `meta(lvl, a, goal)→meta`: actions. …Makes `act1` a little redundant tho.
        #   …If we can detect situations where a→b→c leads to the same outcome as a→c and replace a→b→c with a→c… And do this in layers of 2**n-length options (action-sequences)… Isn't this the essence of node contraction? Isn't this why we have the meta-layer?
        #     …What's the loss that 'detects' this?
        #     …Not quite ready to write this down, huh… Do we need to combine this with `future`, and condition action-getting on the `future` in order to actually detect complex trajectories, and make all meta-actions reside in the same space (so that we can replace the actions) by not conditioning `meta` on the level but making it always return the action?…

        # TODO: …What about the semi-classical idea of contraction hierarchies, where we actually search which future-level is the same for src & dst, then go from src to it and from it to dst? What's our justification for not doing it — or if none, then what's deficient about our simpler method?…

        # TODO: Can we write down a simplified loss that will *always* replace 2-step same-dst paths with 1-step paths? Which we can test out, at least on a simple graph? (Which we can scale up to whole hierarchies easily.)
        #   …We have to infer the 0-lvl action from *exactly the same* meta/future (possibly, `meta=up(future)`; one of the meta/future has to be synthetic, not inferred from concrete actions), and make the higher level action predict the lower level action… And maybe `act` accepts not src & goal directly but their futures… How to write this down, exactly?…












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