"""
Pretraining for all possible goals is like learning a map of how to get from anywhere to anywhere. Here, we gradually build up to the full formulation in an easily-comprehensible and easily-visualizable environment.

We have a square board here, and 1 agent that can only walk it. If we can't learn a map here, then how can we expect to learn a map in more complex environments?

NO RL machinery! Lessons must be transferrable all the way.
"""



from model.momentum_copy import MomentumCopy
from model.log import log



import torch
import torch.nn as nn
device = 'cuda' if torch.cuda.is_available() else 'cpu'



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



N, batch_size = 4, 1000
state_sz = 2
overparameterized = 32

unroll_len, denoising_levels = 2*N-2, 1

generator_loss_mult = .0



next = nn.Sequential( # (board, target_board, state, output) → output_state
    # (`output` is for sampling from the action distribution on each step, initially a random vector in each iteration, denoised one or more times.)
    # (The actions are simply sliced from `output_state`.)
    nn.Linear(N*N + N*N + state_sz + state_sz, overparameterized * state_sz),
    nn.ReLU(),
    nn.LayerNorm(overparameterized * state_sz),
    nn.Linear(overparameterized * state_sz, state_sz),
).to(device)
next_discriminator = nn.Sequential( # (board, target_board, state, output) → will_reach_target
    nn.Linear(N*N + N*N + state_sz + state_sz, overparameterized * state_sz),
    SkipConnection(nn.Sequential(
        nn.ReLU(),
        nn.LayerNorm(overparameterized * state_sz),
        nn.Linear(overparameterized * state_sz, overparameterized * state_sz),
    )),
    nn.ReLU(),
    nn.LayerNorm(overparameterized * state_sz),
    nn.Linear(overparameterized * state_sz, 1, bias=False),
).to(device)
next_discriminator_copy = MomentumCopy(next_discriminator, .99)
opt = torch.optim.Adam([*next.parameters(), *next_discriminator.parameters()], lr=1e-3)
for iters in range(50000):
    # TODO: Run & fix.
    #   …GANs are seriously disappointing too: 10k epochs to get to 70% from 30%, SOMETIMES, with LOTS of instability and performance-regression, after a lot of tuning, on a task so toy that it can be solved in 1.
    #   …So what do we do now? Can we re-examine denoising, caring not only about its result but also the intermediate slightly-more-denoised prediction targets?
    #     Seems to perform even worse than GANs, always slightly reducing correctness and nothing else.
    #   …May have to think about direct prediction harder… (I mean, if distribution-modeling is *significantly* worse than averaging, then we can hardly use it in place of averaging, can we? Not sure if it's the state of modern ML that's failing us here, or our impl, or generativeness is just the wrong approach for learning a transitive closure.)



    #   …If we had a way to assign preference of which trajectory should be preserved, then we wouldn't have needed to remember distributions, just predict the avg action… In the discrete-target case, could order by how many actions we took, and always prefer the shortest path. And if in URL, the learned states carry *all* information, then it makes sense to always prefer quickest routes, since whole-distributions likely carry exponentially-much superfluous info. If we had explicit transition probabilities, could have multiplied them to get path-probability to maximize. But what's the continuous analogue of shortest-route?!
    #     …Could "shortest" really mean "most probable" in densely-connected environments, meaning that we could just make our first action predict: the first action of the closest of the two sampled trajectories (action weighed by min minus target-prediction loss)? Can we hallucinate extra trajectories for this, to not have to do extra unrolls, or even 'consider-all beforehand' AKA minimize via learning?
    #       And what about not just far-off-target but closer-targets; how to make "closer targets are preferred" soft?… Would summing losses suffice, or do we need to multiply loss-weighted distances?
    #       (In the actual URL, this would amount to an additional term on the state itself, which selects most-immediately-successful transitions somehow… Unlike how the ∞-limit target is made predictive, changing the actions that led to it to be more eventually-successful — which might not even be the right approach, since it ignores practicality…)
    #         (If we have a `next` (state,goal)→state model, then we shouldn't sample few discrete trajectories at test-time, we should learn to make output (state & action in one) minimize the distances, prioritizing fewer-steps. …This isn't learned and so can't generalize to longer trajectories than we'd sample in-simulation (which is probably like 2)… So should we learn the distance-to-completion then?… One number?…)
    #           (I guess the loss would be `next_loss = prev_loss + prev_loss*(state-target)**2`, possibly with a more normalized loss than L2 or simply clamping L2 to -1…1; not summing up the loss (so, more precision on which parts will be OK) is probably fine. We *might* be able to bootstrap a predictor, though that multiplication might prevent that: `prev_loss=0` prevents all alterations, and `prev_loss>1` has a multiplicative effect on itself if we try bootstrapping the same trajectory many times… Maybe remove the multiplication…)
    #             (WHY DID WE ARRIVE AT MORE-OR-LESS THE INITIAL URL FORMULATION AGAIN, though now `target` really does refer to the state and not its eventual-target…?)
    #       TODO: At least try directly predicting the action of the shortest-path same-target-board trajectory; only the first action I guess. (Should work very well, but be specialized to this board env. If OK, we can move on to continuous-izing it.)
    #         TODO: Make it work? Come on, not being able to know which way to go on a 4×4 board is embarrassing.



    # Sample a batch of trajectories (pre-deciding the target-board), accumulating the denoising loss, and minimizing it wherever we've reached the target.
    L2 = 0
    state = torch.zeros(batch_size, state_sz, device=device)
    # board = env_init(N, batch_size=batch_size) # TODO: Try making it the same?… Doesn't help too much…
    board = env_init(N, batch_size=1).expand(batch_size, N*N)
    target_board = env_init(N, batch_size=batch_size)
    target_ind = N*N*board.argmax(-1) + target_board.argmax(-1) # For special-casing discrete targets (see below).
    states = [] # For special-casing discrete targets (see below).
    achieved_target = torch.full((batch_size,), False, device=device)
    path_len = torch.full((batch_size,), float(unroll_len), device=device)
    target_reachable = []
    for u in range(unroll_len):
        # Do the RNN transition (and an environment step), `unroll_len` times.
        nexts = [torch.randn(batch_size, state_sz, device=device)]
        for lvl in range(denoising_levels):
            # Denoise the next-state, `denoising_levels` times.
            #   (Diffusion models learn to reverse gradual noising of samples, with thousands of denoising steps. That's too slow in an RNN, so we learn the reversing directly.)
            #   (Doesn't seem to work, though. Maybe a GAN would have better luck.)
            nexts.append(next(torch.cat((board, target_board, state, nexts[-1]), -1)))
            input = torch.cat((board, target_board, state, nexts[-1]), -1)
            target_reachable.append(next_discriminator(input.detach()).sum(-1))
            if generator_loss_mult:
                CP = next_discriminator_copy(input)
                print((CP/2+.5).mean().detach().cpu().numpy(), (CP - 1).square().sum().detach().cpu().numpy()) # TODO:
                L2 = L2 + generator_loss_mult * (CP - 1).square().sum()
        first, last = nexts[0], nexts[-1]
        for lvl, noised in enumerate(nexts[1:-1]):
            # (Preserve the batch dimension, so that we could select which to minimize.)
            frac = (lvl+1) / (denoising_levels-2)
            target = (1-frac)*first + frac*last
            L2 = L2 + (noised - target).square().sum(-1)
        state = nexts[-1]
        board = env_step(N, board, state[..., 0:2])
        achieved_target_now = (board == target_board).all(-1)
        prev_achieved_target = achieved_target
        achieved_target = achieved_target | achieved_target_now

        # Special-casing for this discrete-targets-only environment, to see how well directly predicting the shortest path works.
        #   (Ideally, should have a smooth loss on states/actions, able to handle continuous targets gracefully.)
        path_len = torch.where(~prev_achieved_target & achieved_target_now, torch.full_like(path_len, u), path_len)
        states.append(state)
    matching_path_len = torch.where(
        (target_ind.unsqueeze(-1) == target_ind) & achieved_target,
        path_len,
        torch.full((batch_size, batch_size), float(unroll_len+1), device=device) - torch.eye(batch_size, device=device),
    )
    nearest_target = matching_path_len.argmin(-1)
    for st in states: # Yes, finally we directly predict the shortest same-target actions.
        tt = torch.index_select(st, 0, nearest_target)
        L2 = L2 + (st - tt).square().sum()
    # TODO: Figure out why shortest-plan prediction doesn't work. Because it really should, at least a little!
    #   …Well, now it's SLIGHTLY better, but why is its performance still so poor? Even GANs (30% on 4×4, unstable-ish) are beating it (25% on 4×4, very consistent)… This can't be right.
    #   TODO: Try printing the actions, as in, how frequent each of the 4 actions is, both any-taken and only-when-target-was-reached?
    #   TODO: Try printing the shortest trajectories (in `board` indices, or even xy coords — along with the target), which shouldn't change significantly over time, and be sensible to boot?

    achieved_target = achieved_target.float()
    # L2 = L2 * achieved_target # (This is only for denoising.)
    L2 = L2.sum()
    L22 = 0
    for reachable in target_reachable:
        # print('                                                                 discriminator guessed correctly', str((((reachable > 0).float() == achieved_target).float().mean()*100).round().detach().cpu().numpy())+'%', '>0', str(((reachable > 0).float().mean()*100).round().detach().cpu().numpy())+'%') # TODO:
        L22 = L22 + (reachable - (achieved_target*2-1)).square().sum()
    (L2 + L22).backward()
    opt.step();  opt.zero_grad(True);  next_discriminator_copy.update()
    with torch.no_grad():
        correct_frac = achieved_target.mean()
        log(0, False, generator_L2 = L2.detach().cpu().numpy())
        log(1, False, discriminator_L2 = L22.detach().cpu().numpy())
        log(2, False, correct_target_perc = (correct_frac*100).round().cpu().numpy())
        print(str(iters).rjust(6))
# TODO: Okay, what do we want to learn, building up to URL gradually?
#   - ✓ From board and action (randomly-generated) to board — EASY
#   - ✓ From board and neighboring-board (gotten via env_step with a random action) to action (the randomly-generated one) — will be so easy that it's pointless to implement.
#   - From board & target-board & extra-state & whole-output (or a random vector initially), to the next action & extra state: `next`.
#     - (Need to limit the unroll-length, or else practically everything will count as reachable.)
#     - Average-plan makes no sense because everything is connected to everything, so we need to learn the *distribution* of plans that will lead us to the target, so either:
#       - ⋯ DDPM-like but speedy (not about to do thousands of steps per RNN step): make `next` self-denoising (accept its output as an input, initially a random vector), and wherever we have a loss (here, just: make less-denoised outputs predict more-denoised outputs, only in trajectories that reached the target), make predict-branches have less denoisings than stopgrad-branches to make denoising learned. Possibly, have completely separate RNN-states for different denoising levels. (Sounds quite trainable, but might just collapse diversity like in the initial experiments; maybe using CCL for prediction could help.)
#       - ⋯ GAN-like: train a discriminator (from board & target-board & extra-state & 'whole-output'-randomness) of whether a trajectory will succeed (known after a whole rollout), and maximize the predicted success-probability by all `next`-steps (but not by the discriminator).
#       - ⋯ …Rethink whether we really couldn't enforce a sensible priority on which plans from a distribution we pick, because of impl troubles…
#     - (Until we can reliably learn the next action, with a batch-size of 1, without non-transferrable tasks.)
#   - Almost-URL: learn the distribution of targets, along with distributions of plans to reach them (learning eventual-RNN-states would have been full URL).
#   - Full URL, where goal-of-state is learned too: goal:ev(state);  goal=ev(next(state, goal))