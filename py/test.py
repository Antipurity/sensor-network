"""
# The model that we implement in this file

Let's assume a discrete model of the world: `universe: state → state`. This describes one boring line/circle through time.

Now, let's consider an agent with bidirectional communication with its world, receiving `obs`ervations and returning `act`ions: `universe: (state, act) → (state, obs)`. Now we have a whole infinite graph of possibilities, with the path selected by `act`ions.

Where that agent ultimately goes can be named its goal. In the sequence of the agent's `obs`ervations, this can be written as `SRC <path> DST <path>: MID0 MID1 MID2 … DST`. This is very [similar to ](https://paperswithcode.com/task/text-infilling)[text ](https://nlp.stanford.edu/pubs/donahue2020infilling.pdf)[infilling](https://beta.openai.com/docs/guides/completion/inserting-text), where `DST` is either in `state`-space or in `obs`-space but variable-length.) (RL writes goals as one-number-utility functions. We follow [upside-down RL](https://arxiv.org/abs/1912.02877) here, but both approaches are general enough to implement each other.

To reach goals most efficiently, the agent should build a map of that graph of possibilities, to run shortest-path queries on. Why *not* use something like a [Transformer](https://arxiv.org/abs/1706.03762) with [RNN-state](https://arxiv.org/abs/2202.05780), and do next-`obs` prediction? Without an ideological agenda, there's no reason to claim that Deep Learning can't succeed at general intelligence, being the most data-driven and successful approach that we have.

`act`ing can actually be exactly the same as next-frame prediction, if actions are included in next-frame observations; then it's easy for human and AI actions to be treated exactly the same. But we do want shortest/best paths and not just any paths, so that prediction has to be [critic-](https://arxiv.org/abs/2006.15134)[regularized](https://arxiv.org/abs/1806.05635); if `obs` are stochastic, the model may become overly optimistic instead of averaging/smudging the predictions.

# Relation to self-supervised learning

In recent years, deep learning has seen the rise of methods for self-supervised learning. Among them, a particular kind of non-contrastive methods for extracting useful representations from images bear strong resemblance to URL.

- [Barlow twins](https://arxiv.org/abs/2103.03230): `repr(state1) = repr(state2)` using a cross-correlation loss for soft-whitening.

- [SimSiam](https://arxiv.org/abs/2011.10566): basically `predictor(repr(state1)) = sg(repr(state2))`. Here, `state1` and `state2` are augmented image views put through a neural network.
    - [Bootstrap Your Own Latent](https://arxiv.org/abs/2006.07733): same, but the predicted `repr` on the right is replaced with a slow-moving average of the trained `repr` on the left.

(Image augmentations include: crop, blur, make it grayscale, change brightness and/or contrast and/or saturation. Models get good by learning to ignore those in their learned representations.)

This is similar to just predicting the next input in RNNs, possibly min-distance-filtered. The differences from, say, BYOL: no `repr` and thus no momentum-slowing; asymmetry; and `predictor` is conditioned on both past-RNN-state and action.

Further, the ability to reproduce [the human ability to learn useful representations from interacting with the world](https://xcorr.net/2021/12/31/2021-in-review-unsupervised-brain-models/) can be said to be the main goal of self-supervised learning in computer vision. The structure of the body/environment is usable as data augmentations: for images, we have eyes, which can crop (movement & eyelids), make it grayscale [(eye ](http://hyperphysics.phy-astr.gsu.edu/hbase/vision/rodcone.html)[ro](https://en.wikipedia.org/wiki/Rod_cell)[ds)](https://en.wikipedia.org/wiki/File:Distribution_of_Cones_and_Rods_on_Human_Retina.png), scale and flip and rotate (body movement in 3D), blur (un/focus), adjust brightness (eyelashes), and do many indescribable things, such as "next word" or "next sound sample after this movement".
"""
# (TODO: Mention that we require PyTorch 1.10+ because we use forward-mode AD.)
# (TODO: Document how to use command-line args to import envs, and `module.Env()(sensornet)` with callbacks and `'goal'` at the end for constraint-specification and `.metric()` for logging a dict.)



















# TODO: Run & fix the copy-task in `env/copy.py`, to test our implementation.

# TODO: …Might want to do the simplest meta-RL env like in https://openreview.net/pdf?id=TuK6agbdt27 to make goal-generation much easier and make goal-reachability tracked — with a set of pre-generated graphs to test generalization…

# TODO: Make `graphenv` work not globally but in a class.

# TODO: Should `sn.handle` also accept the feedback-error, which we can set to `1` to communicate bit-feedback?
#   TODO: …For computational efficiency, maybe make `sn` accept the optional feedback-size in addition to cell-shape, so that here we can generate like 8 or 16 bits per cell instead of doing 8 NN calls per step…



# TODO: Make goals the full-RNN-states (latents), not the inputs, like in http://proceedings.mlr.press/v100/nair20a/nair20a.pdf but with discrete-sampling instead of CC-VAEs:
#   TODO: Alloc SRC `'-goal-src-state-'` and DST `'-goal-dst-state-'` for personal use. (Maybe make `modify_name` assert that these are unused.) (The current `'goal'` could probably act as SRC.)
#   TODO: On unroll, in a throwaway episode, rename the frame to DST and sample and put the result into the replay buffer.
#   TODO: On unroll, to pick a goal, fetch some random subset of cells from the replay buffer, rename to SRC and 0-fill their values, and in a throwaway episode, sample (possibly for all goal-groups at once) and set as the goal.
#   TODO: On replay, select random subsets (to support partial goals) of faraway latents as destinations, not of inputs.
#     TODO: Have a 50/50 chance to select either faraway-latents or inputs as dst, so that we don't lose the ability for envs to specify goals.
#       (With this, the interface will unify not only obs-and-act and obs-and-goal, but also allow training to reach other agents' full-RNN-state goals, by simply exposing their inner states as observations. "The individual is obsolete", mm.)
#   TODO: On replay, in a throwaway episode, self-imitate (if regret is positive, max the sample-probability of) faraway-dst's SRC-cells given DST-renamed and 0-filled versions of them.
#     TODO: Also, there, self-imitate random subsets of SRC-renamed frames given their cell-name-only versions.














import asyncio
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd.forward_ad as fw
torch.set_default_tensor_type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)

from model.recurrency import State, SRWM

import sensornet as sn

from model.log import log, clear



# Hyperparameters.
cell_shape = (8,8,8,8, 64)
sn.shape(*cell_shape)

state_sz, goal_sz = 256, 256
slow_mode = .05 # Artificial delay per step.

dist_levels = 2
bits_per_chunk = 8 # How to `sample`.

replays_per_step = 2
max_replay_buffer_len = 1024

save_load = ''
steps_per_save = 1000



# Environments, from the command line.
import sys
import importlib
def prepare_env(path):
    mod = importlib.import_module('env.' + path)
    assert hasattr(mod, 'Env')
    sensor = mod.Env()
    def sensor_with_name(*a, **kw):
        try:
            modify_name.ctx = path
            return sensor(*a, **kw)
        finally:
            modify_name.ctx = None
    sn.sensors.append(sensor_with_name)
    return mod
def modify_name(name):
    assert modify_name.ctx is not None, "Sending data not in an env's sensor; don't send it in callbacks of queries, instead remember to send it on the next step"
    res = [name[i] if i < len(name) else None for i in range(len(cell_shape) - 1)]
    res[-1] = modify_name.ctx
    if name[-1] == 'goal':
        res[len(name) - 1] = None
        res[-2] = 'goal'
    return res
sn.modify_name.append(modify_name)
envs = [prepare_env('graphenv')] if len(sys.argv) < 2 else [prepare_env(e) for e in sys.argv[1:]]



replay_buffer = []



# Our interface to multigroup partial goals (AND/OR goals): the last 2 name parts are ('goal', group_id).
goal_filter = sn.Filter([*[None for _ in cell_shape[:-3]], 'goal', ...])
goal_name = torch.tensor(goal_filter.template(cell_shape))
goal_name = torch.cat((goal_name, torch.full((cell_shape[-1],), float('nan'))), -1)



def goal_group_ids(frame):
    """Returns the `set` of all goal-group IDs contained in the `frame` (a 2D NumPy array), each a byte-objects, suitable for indexing a dictionary with."""
    return set(name.tobytes() for name in np.unique(frame[:, sum(cell_shape[:-2]) : sum(cell_shape[:-1])], axis=0))



class Sampler:
    """
    GANs, VAEs, contrastive learning, BYOL: direct prediction of inputs averages them, so we need tricks to learn actual input distributions    
    Autoregressive discrete sampling: AWAKEN

    Uses a `cells × action_size → cells × 2**bits_per_chunk` RNN (with hidden `State`) to sample from bit sequences, without averaging in-action correlations away, as compute-efficiently as possible (vary `bits_per_chunk` depending on RNN's size).

    Same idea as in language modeling, and [Gato](https://arxiv.org/abs/2205.06175), though per-cell and with explicit bits.

    (This adds autoregressive in-cell sampling. Cells are still processed in parallel, because envs should make cells independent of each other by design. And cell-groups are already autoregressively processed by RNNs.)
    """
    def __init__(self, fn, bits_per_chunk, start, zero=0, one=1):
        bits = bits_per_chunk
        self.fn, self.bits_per_chunk, self.start, self.zero, self.one = fn, bits, start, zero, one
        # Create a `(bits,)`-shaped tensor, for converting bit-patterns to indices (via mult-and-sum).
        self._powers = 2 ** torch.linspace(bits-1, 0, bits, dtype=torch.int32)
        # Create a `(2**bits, bits)`-shaped tensor, for converting indices to their bit-patterns.
        x = zero*torch.ones(*[2 for _ in range(bits)], bits)
        all = slice(None)
        ind = [all] * (bits+1)
        for b in range(bits): # x[:,:,:,1,...,b] = 1
            ind[b], ind[-1] = 1, b
            x[ind] = one
            ind[b] = all
        self._bits = x.reshape(2**bits, bits)
    def __call__(self, query):
        """Given a `query` that's zero-padded to be action-sized, samples a binary action.

        Non-differentiable; to make an action more likely to be sampled, use `.loss`.

        This should advance RNN `State`, so do `State.Episode` shenanigans if unwanted.

        (No non-action-cell sampling AKA L2-prediction here, since we won't use that. Though in theory, could use the outcome-averaged approximation, and complicate the interface by having a per-cell bitmask of whether it's an action.)"""
        with torch.no_grad():
            print('query', query) # TODO: It's a tensor…
            print('query unpack_dual', fw.unpack_dual(query)) # TODO: It's a tensor…
            query = detach(query).clone() # TODO: Why can't we detach the `query`?
            #   No forward-derivative either, and prepare to write to `query` in-place.
            i = self.start
            while i < query.shape[-1]:
                j = min(query.shape[-1], i + bits_per_chunk)
                indices = F.softmax(self.fn(query), -1).multinomial(1)[..., 0]
                bits = self._bits[indices]
                query[:, i:j] = bits[:, 0:j-i]
                i += j
            return query
    def loss(self, act):
        """
        Returns a loss that, when minimized, perfectly copies the probability of sampling seen `act`ions.

        The result is `(cells, 1)`-shaped; `.sum()` it before `.backward()`.

        Non-action observations are also supported (detected & predicted) via L2 loss, though they cannot be sampled, and are outcome-averaged.

        (Does the same RNN calls with the same RNN inputs as sampling `act` would, so computation graphs match exactly.)
        """
        i = self.start
        query = torch.cat((act[:, :i].detach(), torch.zeros(act.shape[0], act.shape[1] - i)), -1)
        is_act = self._act_mask(act[:, i:])
        loss = 0.
        eps = 1e-5
        while i < act.shape[-1]:
            j = min(query.shape[-1], i + bits_per_chunk)
            logits = self.fn(query)
            obs_target = act[:, self.start : self.start+logits.shape[-1]]
            probs = F.softmax(logits, -1)
            chunk = act[:, i:j]
            bits = ((chunk - self.one).abs() < eps).int()
            indices = (bits * self._powers[-bits.shape[-1]:]).sum(-1)
            act_target = F.one_hot(indices, probs.shape[-1]).float()
            #   (Yes, L2 loss with a one-hot index does end up copying the probabilities.)
            loss = torch.where(is_act, probs - act_target, logits - obs_target).square().sum(-1, keepdim=True)
            query[i:j] = act[i:j]
            i += j
        return loss
    def _act_mask(self, act, eps=1e-5):
        return ((act - self.zero).abs().min((act - self.one).abs()) < eps).all(-1, keepdim=True)
sample = Sampler(lambda x: transition_(x)[0], bits_per_chunk=bits_per_chunk, start=sum(cell_shape)-cell_shape[-1], zero=-1, one=1)



def cat(*a, dim=-1): return torch.cat(a, dim)
class SkipConnection(nn.Module):
    """Linearize gradients, to make learning easier."""
    def __init__(self, *fn): super().__init__();  self.fn = nn.Sequential(*fn)
    def forward(self, x):
        y = self.fn(x)
        return y if x.shape[-1]<y.shape[-1] else x + y if x.shape == y.shape else x[..., :y.shape[-1]] + y
def h(ins = state_sz, outs = ...): # A cross-cell transform.
    if outs is ...: outs = ins
    return SkipConnection(
        nn.ReLU(), nn.LayerNorm(ins), SRWM(ins, ins, heads=2),
        nn.ReLU(), nn.LayerNorm(ins), nn.Linear(ins, outs),
    )



try:
    if not save_load: raise FileNotFoundError()
    transition = torch.load(save_load)
except FileNotFoundError:
    transition = nn.Sequential(
        # Input prev-actions and next-observations, output info that `sample` can use and distances.
        #   We predict the next input, thus serving as both an RL agent and a self-supervised learner.
        #   We overcome L2-prediction outcome-averaging via autoregressive `sample`ing, though it's only fully 'solved' for binary outputs (AKA actions).
        h(sum(cell_shape), state_sz),
        h(state_sz, state_sz),
        h(state_sz, 2 ** bits_per_chunk + dist_levels + 1),
    )
def transition_(x):
    """Wraps `transition` to return a tuple of `(sample_info, distance, distance_regret)`."""
    y = transition(x)
    return y[..., :-(dist_levels+1)], 2 ** y[..., -(dist_levels+1):-1], 2 ** y[..., -1:]
optim = torch.optim.Adam(transition.parameters(), lr=1e-3)



def DODGE(loss_fn, model, direction_fn = lambda sz: torch.randn(sz)):
    """
    Implements [directional gradient descent](https://openreview.net/forum?id=5i7lJLuhTm): pick a direction [(a random unit vector by default)](https://arxiv.org/abs/2202.08587), compute a forward-derivative for it, receive a scalar feedback to correct it, and assign the gradients.

    Needs `loss_fn(…)→loss` and `model` for parameters. Do an optimizer step on those parameters yourself.
    """
    responsibility = []
    sz = 0
    for mod in model.modules():
        responsibility.append((mod, []))
        for name, param in mod.named_parameters(recurse=False):
            responsibility[-1][1].append((name, param))
            sz += param.numel()
    def loss_wrapper(*args, **kwargs):
        direction = direction_fn(sz)

        direction = (direction - direction.mean()) / (direction.std() + 1e-8)
        n = 0
        # Give the `direction` to `model`'s parameters.
        for mod, ps in responsibility:
            for name, param in ps:
                assert getattr(mod, name) is param
                tangent = direction[n : n+param.numel()].reshape(*param.shape)
                delattr(mod, name)
                setattr(mod, name, fw.make_dual(param, tangent))
                n += param.numel()
        loss = loss_fn(*args, **kwargs)
        loss_tangent = fw.unpack_dual(loss).tangent
        # Approximate the gradient by multiplying forward-gradient by the `loss_tangent` number.
        for mod, ps in responsibility:
            for name, param in ps:
                d = fw.unpack_dual(getattr(mod, name))
                grad = d.tangent * loss_tangent
                param.grad = grad if param.grad is None else param.grad + grad
                delattr(mod, name)
                setattr(mod, name, param)
    return loss_wrapper



def detach(x):
    return fw.unpack_dual(x).primal.detach()
def loss(prev_ep, frame, dst, timediff, regret_cpu):
    """`loss(prev_ep, frame, dst, timediff)`

    Predicts the `prev_frame`→`frame` transition IF it's closer to the goal than what we can sample.

    - `prev_ep` is the context that knows `prev_frame`.
    - `frame` is the immediately-next prev-action and next-observation cells.
    - `dst` is some faraway goal.
    - `timediff` is the distance to that goal, as a `(cells, 1)`-shaped int32 tensor."""

    with prev_ep:
        # Multigroup ("AND") goals, where many threads of experience (with their group IDs) can pursue separate goals.
        #   (The group ID replaces the last name-part.)
        is_learned = torch.rand(frame.shape[0], 1) < (.1+.9*random.random())
        dst_group_id = torch.rand(1, cell_shape[-2])*2-1
        src_group_id = torch.where(is_learned, dst_group_id, torch.rand(frame.shape[0], cell_shape[-2]*2-1))
        frame = torch.cat((frame[:, :sum(cell_shape[:-2])], src_group_id, frame[:, sum(cell_shape[-1:]):]), -1)
        dst_group_id = dst_group_id.expand(dst.shape[0], cell_shape[-2])
        dst = torch.cat((dst[:, :sum(cell_shape[:-2])], dst_group_id, frame[:, sum(cell_shape[-1:]):]), -1)

        # Name `dst` with the fact that it's all goal-cells, and add it to the `frame`.
        dst = torch.where(goal_name == goal_name, goal_name, dst)
        frame = torch.cat((dst, frame), 0)
        is_learned = torch.cat((torch.full((dst.shape[0], 1), True), is_learned), 0).float()

        # What to contrast `frame` with.
        with State.Episode(start_from_initial=False):
            vals = cell_shape[-1]
            frame_query = torch.cat((frame[:, :-vals], torch.zeros(frame.shape[0], vals)), -1)
            frame_pred = sample(frame_query)

        # Distances of `frame` and `frame_pred` to contrast.
        with State.Episode(start_from_initial=False):
            _, frame_dist, _ = transition_(frame)[1]
        with State.Episode(start_from_initial=False):
            _, frame_pred_dist, frame_pred_regret = transition_(frame_pred)[1]

        # Critic-regularized regression: if `frame_pred` regrets not being `frame`, make it into `frame`.
        dist_is_better = frame_dist[:, -1:] < frame_pred_dist[:, -1:]
        mask = (dist_is_better | ~sample._act_mask(frame)).float()
        predict_loss = (is_learned * sample.loss(frame) * mask).sum()

        # Remember our regret of not being the `frame`. *Only* remember what we regret.
        regret = (is_learned * (frame_pred_dist[:, -1:] - frame_dist[:, -1:])).sum() / (is_learned.sum() + 1e-2)
        regret_cpu.copy_(regret, non_blocking=True)

        # Learn the regret, since estimating prediction-error by training several copies of the model is too expensive.
        regret_loss = (is_learned * (frame_pred_regret.log() - detach(regret).clamp(0.).log()).square()).sum()

        # Critic regression: `dist = sg timediff`
        #   We try to learn *min* dist, not just mean dist, by making each next dist-level predict `min(timediff, prev_level)`.
        #     (Worst-case, can also try tilted L1 loss.)
        dist_limit = torch.cat((timediff, frame_dist[:, :-1]), -1)
        dist_loss = (is_learned * (frame_dist.log2() - detach(timediff.min(dist_limit)).log2()).square()).sum()

        # GAN-like penalization of ungrounded plans.
        dist_penalty = 1.05
        ungrounded_dist_loss = (is_learned * (frame_pred_dist - detach(frame_pred_dist) * dist_penalty).square()).sum()

        log(0, False, torch, improvement = mask.mean() - (~sample._act_mask(frame)).float().mean())
        log(1, False, torch, predict_loss=predict_loss, regret_loss=regret_loss, dist_loss=dist_loss, ungrounded_dist_loss=ungrounded_dist_loss)
        n = 2
        for env in envs:
            if hasattr(env, 'metric'):
                log(n, False, torch, **env.metric())
                n += 1

        loss = predict_loss + regret_loss + dist_loss + ungrounded_dist_loss
        return loss
loss_fn = DODGE(loss, transition)

def replay(optim, current_frame, current_time):
    """Remembers a frame from a distant past, so that the NN can reinforce actions and observations when it needs to go to the present."""
    if len(replay_buffer) < 8: return

    for _ in range(replays_per_step):

        time, ep, frame, regret = random.choice(replay_buffer)

        # Learn partial ("OR") goals, by omitting some cells.
        #   (Note: it's possible to sample several goals and pick a 'best' one, [like AdaGoal does](https://arxiv.org/abs/2111.12045). But that works best with a measure of 'error' to maximize, and our replay-buffer already kinda maximizes regret.)
        #     (Could be done with either the `regret` output of `transition_`, or by keeping several `transition`s to measure ensemble disagreement. Could also [train some generative model](http://proceedings.mlr.press/v100/nair20a/nair20a.pdf), but those are hard to train, hence the sampling.)
        #       (All of which is too expensive.)
        dst_is_picked = np.random.rand(current_frame.shape[0]) < (.05+.95*random.random())
        dst = current_frame[dst_is_picked]

        # Learn.
        loss_fn(ep, frame, dst=dst, timediff = torch.full((dst.shape[0], 1), float(current_time - time)), regret=regret)

        # If our replay buffer gets too big, leave only max-regret samples.
        if len(replay_buffer) > max_replay_buffer_len:
            replay_buffer.sort(key = lambda sample: sample[3], reverse=True)
            del replay_buffer[-(max_replay_buffer_len // 2):]

    # Optimize NN params.
    optim.step();  optim.zero_grad(True)



@sn.run
async def main():
    with State.Setter(lambda state, to: state.initial*.001 + .999*to): # Soft-reset.
        with State.Episode() as life:
            with torch.no_grad():
                prev_q, action, frame = None, None, None
                goals = {} # goal_group_id → (gpu_goal_cells, cpu_expiration_time)
                time = 0 # Hopefully this doesn't get above 2**31-1.
                while True:
                    if slow_mode > 0:
                        await asyncio.sleep(slow_mode)

                    obs, query, data_error, query_error = await sn.handle(sn.torch(torch, action))
                    #   (If getting out-of-memory, might want to chunk data/query processing.)

                    # (The replay buffer won't want to know any user-specified goals.)
                    #   (And fetch goal-group IDs to add constraints for exploration, even if the env has set some goals.)
                    frame_names = np.concatenate((prev_q, obs[:, :prev_q.shape[1]]), 0) if prev_q is not None else obs[:, :query.shape[1]]
                    goal_cells = goal_filter(frame_names, cell_shape=cell_shape)
                    not_goal_cells = goal_filter(frame_names, cell_shape=cell_shape, invert=True)
                    prev_q = query
                    groups = goal_group_ids(frame_names)

                    # Zero-pad `query` to be action-sized.
                    obs, query = torch.tensor(obs), torch.tensor(query)
                    query = torch.cat((query, torch.zeros(query.shape[0], obs.shape[1] - query.shape[1])), -1)
                    frame = torch.cat((action, obs), 0) if action is not None else obs

                    # Append prev-RNN-state and next-frame to the replay-buffer.
                    replay_buffer.append((
                        time,
                        life.clone(remember_on_exit=False),
                        frame[not_goal_cells],
                        torch.tensor(1000., device='cpu'),
                    ))

                    # Delete/update our `goals` when we think we reached them.
                    for group, (cells, expiration) in goals.copy().items():
                        if time > expiration:
                            del goals[group]
                    for group in groups:
                        if group not in goals:
                            goal = random.choice(replay_buffer)[2]
                            goal = goal[np.random.rand(goal.shape[0]) < (.05+.95*random.random())]
                            goal = torch.where(goal_name == goal_name, goal_name, goal)
                            expiration_cpu = torch.tensor(time+10000, device='cpu').int()
                            expiration_gpu = torch.tensor(time+10000).int()
                            goals[group] = (goal, expiration_cpu, expiration_gpu)

                    # Give goals to the RNN.
                    extra_cells = []
                    for group, (cells, expiration_cpu, expiration_gpu) in goals.items():
                        extra_cells.append(cells)
                    if len(extra_cells): frame = torch.cat([*extra_cells, frame], 0)

                    # Give prev-action & next-observation, remember distance estimates, and sample next action.
                    _, dist, regret = transition_(frame)
                    n = 0
                    for group, (cells, expiration_cpu, expiration_gpu) in goals.items():
                        # (If we do better than expected, we leave early.)
                        if cells.shape[0] > 0:
                            group_dist = (time + dist.detach()[n : n+cells.shape[0], -1].mean() + 1).int().min(expiration_gpu)
                            expiration_cpu.copy_(group_dist, non_blocking=True)
                            expiration_gpu.copy_(group_dist)
                        else:
                            expiration_cpu.fill_(0)
                        n += cells.shape[0]
                    with State.Episode(start_from_initial=False):
                        action = sample(query)
                        #   (Can also do safe exploration / simple planning, by `sample`ing several actions and only using the lowest-dist-sum (inside `with State.Episode(False): ...`) of those.)
                        #     (Could even plot the regret of sampling-one vs sampling-many, and see if/when it's worth it.)

                    # Learn.
                    replay(optim, frame, time)

                    if save_load and time % steps_per_save == 0:
                        torch.save(transition, save_load)

                    time += 1