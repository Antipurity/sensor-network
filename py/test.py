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




#   TODO: …Should we make `DST` `state`-space in our impl too, and to bridge from obs-space to state-space, have a NN from obs to RNN-state, possibly just a single episode of the main NN with just the desired-obs inputted? TODO: But how would this model partial goals, AKA "we don't care about any of the other cells"? By training dists with partial goals?…
#     TODO: Since RNN state is unwieldy to actually condition on, should we learn a separate cell type just for goals, outputted by the main RNN per-cell? How is it learned, exactly? …We either need to store the unroll's goal-cells in replay (and maybe update this on replay), or do a no-DST unroll first then a with-DST unroll… Which approach is better?

















# TODO: Maybe, mine for regret harder: replay-buffer prioritization of max-regret (of `b`: mean/max `dist(a,b) - (j-i)` among `a`) samples for fastest spreading of the influence of discovered largest shortcuts. At unroll, always overwrite the last sample; at replay, sample that plus randoms, and sort by minibatch-regret and write-back that sorted data. (Also good for easily sampling good unroll-time goals, getting more data in most-promising AKA max-regret areas.)
#   Wouldn't this make the replay buffer not contiguous anymore, though? …Maybe it's fine.
#   …How would this prioritization be implemented? Do we send the regret to CPU, and when it arrives, sort a few? Or can/should we update RNN states & frames on-GPU?

# TODO: For goals, use a PyTorch-backend `sn.Namer`, which puts `'goal'` in the last spot.
#   TODO: …Should we make the -2th slot the group id, so that we can specify per-group goals?… It's the best way to support both AND and OR for goals, isn't it…
#     TODO: How do we implement these per-cell extractions of grouped goals?
#   TODO: When saving unroll-time frames, filter out the cells that match the goal-name.
#   TODO: At unroll-time, generate observation-cell/s and estimate time-to-reach-it; at every step, append named-as-goal cells to obs (*unless there are any goal-cells in observations*); and when the prev estimated time runs out, pick new goal-cells and update the estimated time.
#     TODO: At unroll-time, save to the replay buffer.
#     TODO: At unroll-time, give goals at every step.
#   TODO: At replay-time, TODO: what do we do, exactly?… How to learn per-cell goals?…
#     TODO: At replay-time, inside an episode: should pick a starting point (from which the episode-copy should be taken), sample a faraway goal from the future (preferably) and put it at the start, and unroll several from the starting point steps, applying the loss.

# (…We could also make the distance-network learn not only distance but its own prediction-regret (or maybe regret-per-step), so that goal-generation can maximize (a learned measure of) regret, at least by considering a few goals…)
#   Is this a good idea?
#   TODO: Do we maybe want to learn multiple metrics-about learning, such as misprediction? Which, exactly?

# (…Might want to do the simplest meta-RL env like in https://openreview.net/pdf?id=TuK6agbdt27 to make goal-generation much easier and make goal-reachability tracked — with a set of pre-generated graphs to test generalization…)
#   TODO: Maybe move `minienv` to `env/`, so that we can safely implement as many environments as we want?…

# TODO: …Also save/load the model…

# TODO: …May also want to implement importing the modules that the command line has requested, for easy env-switching…
#   TODO: …Maybe implement & use a copy-task, initially, to test our implementation…

# TODO: Should `sn.handle` also accept the feedback-error, which we can set to `1` to communicate bit-feedback?
#   TODO: …For computational efficiency, maybe make `sn` accept the optional feedback-size in addition to cell-shape, so that here we can generate like 8 or 16 bits per cell instead of doing 8 NN calls per step…
#     Maybe even fully turn `sn` over to discrete actions, and only ever have 1 NN call per step, demanding that envs adapt instead…














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



cell_shape = (8,8,8,8, 64)
sn.shape(*cell_shape)

state_sz, goal_sz = 256, 256

dist_levels = 2
bits_per_chunk = 8 # How to `sample`.



class ReplayBuffer:
    """Stores the in-order sequence of most-recent events. Needs `max_len`. Supports `len(rb)`, `rb.append(data)`, `rb[index]`."""
    __slots__ = ('buffer', 'head', 'max_len')
    def __init__(self, max_len):
        self.buffer, self.head, self.max_len = [], 0, max_len
    def __len__(self):
        """How many data-samples are currently contained. Use this to decide the indices to sample at."""
        return len(self.buffer)
    def append(self, data):
        """Appends a data-sample to the replay buffer."""
        if len(self.buffer) == self.max_len:
            self.buffer[self.head] = data
        else:
            self.buffer.append(data)
        self.head = (self.head + 1) % self.max_len
    def __getitem__(self, index):
        """Returns the sample at the given index. For example, use `random.randrange(len(replay_buffer))`.

        Consecutive indices are guaranteed to be a part of a contiguous sequence."""
        assert isinstance(index, int)
        return self.buffer[(self.head + index) % len(self)]
    def __iter__(self):
        for i in range(len(self)): yield self[i]
replay_buffer = ReplayBuffer(1024)



import minienv
minienv.reset(can_reset_the_world = False, allow_suicide = False, max_nodes=1000)
#   TODO: Should at least make `minienv` work not globally but in a class.



class SkipConnection(nn.Module):
    """Linearize gradients, to make learning easier."""
    def __init__(self, *fn): super().__init__();  self.fn = nn.Sequential(*fn)
    def forward(self, x):
        y = self.fn(x)
        return y if x.shape[-1]<y.shape[-1] else x + y if x.shape == y.shape else x[..., :y.shape[-1]] + y
def cat(*a, dim=-1): return torch.cat(a, dim)
def h(ins = state_sz, outs = ...): # A cross-cell transform.
    if outs is ...: outs = ins
    return SkipConnection(
        nn.ReLU(), nn.LayerNorm(ins), SRWM(ins, ins, heads=2),
        nn.ReLU(), nn.LayerNorm(ins), nn.Linear(ins, outs),
    )



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
            query = detach(query).clone()
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



transition = nn.Sequential(
    # Input prev-actions and next-observations, output info that `sample` can use and distances.
    #   We predict the next input, thus serving as both an RL agent and a self-supervised learner.
    #   We overcome L2-prediction outcome-averaging via autoregressive `sample`ing, though it's only fully 'solved' for binary outputs (AKA actions).
    h(sum(cell_shape), state_sz),
    h(state_sz, state_sz),
    h(state_sz, 2 ** bits_per_chunk + dist_levels),
)
def transition_(x):
    """Wraps `transition` to return a tuple of `(sample_info, distance)`."""
    y = transition(x)
    return y[..., :-dist_levels], y[..., -dist_levels:]
optim = torch.optim.Adam(transition.parameters(), lr=1e-3)



def DODGE(loss_fn, model, direction_fn = lambda sz: torch.randn(sz)):
    """
    Implements [directional gradient descent](https://openreview.net/forum?id=5i7lJLuhTm): pick a direction [(a random unit vector by default)](https://arxiv.org/abs/2202.08587), compute a forward-derivative for it, receive a scalar feedback to correct it, and assign the gradients.

    Needs `loss_fn(…)→loss` and `model` for parameters. Do an optimizer step on those parameters yourself.
    """
    responsibility = []
    sz = 0
    for mod in model.modules():
        responsibility.append(mod, [])
        for name, param in mod.named_parameters(recursive=False):
            responsibility[-1].append((name, param))
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
def loss(prev_ep, frame, dst, timediff):
    """`loss(prev_ep, frame, dst, timediff)`

    Predicts the `prev_frame`→`frame` transition IF it's closer to the goal than what we can sample.

    - `prev_ep` is the context that knows `prev_frame`.
    - `frame` is the immediately-next prev-action and next-observation cells.
    - `dst` is some faraway goal.
    - `timediff` is the distance to that goal, as a `(cells, 1)`-shaped int32 tensor."""

    with prev_ep:
        loss = 0.

        # TODO: Name `dst` with goal-related name parts.
        frame = torch.cat((dst, frame), 0)

        # What to contrast `frame` with.
        with State.Episode(start_from_initial=False):
            vals = cell_shape[-1]
            frame_query = torch.cat((frame[:, :-vals], torch.zeros(frame.shape[0], vals)), -1)
            frame_pred = sample(frame_query)

        # Distances of `frame` and `frame_pred` to contrast.
        with State.Episode(start_from_initial=False):
            frame_dist = transition_(frame)[1]
        with State.Episode(start_from_initial=False):
            frame_pred_dist = transition_(frame_pred)[1]

        # Critic-regularized regression: if `frame_pred` regrets not being `frame`, make it into `frame`.
        dist_is_better = frame_dist[:, -1:] < frame_pred_dist[:, -1:]
        mask = (dist_is_better | ~sample._act_mask(frame)).float()
        loss = loss + (sample.loss(frame) * mask).sum()

        # Critic regression: `dist = sg timediff`
        #   We try to learn *min* dist, not just mean dist, by making each next dist-level predict `min(timediff, prev_level)`.
        #     (Worst-case, can also try tilted L1 loss.)
        dist_limit = torch.cat((timediff, frame_dist[:, :-1]), -1)
        loss = loss + (frame_dist - detach(timediff.min(dist_limit))).square().sum()

        # GAN-like penalization of ungrounded plans.
        loss = loss + (frame_pred_dist - detach(frame_pred_dist)*1.05).square().sum()

        log(0, False, torch, improvement = mask.mean() - (~sample._act_mask(frame)).float().mean())

        return loss
loss_fn = DODGE(loss, transition)

def replay(optim, current_frame, current_time):
    """Remembers a frame from a distant past, so that the NN can reinforce actions and observations when it needs to go to the present."""
    if len(replay_buffer) < 8: return

    time, ep, frame = random.choice(replay_buffer)

    # TODO: Maybe select only a subsample of `current_frame`'s cells to use as a goal.
    loss_fn(ep, frame, current_frame, current_time - time)
    #   TODO: The time should become a (cells,1) tensor.

    optim.step();  optim.zero_grad(True)



@sn.run
async def main():
    with State.Setter(lambda initial, current: initial*.001 + .999*current): # Soft-reset.
        with State.Episode() as life:
            with torch.no_grad():
                action = None
                frame = None
                time = 0
                while True:
                    await asyncio.sleep(.05) # TODO: Remove this to go fast.

                    obs, query, data_error, query_error = await sn.handle(sn.torch(torch, action))
                    #   (If getting out-of-memory, might want to chunk data/query processing.)

                    # Zero-pad `query` to be action-sized.
                    obs, query = torch.tensor(obs), torch.tensor(query)
                    query = torch.cat((query, torch.zeros(query.shape[0], obs.shape[1] - query.shape[1])), -1)

                    # Append prev-RNN-state and next-frame to the replay-buffer.
                    replay_buffer.append((
                        time,
                        life.clone(remember_on_exit=False),
                        frame,
                    ))

                    # Give prev-action & next-observation, and sample next action.
                    frame = torch.cat((action, obs), 0) if action is not None else obs
                    # TODO: Append `goal` to every step's input, unless it already has `'goal'`-named cells.
                    #   (…Once we actually have goals.)
                    transition_(frame)
                    with State.Episode(start_from_initial=False):
                        action = sample(query)
                        #   (Can also do safe exploration / simple planning, by `sample`ing several actions and only using the lowest-dist-sum (inside `with State.Episode(False): ...`) of those.)
                        #     (Could even plot the regret of sampling-one vs sampling-many, and see if/when it's worth it.)

                    replay(optim, frame, time)

                    time += 1