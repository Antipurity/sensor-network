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




#   TODO: …Should we make `DST` `state`-space in our impl too, and to bridge from obs-space to state-space, have a NN from obs to RNN-state, possibly just a single episode of the main NN with just the desired-obs inputted? TODO: But how would this model partial goals, AKA "we don't care about any of the other cells"? By training dists with partial goals?…
#     TODO: Since RNN state is unwieldy to actually condition on, should we learn a separate cell type just for goals, outputted by the main RNN per-cell? How is it learned, exactly? …We either need to store the unroll's goal-cells in replay (and maybe update this on replay), or do a no-DST unroll first then a with-DST unroll… Which approach is better?















# TODO: On replay, sample src & dst, then give `dst` to the RNN as its goal, and unroll several steps of tBPTT of the RNN with loss.
#   (Give `dst` to *every* step, right? Yeah, why not.)
# TODO: Loss:
#   TODO: Prediction (minimizing `sample_loss` of next-frame, *not* L2 loss directly), filtered by advantage (`(dist_replay.sum() < dist_policy.sum()).float()`). The policy-action is `sample`d.
#     TODO: Distances are summed up over *whole* states (prev-act + next-obs), BUT only for those cells where all non-name numbers are -1|1 (to handle act-in-obs the same as our own actions, while not pretending that obs can be sampled).
#     (Could also plot the filtered-through percentage.)
#   TODO: Min-dist learning: try `next_dist_replay = where(i<j, min(j-i, prev_dist_replay), next_dist*1.1)`. (Worst-case, can try tilted L1 loss.) (Storing `dst` in replays in increasing dist-to-it is *probably* not worth it.)
#   TODO: Maybe, `dist_policy = sg dist_policy*1.1` for GAN-like penalization of ungrounded plans.
#     TODO: …Isn't this unnecessary if sampling-probabilities are indeed copied correctly? Wouldn't on-policy rollouts correct our mistakes? …But what if there *aren't* on-policy rollouts.
#   (It's a simple loss, but quite general.)

# TODO: Maybe, mine for regret harder: replay-buffer prioritization of max-regret (of `b`: mean/max `dist(a,b) - (j-i)` among `a`) samples for fastest spreading of the influence of discovered largest shortcuts. At unroll, always overwrite the last sample; at replay, sample that plus randoms, and sort by minibatch-regret and write-back that sorted data. (Also good for easily sampling good unroll-time goals, getting more data in most-promising AKA max-regret areas.)

# TODO: For goals, use a PyTorch-backend `sn.Namer`, which puts `'goal'` in the last spot.
#   TODO: …Should we make the -2th slot the group id, so that we can specify per-group goals?… It's the best way to support both AND and OR for goals, isn't it…
#     TODO: How do we implement these per-cell extractions of grouped goals?
#   TODO: At unroll-time, generate observation-cell/s and estimate time-to-reach-it; at every step, append named-as-goal cells to obs (*unless there are any goal-cells in observations*); and when the prev estimated time runs out, pick new goal-cells and update the estimated time.
#     TODO: At unroll-time, save to the replay buffer.
#   TODO: At replay-time, TODO: what do we do, exactly?… How to learn per-cell goals?…
#     TODO: At replay-time, inside an episode: should pick a starting point (from which the episode-copy should be taken), sample a faraway goal from the future (preferably) and put it at the start, and unroll several from the starting point steps, applying the loss.

# (…We could also make the distance-network learn not only distance but its own prediction-regret (or maybe regret-per-step), so that goal-generation can maximize (a learned measure of) regret, at least by considering a few goals…)
#   Is this a good idea?
#   TODO: Do we maybe want to learn multiple metrics-about learning, such as misprediction? Which, exactly?

# (…Might want to do the simplest meta-RL env like in https://openreview.net/pdf?id=TuK6agbdt27 to make goal-generation much easier and make goal-reachability tracked — with a set of pre-generated graphs to test generalization…)
#   TODO: Maybe move `minienv` to `env/`, so that we can safely implement as many environments as we want?…

# TODO: …Also save/load the model…

# TODO: …May also want to implement importing the modules that the command line has requested, for easy env-switching…

# TODO: Should `sn.handle` also accept the feedback-error, which we can set to `1` to communicate bit-feedback?













import asyncio
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_default_tensor_type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)

from model.recurrency import State, SRWM

import sensornet as sn

from model.log import log, clear
#   TODO: Make `log` able to accept PyTorch tensors, awaiting their values async.



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
def to_np(x): return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x # TODO: Should use `sn.torch`, and make `log` able to accept futures — or maybe, just make it able to accept PyTorch tensors. Somehow.
def cat(*a, dim=-1): return torch.cat(a, dim)
def f(ins = state_sz, outs = ...): # A per-cell transform.
    if outs is ...: outs = ins
    return SkipConnection(nn.ReLU(), nn.LayerNorm(ins), nn.Linear(ins, outs))
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
            query = query.clone()
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
        is_act = ((act - self.zero).abs().min((act - self.one).abs()) < eps)[:, i:].all(-1, keepdim=True)
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



@sn.run
async def main():
    with State.Setter(lambda initial, current: initial*.001 + .999*current): # Soft-reset.
        with torch.no_grad():
            action = None
            prev_frame, frame = None, None
            while True:
                await asyncio.sleep(.05) # TODO: Remove this to go fast.

                obs, query, data_error, query_error = await sn.handle(sn.torch(torch, action))
                #   (If getting out-of-memory, might want to chunk data/query processing.)

                # Zero-pad `query` to be action-sized.
                obs, query = torch.tensor(obs), torch.tensor(query)
                query = torch.cat((query, torch.zeros(query.shape[0], obs.shape[1] - query.shape[1])), -1)

                # Give prev-action & next-observation, and sample next action.
                prev_frame, frame = frame, (torch.cat((action, obs), 0) if action is not None else obs)
                transition_(frame)
                action = sample(query)
                #   (Can also do safe exploration / simple planning, by `sample`ing several actions and only using the lowest-dist-sum (inside `with State.Episode(False): ...`) of those.)

                # TODO: On unroll, store in the replay buffer, as a contiguous sequence.
                #   TODO: …How to store the RNN state…
                #   TODO: What do we store, exactly?
                #     `prev_frame`
                #     …And nothing else? Only these two things?
                # TODO: To predict even the first frame of new-name obs, when storing to the replay buffer, pattern-match the labels of prev & next frames, and insert the missing zero-padded next-frame labels into prev-frame.
                #   …For this, want to preserve NumPy-side `prev_names` and `names`, don't we…
                #   TODO: …How do we do this, exactly…