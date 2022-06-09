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
# (TODO: Document how to use command-line args to import envs, and `module.Env()(sensornet)` with callbacks and impossible-to-collide naming and `.metric()` for logging a dict.)








# (BMIs with even futuristic levels of tech can't do anything like downloading knowledge into your brain or capturing and controlling bodily functions for a full-dive into virtual worlds. Even write-access (computer-to-brain) is very hard to make out, and may in fact need years of training. But read-access (brain-to-computer) can explore a human's internal reactions, so that the human can revisit those reactions at will without having to expend effort; in addition, predicting neural-input could be useful for learning good represetations. And maybe you'd need RTX 6090 to run the AI part in real-time, since it may be Gato-sized.)
#   ("Downloading knowledge" can only be done onto a computer, since a human brain really wasn't designed for this. Having a personal AI agent is the best way of downloading skills.) TODO: …Also, "mind uploading" is the same as "replacing all computer communication with AI-generated goal-directable byte streams" but for humans, with all the same caveats such as "need way way too much compute to encode everything exactly, and it is likely better to integrate the protocol with the AI tighter", right? "AI is goal-directed generative models of actions/observations" is starting to make sense now…
#     (And really, no one actually wants "downloading knowledge" to be an actual capability of brain-machine interfaces, without an indirection like that. Human culture isn't ready to treat humans like programs, with infinite copying and zero intrinsic value. For instance: markets get overtaken by the few that have all the knowledge and the bodies to put it into for profit; democracy loses connection to populations and becomes a tool of control by the most powerful ideas; war and murder become routine and one global superpower emerges since the only price of destruction is now some resources; creating new ideas rather than spreading existing ones becomes nearly impossible.)
# (TODO: …What we're requiring now is a file explaining the differences between analog and digital sampling, possibly even with "AI = goal-directed generative model", which our read-me can link to.)







# TODO: …A `model/???.py` file for VAE's normalization, with a test that with it, a NN from a 0…1 dataset to a hopefully-normal distribution would indeed learn to have EVERY output have 0 mean and 1 variance across the whole dataset?… (With it, making VAEs should be easy.)
#   From looking at code: for normal distributions, where we have mean & variance computed by NNs, the KL-loss to minimize is `mean**2 + var.exp()-var`, which makes each number's mean into 0 and variance into 1… …Isn't this exactly what I was looking for?
#     TODO: Write this down in a function (`loss_make_normal`).
#     TODO: Test `loss_make_normal` in a separate file, `model/vae.py`.
#       …Doesn't it make more sense to minimize `mean**2 + (var-1).exp()-var`, because this actually has a minimum at var=1? Need to test I guess…



# TODO: …Do we want a class for sampling given mean & variance (or given nothing for free-standing generation), with the KL-divergence-from-randn `.loss(mean, var)`?… Or is it too trivial, since the call is literally just `mean + var*torch.randn_like(var)`…



# TODO: Have `gated_generative_loss(dist_pred, pred, dist_target, target)`, which we can call both during the main loop (with DODGE, no backprop) and during replay (backprop-only, no DODGE).
#   …Would this learn goals, to maximize predicted-regret?

# TODO: …Maybe, DODGE should only optimize a small subset of parameters, so that its random-direction-picking doesn't drown the net in variance?…
#   …Which subset, though…
#     Maybe, when generating the direction, generate not the full randn vector but only make like 100 numbers optimized, the rest have 0 in the direction?
#       TODO: What hyperparams do we need?

# TODO: …We can actually incorporate VAEs into the loss without extra sampling: first put the real frame into the RNN (which would give us its dists & VAE-variables), *then* sample a fictitious frame to contrast with.
#   TODO: …Write down what we need our RNN (and `Sampler`) to do to incorporate VAEs…
#     Do we want to make `transition_` return 2 extra items (mean & variance), and maybe accept one extra sampled-`latent` arg, and return a `namedtuple` because the output-complexity is getting ridiculous?



# TODO: Stop sampling goals from the replay buffer, and start *generating* goals:
#   TODO: In `loss`, get latents of `dst` first, then use those latents to turn 0-filled `dst`-sized cells into `dst` (in the same GPU-call as the rest of the `frame`) if good.
#     TODO: For goals, "good" means "sample-dist is higher" (whereas for actions, "good" means "sample-dist is lower"). (Goals are then neither too hard (AKA when never occuring as real targets) nor too easy (AKA when real-dist is lower than expected).)
#       (This is [AMIGo](https://arxiv.org/pdf/2006.12122.pdf)-inspired. Alternatively, could make goals learn regret rather than dists and maximize regret rather than dists; or even be [AdaGoal](https://arxiv.org/pdf/2111.12045.pdf)-like and have an ensemble of RNNs, and maximize dist-error.)
#   TODO: At unroll-time & goal-selection-time, rather than choosing anything from the replay buffer, simply choose a random number of goal-cells to generate (and a `torch.randn` latent) then generate them.



# TODO: Maybe, allow `sn.Int`s to have only `1` option, so that it's easy for users to dispatch "events" which can be set as goals. Maybe also turn no-data no-type `.set` calls into such events.
# TODO: Maybe, if `choices_per_cell` is not defined in `sn.info`, it should be treated as `sn.cell_shape[-1]`, and `sn.Int`'s query should get argmax — possibly set as one-hot encodings too… (If ints are not explicitly supported, provide at least some inefficient support of them.) (And, `sn.info['analog']` checks would probably be unnecessary with this.)
# TODO: Also support [mu-law-encoded](https://en.wikipedia.org/wiki/%CE%9C-law_algorithm) (and/or linearly-encoded) floats-in-ints. `IntFloat(*shape, opts=256, mu=mu, bounds=(min,max))`. `mu=0` should be special-cased to be linear.
# TODO: Also support fixed-size strings (with tokenizers) and image-patches. (The most important 'convenience' datatypes.)
# TODO: Maybe, have `.metrics()` on handlers, and have two metrics: cells-per-second (exponentially-moving average) (which doesn't count the time spent on waiting for data) and latency (EMA too) (time from a `.handle` call to when its `feedback` is actually available to us, in seconds).






# TODO: An env that has both analog actions and analog goals: make the goal a random image (plus the "at the final-state" bit), then get like 3 images (exposing "NOT at the final-state"), then expose the sum of those actions and the fact that this is the final-state.
#   TODO: …Try not L2 prediction but a GAN, with the 'discriminator' being the distance network (in other words, only do self-imitation for digital cells, and do DDPG for analog cells instead — after giving random noise as an extra input)?
#     TODO: …Or is it sufficient to expose a ghost digital query, which will get reinforced when on correct paths for enhanced L2 prediction of the correct path?… (This *might* even be usable as 'full-RNN-state' goals, with zero effort on our side…) (A bit like humans using language/thoughts to augment their learning.)
#     (Possibly a VAE, with each cell having a few extra numbers on input and output, past's prediction being conditioned on future's output. Of course, filtered by improved-distance. The only problem is that the first generation of the past doesn't know the future's output — in addition to the latent space not being fully covered, causing blurriness, which we'd need random-sampling and DDPG/GAN to fix.)
# TODO: …Is it possible to have an analog-reward-goal which isn't ever actually encountered, but is set to 1 so that the model always maximizes reward? What would we need for this? The unroll-time measured-distance-to-analog-goal, which is 0…1 and is only there for higher precision?… …But how would we detect if cells are same-named, and what would we do if we don't actually have a same-name observation cell…














# TODO: Run & fix the copy-task in `env/copy.py`, to test our implementation.

# TODO: …Might want to do the simplest meta-RL env like in https://openreview.net/pdf?id=TuK6agbdt27 to make goal-generation much easier and make goal-reachability tracked — with a set of pre-generated graphs to test generalization…

# TODO: Make `graphenv` work not globally but in a class.



# TODO: Make goals the full-RNN-states (latents), not the inputs, like in http://proceedings.mlr.press/v100/nair20a/nair20a.pdf but with discrete-sampling instead of CC-VAEs:
#   TODO: Alloc SRC `'-goal-src-state-'` and DST `'-goal-dst-state-'` for personal use. (Maybe make `modify_name` assert that these are unused.) (The current `'goal'` could probably act as SRC.)
#   TODO: On unroll, in a throwaway episode, rename the frame to DST and sample and put the result into the replay buffer.
#   TODO: On unroll, to pick a goal, fetch some random subset of cells from the replay buffer, rename to SRC and 0-fill their values, and in a throwaway episode, sample (possibly for all goal-groups at once) and set as the goal.
#   TODO: On replay, select random subsets (to support partial goals) of faraway latents as destinations, not of inputs.
#     TODO: Have a 50/50 chance to select either faraway-latents or inputs as dst, so that we don't lose the ability for envs to specify goals.
#       (With this, the interface will unify not only obs-and-act and obs-and-goal, but also allow training to reach other agents' full-RNN-state goals, by simply exposing their inner states as observations. "The individual is obsolete", mm.)
#   TODO: On replay, in a throwaway episode, self-imitate (if regret is positive, max the sample-probability of) faraway-dst's SRC-cells given DST-renamed and 0-filled versions of them.
#     TODO: Also, there, self-imitate random subsets of SRC-renamed frames given their cell-name-only versions.
#   …Hold on: if we allow *actions* to be parts of goals, and allow (maybe enforce) a few untethered actions per step, then does this automatically do full-RNN-state goals (with both goal-striving-by-src and goal-self-imitation-by-dst)?…














import asyncio
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd.forward_ad as fw
torch.set_default_tensor_type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)

from model.recurrency import State, SRWM
from model.dodge import DODGE, LayerNorm, Softmax, ReLU, detach

import sensornet as sn

from model.log import log, clear



# Hyperparameters.
cell_shape = (8,8,8,8, 64)
sn.shape(*cell_shape)

state_sz, goal_sz = 256, 256
slow_mode = .05 # Artificial delay per step.

bits_per_chunk = 8 # How to `sample`.
#   TODO: Should be `frexp(choices_per_cell-1)[1]`, none of that "chunk" business.
#   TODO: Also, should be named `bits_per_cell`, right? …And maybe just have `choices_per_cell` as the main hyperparam, in case we're feeling not-power-of-2.

# TODO: …Do we maybe also want a hyperparam for "how many actions to pick between at unroll-time"?… (…Which we can *technically* allow the model to modify, for better or worse — same as learning-rate and replays-per-step and max-replay-buffer-len… Though maybe it's better to just learn an "expected improvement with this action-count" NN/table.)

lr = 1e-3
replays_per_step = 2
max_replay_buffer_len = 1024

save_load = '' # A filename, if saving/loading occurs.
steps_per_save = 1000

sn.info = {
    'docs': """TODO:""",
    # TODO: Also mention that we clip all inputs/outputs to -1…1.
    'analog': True,
    'choices_per_cell': 2 ** bits_per_chunk,
}



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
    # Remove inter-env collisions by adding the group ID to the end of their names.
    #   (`sn.RawFloat` and `sn.Int` prepend a name-part, so `res` contains one less.)
    assert modify_name.ctx is not None, "Sending data not in an env's sensor; don't send it in callbacks of queries, instead remember to send it on the next step"
    res = [name[i] if i < len(name) else None for i in range(len(cell_shape) - 2)]
    assert not isinstance(res[-1], tuple), "The group-id shouldn't be a tuple (to make our lives easier)"
    res[-1] = modify_name.ctx if res[-1] is None else (modify_name.ctx + '.' + res[-1])
    return tuple(res)
sn.modify_name.append(modify_name)
envs = ['graphenv'] if len(sys.argv) < 2 else sys.argv[1:]
envs = { e: prepare_env(e) for e in envs }
replay_buffer = []
# Our interface to multigroup partial goals (AND/OR goals): the last name part is `group_id`, AKA user ID.
#   Via carefully engineering the loss, users can have entirely separate threads of experience that reach ALL the goals in their group (but the NN can of course learn to share data between its threads).
def goal_group_ids(frame):
    """Returns the `set` of all goal-group IDs contained in the `frame` (a 2D NumPy array), each a byte-objects, suitable for indexing a dictionary with."""
    return set(name.tobytes() for name in np.unique(frame[:, sum(cell_shape[:-2]) : sum(cell_shape[:-1])], axis=0))



class Sampler:
    """
    GANs, VAEs, contrastive learning, BYOL: direct prediction of inputs averages them, so we need tricks to learn actual input distributions    
    Autoregressive discrete sampling: AWAKEN

    (TODO: …Regrettably, the meme above holds no water anymore with the "AI = goal-conditioned generative models" viewpoint; GANs/VAEs are just analog-space generative models… So, need better exposition.)

    Uses a `cells × action_size → cells × 2**bits_per_chunk` RNN (with hidden `State`) to sample from bit sequences, without averaging in-action correlations away, as compute-efficiently as possible (vary `bits_per_chunk` depending on RNN's size).

    Same idea as in language modeling, and [Gato](https://arxiv.org/abs/2205.06175), though per-cell and with explicit bits.

    (This adds autoregressive in-cell sampling. Cells are still processed in parallel, because envs should make cells independent of each other by design. And cell-groups are already autoregressively processed by RNNs.)
    """
    def __init__(self, fn, bits_per_cell, start, end): # TODO: Should we support non-power-of-2 sampling? (And compute `bits_per_cell` as `frexp(opts-1)[1]`.)
        self.bits_per_cell = bits_per_cell
        self.fn, self.start, self.end = fn, start, end
        self.softmax = Softmax(-1)
    def __call__(self, query):
        """Given a name-only `query`, samples a binary action.

        The result is `(action, repr)`. Use `action` to act; use L2 loss on `repr` and `.target(action)` to predict the action.

        This will advance RNN `State`, so do `State.Episode` shenanigans if that is unwanted."""
        is_ana = self.analog_mask(detach(query)).float()
        i, j, cells = self.start, self.end, query.shape[0]
        input = torch.cat((query, torch.zeros(cells, j - i)), -1)
        name, logits = self.fn(input)
        assert logits.shape[-1] == 2 ** self.bits_per_cell
        analog = torch.cat((name, logits[..., :j-i]), -1)
        with torch.no_grad():
            digprob = self.softmax(detach(logits))
            indices = digprob.multinomial(1)[..., 0]
            bits = sn.Int.encode_ints(sn, indices, self.bits_per_cell)
            digital = torch.cat((query, bits, torch.zeros(cells, j - self.bits_per_cell - i)), -1)
        action = is_ana * analog + (1-is_ana) * digital
        repr = torch.cat((is_ana * logits + (1-is_ana) * digprob), -1)
        return action, repr
    def target(self, act):
        """Converts a presumably-sampled `action` to its `repr`, which can be used to compute L2 loss."""
        # The representation: for analog, just the zero-padded prediction as-is; for digital, the post-softmax probability distribution.
        i, j, cells = self.start, self.end, act.shape[0]
        options = 2 ** self.bits_per_cell
        act = detach(act)
        with torch.no_grad():
            is_ana = self.analog_mask(act).float()
            analog = torch.cat((act, torch.zeros(cells, options - j)), -1)
            indices = sn.Int.decode_bits(sn, act[:, i : i + self.bits_per_cell]) % options
            digital = F.one_hot(indices, options).float()
            return is_ana * analog + (1-is_ana) * digital
    @staticmethod
    def goal_mask(frame):
        return frame[:, 0:1] > 0
    @staticmethod
    def analog_mask(frame):
        return frame[:, 1:2] > 0
    @staticmethod
    def as_goal(frame):
        return torch.cat((torch.ones(frame.shape[0], 1), frame[:, 1:]), -1)
sample = Sampler(lambda x: transition_(x)[0:1], bits_per_chunk, sum(cell_shape[:-1]), sum(cell_shape))



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
        LayerNorm(ins), ReLU(), SRWM(ins, ins, heads=2, Softmax=Softmax),
        LayerNorm(ins), ReLU(), nn.Linear(ins, outs),
    )



try:
    if not save_load: raise FileNotFoundError()
    transition = torch.load(save_load)
except FileNotFoundError:
    transition = nn.Sequential(
        # Input prev-actions and next-observations, output name and `sample`-logits and distance and regret.
        #   We predict the next input, thus serving as both an RL agent and a goal-directed generative model.
        #   We overcome L2-prediction outcome-averaging via autoregressive `sample`ing, though it's only fully 'solved' for binary outputs (AKA actions). TODO: …We'll be doing proper autoregressive modeling now, so no need for this remark.
        h(sum(cell_shape), state_sz),
        h(state_sz, state_sz),
        h(state_sz, sum(cell_shape[:-1]) + 2 ** bits_per_chunk + 1 + 1),
    )
def transition_(x):
    """Wraps `transition` to return a tuple of `(sample_info, distance, distance_regret)`."""
    y = transition(x)
    name = sn.cell_size - cell_shape[-1]
    return y[..., :name], y[..., name:-2], 2 ** y[..., -2:-1], 2 ** y[..., -1:]
dodge = DODGE(transition)
optim = torch.optim.Adam(transition.parameters(), lr=lr)



def loss(prev_ep, frame, dst, timediff, regret_cpu):
    """`loss(prev_ep, frame, dst, timediff)`

    Predicts the `prev_frame`→`frame` transition IF it's closer to the goal than what we can sample.

    - `prev_ep` is the context that knows `prev_frame`.
    - `frame` is the immediately-next prev-action and next-observation cells.
    - `dst` is some faraway goal.
    - `timediff` is the distance to that goal, as a `(frame_cells + dst_cells, 1)`-shaped int32 tensor. Must always be *more* than 0."""

    with prev_ep:
        # Multigroup ("AND") goals, where many threads of experience (with their group IDs) can pursue separate goals.
        #   (The group ID replaces the last name-part.)
        is_learned = torch.rand(frame.shape[0], 1) < (.1+.9*random.random())
        dst_group_id = torch.rand(1, cell_shape[-2])*2-1
        src_group_id = torch.where(is_learned, dst_group_id, torch.rand(frame.shape[0], cell_shape[-2])*2-1)
        frame = torch.cat((frame[:, :sum(cell_shape[:-2])], src_group_id, frame[:, sum(cell_shape[:-1]):]), -1)
        dst_group_id = dst_group_id.expand(dst.shape[0], cell_shape[-2])
        dst = torch.cat((dst[:, :sum(cell_shape[:-2])], dst_group_id, dst[:, sum(cell_shape[:-1]):]), -1)

        # Name `dst` with the fact that it's all goal-cells, and add it to the `frame`.
        dst = sample.as_goal(dst)
        frame = torch.cat((dst, frame), 0)
        is_learned = torch.cat((torch.full((dst.shape[0], 1), True), is_learned), 0).float()

        # What to contrast `frame` with.
        with State.Episode(start_from_initial=False):
            frame_pred, frame_pred_repr = sample(frame[:, :-cell_shape[-1]])

        # Distances of `frame` and `frame_pred` to contrast.
        with State.Episode(start_from_initial=False):
            _, _, frame_dist, _ = transition_(frame)
        with State.Episode(start_from_initial=False):
            _, _, frame_pred_dist, frame_pred_regret = transition_(frame_pred)

        # Critic-regularized regression: if `frame_pred` regrets not being `frame`, make it into `frame`.
        dist_is_better = detach(frame_dist)[:, -1:] < detach(frame_pred_dist)[:, -1:]
        mask = (dist_is_better | ~sample.analog_mask(frame)).float()
        predict_loss = (is_learned * (frame_pred_repr - sample.target(frame)).square() * mask).sum()

        # Remember our regret of not being the `frame`. *Only* remember what we regret.
        regret = (is_learned * (frame_pred_dist[:, -1:] - frame_dist[:, -1:])).sum() / (is_learned.sum() + 1e-2)
        regret_cpu.copy_(detach(regret), non_blocking=True)

        # Learn the regret, since estimating prediction-error by training several copies of the model is too expensive.
        regret_loss = 0 # (is_learned * (frame_pred_regret.log2() - detach(regret).clamp(1e-5).log2()).square()).sum() # TODO:

        # Critic regression: `dist = sg timediff`, but *optimistic*, meaning, min dist and not mean dist.
        #   (Worst-case, can also try tilted L1 loss AKA distributional RL.)
        dist_limit = frame_dist + .5 # Always be able to increase, but not by too much.
        dist_loss = (is_learned * (frame_dist.log2() - detach(timediff.min(dist_limit)).log2()).square()).sum()

        # GAN-like penalization of ungrounded plans.
        dist_penalty = 1.05
        ungrounded_dist_loss = 0 # (is_learned * (frame_pred_dist - detach(frame_pred_dist) * dist_penalty).square()).sum() # TODO:

        log(0, False, torch, improvement = mask.mean() - (~sample.analog_mask(frame)).float().mean())
        #   TODO: Why is improvement always 0, even though `predict_loss` is not? …Wait, is it `nan` sometimes, from looking at the plots?…
        #     (From looking at the plot, it's `nan` sometimes in the beginning, but not in the end.)
        log(1, False, torch, predict_loss=predict_loss, regret_loss=regret_loss, dist_loss=dist_loss, ungrounded_dist_loss=ungrounded_dist_loss)
        n = 2
        for name, env in envs.items():
            if hasattr(env, 'metric'):
                log(n, False, torch, **{name+'.'+k: v for k,v in env.metric().items()})
                n += 1
        # TODO: …How can we possibly fix the loss not decreasing?…

        loss = predict_loss + regret_loss + dist_loss + ungrounded_dist_loss
        return loss

def replay(optim, current_frame, current_time):
    """Remembers a frame from a distant past, so that the NN can reinforce actions and observations when it needs to go to the present."""
    if len(replay_buffer) < 8: return

    L = 0
    for _ in range(replays_per_step):

        time, ep, frame, regret_cpu = random.choice(replay_buffer)

        # Learn partial ("OR") goals, by omitting some cells.
        #   (Note: it's possible to sample several goals and pick a 'best' one, [like AdaGoal does](https://arxiv.org/abs/2111.12045). But that works best with a measure of 'error' to maximize, and our replay-buffer already kinda maximizes regret.)
        #     (Could be done with either the `regret` output of `transition_`, or by keeping several `transition`s to measure ensemble disagreement. Could also [train some generative model](http://proceedings.mlr.press/v100/nair20a/nair20a.pdf), but those are hard to train, hence the sampling.)
        #       (All of which is too expensive.)
        dst_is_picked = np.random.rand(current_frame.shape[0]) < (.05+.95*random.random())
        dst = current_frame[dst_is_picked]

        # Learn.
        timediff = torch.full((frame.shape[0] + dst.shape[0], 1), float(current_time - time))
        L = L + loss(ep, frame, dst=dst, timediff=timediff, regret_cpu=regret_cpu)

        # If our replay buffer gets too big, leave only max-regret samples.
        if len(replay_buffer) > max_replay_buffer_len:
            replay_buffer.sort(key = lambda sample: sample[3], reverse=True)
            del replay_buffer[(max_replay_buffer_len // 2):]

    # Optimize NN params.
    dodge.minimize(L)
    optim.step();  optim.zero_grad(True)



@sn.run
async def main():
    with fw.dual_level():
        with State.Setter(lambda state, to: state.initial*.001 + .999*to): # Soft-reset.
            dodge.restart()
            with State.Episode() as life:
                with torch.no_grad(): # TODO: This should only apply to the parts where we aren't learning from the replay buffer.
                    prev_q, action, frame = None, None, None
                    goals = {} # goal_group_id → (gpu_goal_cells, cpu_expiration_time)
                    time = 0 # Hopefully this doesn't get above 2**31-1.
                    while True:
                        if slow_mode > 0:
                            await asyncio.sleep(slow_mode)

                        obs, query, error = await sn.handle(sn.torch(torch, action))
                        obs   = np.nan_to_num(  obs.clip(-1., 1.), copy=False)
                        query = np.nan_to_num(query.clip(-1., 1.), copy=False)
                        #   (If getting out-of-memory, might want to chunk data/query processing.)

                        # (The replay buffer won't want to know any user-specified goals.)
                        #   (And fetch goal-group IDs to add constraints for exploration, even if the env has set some goals.)
                        frame_names = np.concatenate((prev_q, obs[:, :prev_q.shape[1]]), 0) if prev_q is not None else obs[:, :query.shape[1]]
                        not_goal_cells = ~sample.goal_mask(frame_names)[:, 0]
                        prev_q = query
                        groups = goal_group_ids(frame_names)

                        # Zero-pad `query` to be action-sized.
                        obs, query = torch.tensor(obs), torch.tensor(query)
                        frame = torch.cat((action, obs), 0) if action is not None else obs
                        frame_to_save = frame[not_goal_cells]
                        life_to_save = life.clone(remember_on_exit=False)

                        # Delete/update our `goals` when we think we reached them.
                        for group, (cells, expiration_cpu, expiration_gpu) in goals.copy().items():
                            if time > expiration_cpu:
                                del goals[group]
                        if len(replay_buffer):
                            for group in groups:
                                if group not in goals:
                                    goal = random.choice(replay_buffer)[2]
                                    goal = goal[np.random.rand(goal.shape[0]) < (.05+.95*random.random())]
                                    goal = sample.as_goal(goal)
                                    expiration_cpu = torch.tensor(time+10000, device='cpu').int()
                                    expiration_gpu = torch.tensor(time+10000).int()
                                    goals[group] = (goal, expiration_cpu, expiration_gpu)
                                    # And change DODGE direction when changing the goal.
                                    if random.randint(1, len(goals)) == 1:
                                        dodge.restart()

                        # TODO: Here is the perfect place to predict `frame`'s data from its zeroed-out-data version, if we DODGE-predict online. (Though, might want to remove the goal-cells.) (Replays can still do prediction, just, one-step.)
                        #   (Gating the prediction by improvement is easy here, because we'll have 2 distances: of prediction and of `frame` a bit below. And even of the `action` from the previous step…)

                        # Give goals to the RNN.
                        extra_cells = []
                        for group, (cells, expiration_cpu, expiration_gpu) in goals.items():
                            extra_cells.append(cells)
                        if len(extra_cells): frame = torch.cat([*extra_cells, frame], 0)

                        # Give prev-action & next-observation, remember distance estimates, and sample next action.
                        _, _, dist, regret = transition_(frame)
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
                            action = sample(query)[0]
                            #   (Can also do safe exploration / simple planning, by `sample`ing several actions and only using the lowest-dist-sum (inside `with State.Episode(False): ...`) of those.)
                            #     (Could even plot the regret of sampling-one vs sampling-many, and see if/when it's worth it.)

                        # Learn. (Distance is always non-0.)
                        replay(optim, frame, time)

                        # Save. (Both replay-buffer and disk.)
                        replay_buffer.append((
                            time,
                            life_to_save,
                            frame_to_save,
                            torch.tensor(1000., device='cpu'),
                        ))
                        if save_load and time % steps_per_save == 0:
                            torch.save(transition, save_load)

                        time += 1