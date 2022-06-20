"""
# The model that we implement in this file

Goal-directed generative model of the world.

Let's write down the most generic discrete-time model of the world: `universe: state → state`. This describes one boring line/circle through time.

Now, let's consider an agent with bidirectional communication with its world, receiving `obs`ervations and returning `act`ions: `universe: (state, act) → (state, obs)`. Now we have a whole infinite graph of possibilities, with the path selected by `act`ions.

Where that agent ultimately goes can be named its goal. In the sequence of the agent's `obs`ervations, this can be written as `SRC <path> DST <path>: MID0 MID1 MID2 … DST`. This is very [similar to ](https://paperswithcode.com/task/text-infilling)[text ](https://nlp.stanford.edu/pubs/donahue2020infilling.pdf)[infilling](https://beta.openai.com/docs/guides/completion/inserting-text), where `DST` is either in `state`-space or in `obs`-space but variable-length.) (RL writes goals as one-number-utility functions. We follow [upside-down RL](https://arxiv.org/abs/1912.02877) here, but both approaches are general enough to implement each other.

To reach goals most efficiently, the agent should build a map of that graph of possibilities, to run shortest-path queries on. Why *not* use something like a [Transformer](https://arxiv.org/abs/1706.03762) with [RNN-state](https://arxiv.org/abs/2202.05780), and do next-`obs` prediction? Without an ideological agenda, there's no reason to claim that Deep Learning can't succeed at general intelligence, being the most data-driven and successful approach that we have.

`act`ing can actually be exactly the same as next-frame prediction, if actions are included in next-frame observations; then it's easy for human and AI actions to be treated exactly the same. But we do want shortest/best paths and not just any paths, so our prediction has to be [critic-](https://arxiv.org/abs/2006.15134)[regularized](https://arxiv.org/abs/1806.05635); if `obs` are stochastic, the model may become overly optimistic instead of averaging/smudging the predictions.

# (TODO: Mention that we require PyTorch 1.10+ because we use forward-mode AD.)
# (TODO: Document how to use command-line args to import envs, and `module.Env()(sensornet)` with everything scheduled synchronously and impossible-to-collide naming and `.metric()` for logging a dict.)

# Relation to self-supervised learning

In recent years, deep learning has seen the rise of methods for self-supervised learning. Among them, a particular kind of non-contrastive methods for extracting useful representations from images bear strong resemblance to URL. TODO: We're now neither URL nor BYOL-related. We're now simply doing generative modeling. Image augs are good to note, but other things are not.

- [Barlow twins](https://arxiv.org/abs/2103.03230): `repr(state1) = repr(state2)` using a cross-correlation loss for soft-whitening.

- [SimSiam](https://arxiv.org/abs/2011.10566): basically `predictor(repr(state1)) = sg(repr(state2))`. Here, `state1` and `state2` are augmented image views put through a neural network.
    - [Bootstrap Your Own Latent](https://arxiv.org/abs/2006.07733): same, but the predicted `repr` on the right is replaced with a slow-moving average of the trained `repr` on the left.

(Image augmentations include: crop, blur, make it grayscale, change brightness and/or contrast and/or saturation. Models get good by learning to ignore those in their learned representations.)

This is similar to just predicting the next input in RNNs, possibly min-distance-filtered. The differences from, say, BYOL: no `repr` and thus no momentum-slowing; asymmetry; and `predictor` is conditioned on both past-RNN-state and action.

Further, the ability to reproduce [the human ability to learn useful representations from interacting with the world](https://xcorr.net/2021/12/31/2021-in-review-unsupervised-brain-models/) can be said to be the main goal of self-supervised learning in computer vision. The structure of the body/environment is usable as data augmentations: for images, we have eyes, which can crop (movement & eyelids), make it grayscale [(eye ](http://hyperphysics.phy-astr.gsu.edu/hbase/vision/rodcone.html)[ro](https://en.wikipedia.org/wiki/Rod_cell)[ds)](https://en.wikipedia.org/wiki/File:Distribution_of_Cones_and_Rods_on_Human_Retina.png), scale and flip and rotate (body movement in 3D), blur (un/focus), adjust brightness (eyelashes), and do many indescribable things, such as "next word" or "next sound sample after this movement".
"""








# (BMIs with even futuristic levels of tech can't do anything like downloading knowledge into your brain or capturing and controlling bodily functions for a full-dive into virtual worlds. Even write-access (computer-to-brain) is very hard to make out, and may in fact need years of training. But read-access (brain-to-computer) can explore a human's internal reactions, so that the human can revisit those reactions at will without having to expend effort; in addition, predicting neural-input could be useful for learning good represetations. And maybe you'd need RTX 6090 to run the AI part in real-time, since it may be Gato-sized.)
#   ("Downloading knowledge" can only be done onto a computer, since a human brain really wasn't designed for this. Having a personal AI agent is the best way of downloading skills.) TODO: …Also, "mind uploading" is the same as "replacing all computer communication with AI-generated goal-directable byte streams" but for humans, with all the same caveats such as "need way way too much compute to encode everything exactly, and it is likely better to integrate the protocol with the AI tighter", right? "AI is goal-directed generative models of actions/observations" is starting to make sense now…
#     (And really, no one actually wants "downloading knowledge" to be an actual capability of brain-machine interfaces, without an indirection like that. Human culture isn't ready to treat humans like programs, with infinite copying and zero intrinsic value. For instance: markets get overtaken by the few that have all the knowledge and the bodies to put it into for profit; democracy loses connection to populations and becomes a tool of control by the most powerful ideas; war and murder become routine and one global superpower emerges since the only price of destruction is now some resources; creating new ideas rather than spreading existing ones becomes nearly impossible.)
# (TODO: …What we're requiring now is a file explaining the differences between analog and digital sampling, possibly even with "AI = goal-directed generative model", which our read-me can link to.)
#   (…A tutorial, maybe?)































# TODO: Run & fix the copy-task in `env/copy.py`, to test our implementation.
#   TODO: Figure out why we have float64.

# TODO: An env that has both analog actions and analog goals: make the goal a random image (plus the "at the final-state" bit), then get like 3 images (exposing "NOT at the final-state"), then expose the sum of those actions and the fact that this is the final-state.
#   TODO: Also an env for out-of-distribution goals: expose a reward that caps off at 0 (reward of 'how close the image is to the non-goal target image' maybe), and always say that it must be 1.

# TODO: Might want to do the simplest meta-RL env like in https://openreview.net/pdf?id=TuK6agbdt27 to make goal-generation much easier and make goal-reachability tracked — with a set of pre-generated graphs to test generalization…

# TODO: Make `graphenv` work not globally but in a class.






# TODO: After pure-DODGE fails to learn well:
#   TODO: Figure out how to use `DODGE` simultaneously with regular backprop w.r.t. weights. And make at least `per_goal_loss` do regular backprop *in addition to* DODGE.
#   TODO: Instead of simultaneous DODGE+backprop, make backprop follow at ≈1000 steps back, and make it aware of dist/smudge targets for `global_loss` if available.
#   TODO: Do per-`State`-matrix-`W` synth-grad, via a neural net that takes `q` (a random query) and `q@W` (its value) and returns `q@W.grad`.

# TODO: Maybe, if `choices_per_cell` is not defined in `sn.info`, it should be treated as `sn.cell_shape[-1]`, and `sn.Int`'s query should get argmax — possibly set as one-hot encodings too… And, could probably polyfill analog support too, by making `RawFloat` defer to `Float` when not `sn.info['analog']`.
# TODO: Also support fixed-size strings (with tokenizers) and image-patches (auto-fit patch size to cell size, and zero-pad the image for simplicity, and input dims should be max dims, not guaranteed dims; and in fact, can't all of this just be incorporated into `RawFloat`?). (The most important 'convenience' datatypes.)
# TODO: Maybe, have `.metrics()` on handlers, and have two metrics: cells-per-second (exponentially-moving average) (which doesn't count the time spent on waiting for data) and latency (EMA too) (time from a `.handle` call to when its `feedback` is actually available to us, in seconds).
# TODO: …What if we did make `sn.query` return double-`await`ables, so first `await` (fulfilled in either `.discard` or `._take_data`) would wait for submission and second `await` would actually give the data? (Then efficient piping is as simple as single-`await`ing in an infinite loop, and not just here but also wherever we want anything like a stream of data, such as dynamically-sized strings… And, no need to make `.set` `await`able, since it can just await queries. Overall, a good idea: usability is key. …Then again, we *could* hide the complexity of `sn.listeners` queues in our own classes…)
# TODO: …Possibly, have the `sn.func(name, sn=sn)` async-func decorator that integrates with Python datatypes to expose both inputs and outputs as data whenever the func is called, or when `None` is returned, requests output from `sn`… ("Mind uploading" for code, condensed to a single line of code.)
#   TODO: Maybe, make `sn.List` add an additional dimension for `Int`/`RawFloat` to iterate over, instead of having unique names for each part.



# TODO: Compressed history as goals:
#   TODO: NEED an env that consists of many buttons, which have to be pushed in a password-sequence that changes rarely, with a bit-indicator for "was the last button correct". With enough buttons, state-based exploration should discover nothing, whereas proper full-past exploration *should* discover the sequence eventually.
#   TODO: Have a hyperparam that makes `local_dist` only consider `src`-cells (non-inputs) for smudgings, and given an env that needs nothing but exploration, verify that only using `src`-cells still works.
#   TODO: Find some way of compressing the past that works in these envs — some embedding that's influenced-by (predicts) the past. Likely: from `cells_override(ones, goal=False, analog=?, group)`-input `src` cells, computed from the future, predict all past cells (prev frame's inputs and `src`s).










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
from model.vae import make_normal, normal

import sensornet as sn

from model.log import log, clear



# Hyperparameters.
cell_shape = (8,8,8,8, 256)
sn.shape(*cell_shape)

state_sz = 256
latent_sz = 32 # For VAE generation.
choices_per_cell = 256

slow_mode = .05 # Artificial delay per step.

clamp = 1. # All inputs/outputs are clipped to `-clamp…clamp`.
top_p = .99 # Nucleus sampling (`1` to disable), for discounting the unreliable probability tail for digital queries: https://arxiv.org/abs/1904.09751
digital_embs = 'use' # 'no'|'use'|'learn': whether RNN-inputs will be binary-masks or random-vectors or learned-vectors, for digital cells.

lr = 1e-3
dodge_optimizes_params = 1000 # `DODGE` performs better with small direction-vectors.
dodge_len = 1. # `DODGE` is BPTT-like, and each learning episode will run for `dodge_len` multiplied by the average steps-to-reach-goal.
optimism = .5 # If None, average of dist&smudging is learned; if >=0, pred-targets are clamped to be at most pred-plus-`optimism`.

logging = True
save_load = '' # A filename, if saving/loading occurs.
steps_per_save = 1000

sn.info = sn.default.info = {
    'docs': """TODO:

    We clip all inputs/outputs to -1…1.""",
    'analog': True,
    'choices_per_cell': choices_per_cell,
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
    sn.sensors.add(sensor_with_name)
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



# Sampling actions.
class Sampler:
    """
    Handles analog+digital sampling from the `transition` RNN model.

    RL is goal-directed generative models, so sampling is enough for both (optimistic) prediction of observations and generation of good actions.

    Uses a `cells × action_size → cells × choices_per_cell` RNN (with hidden `State`) to sample from bit sequences, without averaging in-action correlations away, as compute-efficiently as possible (vary `choices_per_cell` depending on RNN's size to maximize information throughput).

    Digital sampling explicitly models all probabilities of each choice. It is the idea that powers autoregressive language models and many agents, including [GPT-3](https://arxiv.org/abs/2005.14165) and [Gato](https://arxiv.org/abs/2205.06175). Unlike analog sampling (i.e. a GAN), cells have to be processed not in parallel but in sequence, which is already handled by the `sn.Int` datatype.
    """
    __slots__ = ('cpc', 'bpc', 'fn', 'start', 'end', 'softmax')
    def __init__(self, fn, choices_per_cell, start, end):
        from math import frexp
        bits_per_cell = frexp(choices_per_cell - 1)[1]
        assert bits_per_cell <= end-start, "Not enough space to encode digital actions (RNN input is ridiculously small)"
        assert end-start <= choices_per_cell, "For simplicity, there should be more choices than values"
        self.cpc = choices_per_cell
        self.bpc = bits_per_cell
        self.fn, self.start, self.end = fn, start, end
        self.softmax = Softmax(-1)
    def __call__(self, query: torch.Tensor, query_names: np.ndarray, latent=...):
        """Does analog+digital sampling.
        
        Inputs: a full-size `query` (zero-pad names if needed, or `augment` frames), its name-only NumPy version (for `Sampler.sample_order`), and optionally `latent`.

        The result is `(action, logits)`. Use `action` to act; use L2 loss between `logits` and `.target(target_action)` to copy the target. (Might also want to discourage generated-but-unseen targets, similarly to GANs or [Unlikelihood Training](https://arxiv.org/abs/1908.04319).)

        This will advance RNN `State`, so reset via `State.Episode` if that is unwanted."""
        i, j, cells = self.start, self.end, query.shape[0]
        assert len(query.shape) == len(query_names.shape) == 2 and query.shape[-1] == j
        action = torch.zeros(cells, j)
        logits = torch.zeros(cells, i + self.cpc)
        for inds in Sampler.sample_order(query_names):
            inds = torch.as_tensor(inds)
            query_part = query[inds]
            is_ana = Sampler.analog_mask(detach(query_part)).float()
            name_and_logits = self.fn(query_part, ... if latent is ... else latent[inds])[0]
            assert len(name_and_logits.shape) == 2 and name_and_logits.shape[1] == i + self.cpc
            name = name_and_logits[:, :i] # (Even for digital cells, names are VAE-generated.)
            analog = name_and_logits[:, :j]
            with torch.no_grad():
                digprob = self.softmax(name_and_logits[:, i : i + self.cpc])
                indices = Sampler.nucleus_sampling(digprob).multinomial(1)
                bits = sn.Int.encode_ints(torch, indices, self.bpc)
                digital = torch.cat((name, bits, torch.zeros(name.shape[0], j - self.bpc - i)), -1)
            action_part = (is_ana * analog + (1-is_ana) * digital).clamp(-clamp, clamp)
            logits_part = is_ana * name_and_logits + (1-is_ana) * torch.cat((name, digprob), -1)
            self.fn(action_part, ... if latent is ... else latent[inds]) # Input what we sampled.
            action = action.index_put([inds], action_part)
            logits = logits.index_put([inds], logits_part)
        return action, logits
    def target(self, act: torch.Tensor, one_hot=1):
        """Converts a presumably-sampled `action` to its `logits`, which can be used to compute L2 loss."""
        # The representation: for analog, just the prediction as-is; for digital, name & the post-softmax probability distribution.
        #   (L2 loss on probabilities ends up copying the probabilities.)
        i, j, cells = self.start, self.end, act.shape[0]
        assert len(act.shape) == 2 and act.shape[1] == j
        act = detach(act)
        with torch.no_grad():
            is_ana = Sampler.analog_mask(act).float()
            analog = torch.cat((act, torch.zeros(cells, i+self.cpc - j)), -1)
            indices = sn.Int.decode_bits(torch, act[:, i : i + self.bpc]) % self.cpc
            digital = torch.cat((act[:, :i], F.one_hot(indices, self.cpc).float() * one_hot), -1)
            return is_ana * analog + (1-is_ana) * digital
    @staticmethod
    def goal_mask(frame):
        return frame[:, 0:1] > 0
    @staticmethod
    def analog_mask(frame):
        return frame[:, 1:2] > 0
    @staticmethod
    def sample_order(frame_names: np.ndarray):
        """Allows sampling `frame`s correctly, by iterating over these indices and sampling each index-group in sequence. In particular, analog cells are sampled all at once, then each goal group's digital cell is sampled autoregressively, because learning can't model joint probabilities otherwise, and because goal-groups are supposed to be independent and thus parallelizable."""
        analog_mask = Sampler.analog_mask(frame_names)[:, 0]
        yield analog_mask
        frame_names = frame_names[~analog_mask, :]
        groups = goal_groups(frame_names)
        if groups.shape[0]:
            inds = [same_goal_group(frame_names, g).nonzero()[0] for g in groups]
            for i in range(max(x.shape[0] for x in inds)):
                group_inds = np.stack([x[i] for x in inds if i < x.shape[0]])
                yield group_inds
    @staticmethod
    def nucleus_sampling(p: torch.Tensor):
        """Accept only the `top_p`-fraction of the probability mass: https://arxiv.org/abs/1904.09751"""
        if top_p == 1: return p
        p = detach(p.clone())
        vals, inds = p.sort(-1, descending=False)
        accepted_vals = vals.cumsum(-1) >= (1-top_p) - 1e-8
        return p.scatter_(-1, inds, accepted_vals.float() * p.gather(-1, inds))
    def use_digital_table(self, x: torch.Tensor):
        """Looks up the per-int vectors of digital-cells for RNN input, in accordance with `digital_embs`. These vectors may be easier for the RNN to learn to use than raw binary masks."""
        if digital_embs == 'no': return x
        assert len(x.shape) == 2
        is_ana = Sampler.analog_mask(detach(x)).float()
        bits = detach(x)[:, self.start : self.start + self.bpc]
        untabled = digital_table(sn.Int.decode_bits(torch, bits) % self.cpc)
        return is_ana * x + (1-is_ana) * torch.cat((x[:, :self.start], untabled), -1)



# Model.
class SkipConnection(nn.Module):
    """Linearize gradients, to make learning easier."""
    def __init__(self, *fn): super().__init__();  self.fn = nn.Sequential(*fn)
    def forward(self, x):
        y = self.fn(x)
        return y if x.shape[-1]<y.shape[-1] else x + y if x.shape == y.shape else x[..., :y.shape[-1]] + y
def h(ins = state_sz, outs = ...): # A cross-cell transform.
    if outs is ...: outs = ins
    return SkipConnection(
        ReLU(), LayerNorm(ins), SRWM(ins, ins, heads=2, Softmax=Softmax),
        ReLU(), LayerNorm(ins), nn.Linear(ins, outs),
    )
try:
    if not save_load: raise FileNotFoundError()
    transition, digital_table = torch.load(save_load)
except FileNotFoundError:
    transition, digital_table = nn.Sequential(
        # `(frame, latent) → (name_and_logits_or_values, latent_mean_and_var_log, dist, smudge)`
        #   `frame` is goals and prev-actions and next-observations.
        #   `latent` is for VAEs: sampled from a normal distribution with learned mean & variance-logarithm.
        nn.Linear(sum(cell_shape) + latent_sz, state_sz),
        h(state_sz, state_sz),
        h(state_sz, state_sz),
        h(state_sz, sum(cell_shape[:-1])+choices_per_cell + 2*latent_sz + 1 + 1),
    ), nn.Embedding(choices_per_cell, cell_shape[-1])
digital_table.requires_grad_(digital_embs == 'learn')
def transition_(x, latent=...):
    """Wraps `transition` to return a tuple of `(name_and_logits, latent_info, distance, smudge)`. `name_and_logits` is for `Sampler`, `latent_info` is for `normal` and `make_normal`."""
    if latent is ...:
        latent = torch.randn(x.shape[0], latent_sz)
    x = x.to(torch.float32)
    x = sample.use_digital_table(x)
    assert len(x.shape) == 2 and len(latent.shape) == 2 and x.shape[0] == latent.shape[0] and latent.shape[1] == latent_sz
    y = transition(torch.cat((x, latent), -1))
    lt = 2*latent_sz
    return y[:, :-(lt+2)], y[:, -(lt+2) : -2], 2 ** y[:, -2:-1], 2 ** y[:, -1:] - 1
dodge = DODGE(transition, lambda p: torch.optim.Adam(p, lr=lr)) # For forward-gradient-feedback potentially-very-long-unrolls RNN training.
sample = Sampler(transition_, choices_per_cell, sum(cell_shape[:-1]), sum(cell_shape))
# For debugging, print the param-count.
pc = sum(x.numel() for x in transition.parameters())
print(str(pc/1e9)+'G' if pc>1e9 else str(pc/1e6)+'M' if pc>1e6 else str(pc/1e3)+'K' if pc>1e3 else pc, 'params')





# Our interface to multigroup combined-cells goals (OR/AND goals): the last name part is `group_id`, AKA user ID.
#   Via careful engineering here, users can have entirely separate threads of experience that reach any goals in their group (but the NN can of course learn to share data between its threads).
def goal_groups(frame: np.ndarray) -> np.ndarray:
    """Returns all goal-group IDs contained in the `frame` (a 2D NumPy array), each a byte-objects, suitable for indexing a dictionary with."""
    return np.unique(frame[:, sum(cell_shape[:-2]) : sum(cell_shape[:-1])], axis=0)
def same_goal_group(frame, group):
    """Returns a bitmask shaped as `(cells,)`."""
    return (frame[:, sum(cell_shape[:-2]) : sum(cell_shape[:-1])] == group).all(-1)
def local_dist(base: torch.Tensor, goal: torch.Tensor, group) -> torch.Tensor:
    """`local_dist(base, goal, group) → smudge`: a single goal's smudging, AKA final local-distance, AKA how close we've come to the goal.

    Quadratic time complexity."""
    assert len(base.shape) == len(goal.shape) == 2 and base.shape[-1] == goal.shape[-1] == sn.cell_size
    if not base.numel(): # We have nothing that could have reached the goal.
        return torch.ones(()) * goal.numel()
    if not goal.numel(): # If no goal-cells, then we've already reached the goal.
        return torch.zeros(())
    base, goal = detach(base), detach(goal)
    max_smudge = sn.cell_size
    with torch.no_grad():
        same_group = torch.unsqueeze(same_goal_group(base, group), -1)
        base_logits, goal_logits = Sampler.target(base, max_smudge), Sampler.target(goal, max_smudge)
        cross_smudges = (.5*(base_logits.unsqueeze(-3) - goal_logits.unsqueeze(-2))).clamp(0., 1.).abs().sum(-1)
        return torch.where(same_group, cross_smudges, max_smudge).min(-2)[0].mean(-1)
def global_dists(smudges: torch.Tensor, dists_pred: torch.Tensor, smudges_pred: torch.Tensor):
    """`global_dists(smudges, dists_pred, smudges_pred) → (smudge, dists)`

    On a path to a goal, the min local-dist (`smudge`) measures how close we can come. `dists` will count out the steps to reach it, to be used as prediction targets."""
    assert smudges.shape == dists_pred.shape == smudges_pred.shape
    smudges, dists_pred, smudges_pred = detach(smudges), detach(dists_pred), detach(smudges_pred)
    with torch.no_grad():
        are_we_claiming_to_have_arrived = (dists_pred < smudges.shape[-1]).any(-1, keepdim=True)
        smudge = smudges.min(-1, keepdim=True)[0]
        smudge = smudge.min(smudges_pred[..., -1:] + (~are_we_claiming_to_have_arrived).float())
        reached = smudges <= smudge+1
        next_of_reached = torch.cat((torch.zeros(*reached.shape[:-1], 1), reached[..., :-1].float()), -1)
        dists = dists_pred[..., -1:] + torch.arange(1, reached.shape[-1]+1) # Count-out.
        left_scan = ((dists - 1) * next_of_reached).cummax(-1)[0]
        dists = (dists - left_scan).flip(-1)
        return (smudge, dists)
def cells_override(based_on, goal_mask, analog_mask, goal_group):
    """`cells_override(based_on, analog_mask, goal_group) → based_on_2`

    Overrides the data needed by goal-cells, namely, goal/analog bits and the goal-group. `based_on` is likely all-0s."""
    assert len(based_on.shape) == len(goal_mask.shape) == len(analog_mask.shape) == len(goal_group.shape) == 2
    assert goal_mask.shape[1] == analog_mask.shape[1] == 1
    start, end = sum(cell_shape[:-2]), sum(cell_shape[:-1])
    goalness = goal_mask.float() if isinstance(goal_mask, torch.Tensor) else goal_mask.astype(np.float32)
    analogness = analog_mask.float() if isinstance(analog_mask, torch.Tensor) else analog_mask.astype(np.float32)
    tpl = (goalness*2-1, analogness*2-1, based_on[:, 2:start], goal_group, based_on[:, end:])
    return torch.cat(tpl, -1) if isinstance(based_on, torch.Tensor) else np.concatenate(tpl, -1)
def sample_goal(cells, goal_group, analog_prob=.5):
    """Samples a new goal. Advances RNN state."""
    start, end = sum(cell_shape[:-2]), sum(cell_shape[:-1])
    goal_mask_np = np.full((cells, 1), True)
    analog_mask_np = np.random.rand(cells, 1) < analog_prob
    goal_group_np = np.resize(goal_group, (cells, end-start))
    z_np = np.zeros((cells, sn.cell_size), dtype=np.float32)
    z_np = cells_override(z_np, goal_mask_np, analog_mask_np, goal_group_np)
    goal_mask = torch.full((cells, 1), True)
    analog_mask = torch.as_tensor(analog_mask_np)
    goal_group = torch.as_tensor(goal_group).expand(cells, end-start)
    z = torch.zeros(cells, sn.cell_size)
    z = cells_override(z, goal_mask, analog_mask, goal_group)
    z = sample(z, z_np)[0].clamp(-clamp, clamp)
    return cells_override(z, goal_mask, analog_mask, goal_group), z_np[:, :end]





# Forward-grad compatibility for older PyTorch versions.
class Index(torch.autograd.Function):
    """`x[inds]`"""
    @staticmethod
    def forward(ctx, x, inds):
        ctx.save_for_backward(inds)
        return x[inds]
    @staticmethod
    def jvp(ctx, dx, _):
        inds, = ctx.to_save # Docs weren't clear whether this is the intended way.
        return dx[inds]





# Loss.
def cell_dropout(frame: torch.Tensor, frame_names: np.ndarray):
    """Makes querying subsets of cells not out-of-distribution, by dropping & shuffling cells randomly at training-time. Call & destructure before `per_goal_loss`."""
    inds = np.random.choice(np.arange(0, frame.shape[0]), random.randint(1, frame.shape[0]), replace=False)
    return Index.apply(frame, torch.as_tensor(inds).long()), frame_names[inds]
def augment(frame: torch.Tensor) -> torch.Tensor:
    """Augments a frame, so that VAEs can be trained to minimize both dependence on input and cross-input dependence, and so that we can sample even without knowing queries (for imagined trajectories).

    In particular, this randomly decides whether to make cells name-only or goal-group-only."""
    frame = detach(frame)
    cells = frame.shape[0]
    start, end = sum(cell_shape[:-2]), sum(cell_shape[:-1])
    is_name = torch.rand(cells, 1) * 2 < 1
    name_only = torch.cat((frame[:, :-cell_shape[-1]], torch.zeros(cells, cell_shape[-1])), -1)
    group_only = cells_override(torch.zeros_like(frame), Sampler.goal_mask(frame), Sampler.analog_mask(frame), frame[:, start:end])
    return torch.where(is_name, name_only, group_only)
def fill_in(target, target_names):
    """VAE integration: encode `target` and decode an alternative to it. If called in pre-`target` RNN state, produces target's info and alternative-target's info.

    Result: `((target_dist, target_smudge, target, target_logits), (pred_dist, pred_smudge, pred, pred_logits), regularization_loss)`"""
    assert len(target.shape) == 2
    with State.Episode(start_from_initial=False): # Decoder & distance.
        _, target_latent_info, target_dist, target_smudge = transition_(target)
        target_logits = sample.target(target)
        target_latent = normal(target_latent_info)
        regularization_loss = make_normal(target_latent_info).sum(-1, keepdim=True)
    pred, pred_logits = sample(augment(target), target_names, target_latent) # Encoder.
    with State.Episode(start_from_initial=False): # Distance.
        _, _, pred_dist, pred_smudge = transition_(pred)
    return ((target_dist, target_smudge, target, target_logits), (pred_dist, pred_smudge, pred, pred_logits), regularization_loss)

def local_loss(frame, frame_names):
    """Does gated autoencoding, always improving goal-direction actions and goal-suggestions. Advances RNN state; needs to be called in the RNN-context just before `frame`. Returns `(loss, dist_preds, smudge_preds)`; use `global_loss` later to learn the gating.

    (Due to our "non-goals have lower dist, goals have different dist" gating, [generated](http://proceedings.mlr.press/v100/nair20a/nair20a.pdf) goals are neither too hard (AKA when never occuring as real targets) nor too easy (AKA when we know what we can do and then reliably do it, with no dist-error).) (Related to: [AMIGo](https://arxiv.org/pdf/2006.12122.pdf), [AdaGoal](https://arxiv.org/pdf/2111.12045.pdf). Fits tighter than either.)"""
    # Contrast `frame` with our own understanding of it.
    (frame_dist, frame_smudge, frame, frame_logits), (pred_dist, pred_smudge, pred, pred_logits), reg_loss = fill_in(frame, frame_names)

    # Compute what we learn to imitate, namely, reached-goal cells for which:
    #   If non-goal cells: distance decreases.
    #   If goal cells: distance misprediction increases.
    min_dist_misprediction = 2
    eventually_closer = frame_smudge <= pred_smudge+1
    is_goal = Sampler.goal_mask(frame)
    dist_is_better = torch.where(is_goal, frame_dist <= pred_dist, (frame_dist - pred_dist).abs() >= min_dist_misprediction)
    imitate = (eventually_closer & dist_is_better).float()

    # Self-imitation (critic-regularized regression) (gated autoencoding).
    reg_loss = (imitate * reg_loss).sum()
    predict_loss = (imitate * (frame_logits - pred_logits).square()).sum()

    # Log and return.
    log_metrics(imitated=imitate.mean(), reg_loss=reg_loss, predict_loss=predict_loss)
    return reg_loss + predict_loss, frame_dist, frame_smudge

def per_goal_loss(frame, frame_names, goals):
    """`per_goal_loss(frame, frame_names, goals) → (loss, smudges, dist_preds, smudge_preds)`

    Wraps `local_loss` to compute all *per-goal* local metrics, which can later be `torch.stack`ed and put into `global_dists` and then into `global_loss`."""
    # Compute per-group means of predictions, then predict means and return means (which would be subject to `global_loss`).
    #   Also compute `local_dist`s.
    loss, dist_pred, smudge_pred = local_loss(frame, frame_names)
    smudges, dist_preds, smudge_preds,  = [], [], [], 
    for V in goals.values():
        group, goal = V[0], V[1]
        same_group = (frame[:, sum(cell_shape[:-2]) : sum(cell_shape[:-1])] == group).all(-1, keepdim=True).float()
        cell_count = same_group.sum()
        mean_dist_pred = (same_group * detach(dist_pred)).sum() / cell_count
        mean_smudge_pred = (same_group * detach(smudge_pred)).sum() / cell_count
        dist_preds.append(mean_dist_pred)
        smudge_preds.append(mean_smudge_pred)
        loss = loss + (same_group * (dist_pred - mean_dist_pred).square()).sum()
        loss = loss + (same_group * (smudge_pred - mean_smudge_pred).square()).sum()
        smudges.append(local_dist(frame, goal, group)) # (`same_group` is recomputed inside.)
    return loss, smudges, dist_preds, smudge_preds

def global_loss(dist_pred, smudge_pred, dist_target, smudge_target):
    """Learns what to gate the autoencoding by. Returns the loss."""
    dist_target = detach(dist_target.min(dist_pred + optimism) if optimism is not None else dist_target)
    smudge_target = detach(smudge_target.min(smudge_pred + optimism) if optimism is not None else smudge_target)
    dist_loss = (dist_pred.log2() - dist_target.log2()).square().sum()
    smudge_loss = ((smudge_pred+1).log2() - (smudge_target+1).log2()).square().sum()
    log_metrics(dist_loss=dist_loss, smudge_loss=smudge_loss)
    return dist_loss + smudge_loss

def log_metrics(**kw):
    """Called with args, this adds to a dict of what to log; called without args, this logs."""
    if not hasattr(log_metrics, 'metrics'):
        log_metrics.metrics = {}
    if kw: # Add if specified.
        for k,v in kw.items():
            log_metrics.metrics[k] = detach(v)
    else: # Log if no-args.
        if logging: log(0, False, torch, **log_metrics.metrics)
        log_metrics.metrics.clear()
        n = 1
        if logging:
            for name, env in envs.items():
                if hasattr(env, 'metric'): # The list of envs for which this is true shouldn't ever change.
                    log(n, False, torch, **{name+'.'+k: v for k,v in env.metric().items()})
                    n += 1





# The main loop.
@sn.run
async def unroll():
    direction = None
    loss_so_far = 0.
    avg_time_to_goal, avg_time_to_goal_momentum = 0., .99
    def update_direction():
        nonlocal direction, loss_so_far
        p = dodge_optimizes_params / dodge.sz
        direction = torch.where(torch.rand(dodge.sz) < p, torch.randn(dodge.sz), torch.zeros(dodge.sz))
        if loss_so_far != 0.: dodge.minimize(loss_so_far)
        dodge.restart(direction)
        loss_so_far = 0.
    def finish_computing_loss(smudge, dist_pred, smudge_pred, start_time=None):
        if not len(smudge): return
        smudges = torch.stack(smudge)
        dist_preds = torch.stack(dist_pred)
        smudge_preds = torch.stack(smudge_pred)
        smudge.clear();  dist_pred.clear();  smudge_pred.clear()
        smudge_target, dist_target = global_dists(smudges, dist_preds, smudge_preds)
        loss_so_far = loss_so_far + global_loss(dist_pred, smudge_pred, dist_target, smudge_target)
        if start_time is not None:
            avg_time_to_goal = avg_time_to_goal_momentum*avg_time_to_goal + (1-avg_time_to_goal_momentum) * (time - start_time)
    with fw.dual_level():
        with State.Setter(lambda state, to: state.initial*.001 + .999*to): # Soft-reset.
            update_direction()
            with State.Episode() as life:
                with torch.no_grad(): # TODO: This shouldn't apply to the delayed synth-grad-aware backprop-train. (After we have that, I mean.)
                    prev_q, action, frame = None, None, None
                    goals = {} # goal_group_id → (group, goal, goal_names, expiration_time_cpu, expiration_time_gpu, smudges, dist_preds, smudge_preds, start_time)
                    time = 0 # If this gets unreasonably high, goal-switching will have problems.
                    while True:
                        if slow_mode > 0:
                            await asyncio.sleep(slow_mode)

                        obs_np, query_np, error_np = await sn.handle(sn.torch(torch, action))
                        obs_np   = np.nan_to_num(  obs_np.clip(-clamp, clamp), copy=False)
                        query_np = np.nan_to_num(query_np.clip(-clamp, clamp), copy=False)
                        #   (If getting out-of-memory, might want to chunk data/query processing.)

                        # Bookkeeping for later.
                        obs, query = torch.tensor(obs_np), torch.tensor(query_np)
                        frame = torch.cat((action, obs), 0) if action is not None else obs
                        frame_names = np.concatenate((prev_q, obs[:, :prev_q.shape[1]]), 0) if prev_q is not None else obs_np[:, :query.shape[1]]
                        prev_q = query

                        # Delete/update our `goals` when we think we've reached them.
                        #   When deleting a goal, minimize its `global_loss`.
                        for hash, (_, _, _, expiration_cpu, _, smudges, dist_preds, smudge_preds, start_time) in goals.copy().items():
                            if time > expiration_cpu:
                                finish_computing_loss(smudges, dist_preds, smudge_preds, start_time)
                                del goals[hash]
                        for group in goal_groups(frame_names):
                            hash = group.tobytes()
                            if hash not in goals:
                                with State.Episode(start_from_initial=False):
                                    goal, goal_names = sample_goal(random.randint(1, obs.shape[0] or 1), group)
                                expiration_cpu = torch.tensor(time+10000, device='cpu', dtype=torch.int64)
                                expiration_gpu = torch.tensor(time+10000, dtype=torch.int64)
                                smudges, dist_preds, smudge_preds = [], [], []
                                goals[hash] = (group, goal, goal_names, expiration_cpu, expiration_gpu, smudges, dist_preds, smudge_preds, time)
                        # Expose self-generated goals to the RNN.
                        if goals:
                            extra_cells = [goal for (_, goal, _, _, _, _, _, _, _) in goals.values()]
                            extra_names = [goal_names for (_, _, goal_names, _, _, _, _, _, _) in goals.values()]
                            frame = torch.cat([*extra_cells, frame], 0)
                            frame_names = np.concatenate([*extra_names, frame_names], 0)
                        # Delimit DODGE-episode boundaries, changing its projection direction sometimes.
                        if dodge_len * random.random() < 1 / (avg_time_to_goal+.5): # (Close enough to what `dodge_len` says.)
                            for _, _, _, _, _, smudges, dist_preds, smudge_preds, start_time in goals.values():
                                finish_computing_loss(smudges, dist_preds, smudge_preds)
                            update_direction()

                        # Learn to predict better, via `per_goal_loss`.
                        #   TODO: …This can actually do normal backprop at the same time, right? Even synthetic gradients… Just do `loss.backward()` (without optimizer-steps) when all params are .requires_grad_(True) and will give correct gradient…
                        with State.Episode(start_from_initial=False):
                            loss, smudge, dist_pred, smudge_pred = per_goal_loss(*cell_dropout(frame, frame_names), goals)
                            loss_so_far = loss_so_far + loss
                            for _, _, _, _, _, smudges, dist_preds, smudge_preds, _ in goals.values(): # For `global_loss`.
                                smudges.append(smudge)
                                dist_preds.append(dist_pred)
                                smudge_preds.append(smudge_pred)
                        log_metrics()

                        # Give prev-action & next-observation, remember dist-to-goal estimates, and sample next action.
                        _, _, dist, _ = transition_(frame)
                        n = 0
                        for group, (group, goal, _, expiration_cpu, expiration_gpu, _, _, _, _) in goals.items():
                            # (If we do better than expected, we leave early.)
                            if goal.shape[0] > 0:
                                group_dist = (time + dist.detach()[n : n+goal.shape[0], -1].mean() + 1).int().min(expiration_gpu)
                                expiration_cpu.copy_(group_dist, non_blocking=True)
                                expiration_gpu.copy_(group_dist)
                            else:
                                expiration_cpu.fill_(0)
                            n += goal.shape[0]
                        with State.Episode(start_from_initial=False):
                            zeros = torch.zeros(query.shape[0], cell_shape[-1])
                            action, _ = sample(torch.cat((query, zeros), -1), query_np)
                            #   (Can also do safe exploration / simple planning, by `sample`ing several actions and only using the lowest-dist-sum (inside `with State.Episode(False): ...`) of those.)
                            #     (Could even plot the regret of sampling-one vs sampling-many, by generating N and having a regret-plot for each (should go to 0 with sufficiently-big N), and see if/when it's worth it.)

                        # Save. (Both replay-buffer and disk.)
                        if save_load and time % steps_per_save == 0:
                            torch.save((transition, digital_table), save_load)

                        time += 1