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















# TODO: Make `unroll` use `loss, *metrics = per_goal_loss(…)` and `dodge.minimize(loss)` and put metrics into a list.
#   TODO: Whenever we change a goal in `unroll`, empty that list: compute all `global_dists`, compute `L = global_loss(…)`, and `dodge.minimize(L)`.
# TODO: When sampling an autoencoding prediction (in `fill_in`), goal-cells should start from all-0-plus-`goal_cells_override`, whereas others should start from name-only. (Though sometimes, *might* want to start from full data?… …And *maybe*, sometimes, might want to even zero out the names, in case we want to generate them too?…)



# TODO: Make docs of `sn.Int` mention its alternative/`_default_typing` forms.
# TODO: Make `sn` unaware of discrete-sampling difficulties, because the models handle them instead.
#   TODO: `Sampler`:
#     TODO: NumPy-detect analog & digital cells. First generate all analog cells in parallel, then generate all digital cells autoregressively. (And make `__call__` able to be conditioned on a `target`, instead of on-policy outputs.) (And have `.loss(…)` that does target-conditioned generation and prediction of `.target(…)`.) (And make `local_loss` and `fill_in` make sampling target-aware.)
#     TODO: Have a hyperparam that makes digital-generation detect distinct goal-groups and generate digital-cells in parallel for distinct groups. For speed.
#   TODO: Make `sn.Int` put everything in at the same time, nothing autoregressive. (Big-int goals will then be possible.)
#   TODO: Make `sn.Handler.pipe` not have any mechanism for spreading out requests along multiple timesteps.



# TODO: …What about that random-cell-dropout that makes `action`-sampling in-distribution (I think we need it); and, should we maybe also make `local_loss` randomly decide to leave in full observations instead of zeroing them out, so that we can unite queries & observations & goals into one big request at unroll-time?…



# TODO: Should we generate a few random-latent `src` cells every step with a special input structure (1s in the first number and 0s everywhere else; queries & setting are possible), and predict/autoencode themselves whenever the path has contained any dist-prediction errors (so, same as `dst`'s gating, but for the past instead for the future)? So that `dst` can strive toward non-input unreliably-reachable outcomes?
#   TODO: NEED an env that consists of many buttons, which have to be pushed in a password-sequence that changes rarely, with a bit-indicator for "was the last button correct". With enough buttons, state-based exploration should discover nothing, whereas proper full-past exploration *should* discover the sequence eventually.
#   TODO: POSSIBLY WANT `dst`-generation to be conditioned on whether it can only use `src`-cells, and compute separate dists & smudges for `src`-only learning, and have a hyperparam that makes only `src` cells generatable (envs can still specify input-space goals).
#     TODO: If so, WANT an env that can only be solved with good exploration (a big maze with resetting), to give us the success criterion: the env works both with and without input-space goals.






# TODO: …Maybe, have per-SRWM-matrix synthetic gradients via making the synth-grad NNs (possibly even the main RNN) learning to output key & value vectors for each matrix (possibly more than 1 pair), the outer product of which is taken to be the gradient?…
#   …Maybe instead of BPTT with its periodic-stalling, we should do backprop-less DODGE online (even with `dodge_optimizes_params=1`, it should help quite a bit) with ≈1000-step-delayed one-step synth-grad learning following at a distance… Disabling DODGE before that and re-enabling it with the old direction afterward… Possibly even with Reptile or MAML, though [it may be superfluous with enough layers](https://arxiv.org/abs/2106.09017)…

# TODO: Do dropout of frame-cells when doing synth-grad replays (forward-unroll with DODGE can see all). (This will make sampling actions-without-next-frame at unroll-time not out-of-distribution, *and* make the net try to minimize dependencies on inputs.)






# TODO: …May want a hyperparam for what to do with digital inputs: leave as binary encodings, VS lookup in a learnable table, VS lookup in a fixed random table…






# TODO: …Maybe also add support for nucleus sampling (like zeroing out all probabilities that are less than 1/40 and re-normalizing), so that we don't very-rarely sample overly-low-probability discrete actions?…





# TODO: Also a hyperparam for whether learned-distances are expected values (AKA averages or medians) or made-optimistic.






# TODO: Run & fix the copy-task in `env/copy.py`, to test our implementation.

# TODO: An env that has both analog actions and analog goals: make the goal a random image (plus the "at the final-state" bit), then get like 3 images (exposing "NOT at the final-state"), then expose the sum of those actions and the fact that this is the final-state.
#   TODO: Also an env for out-of-distribution goals: expose a reward that caps off at 0 (reward of 'how close the image is to the non-goal target image' maybe), and always say that it must be 1.

# TODO: Might want to do the simplest meta-RL env like in https://openreview.net/pdf?id=TuK6agbdt27 to make goal-generation much easier and make goal-reachability tracked — with a set of pre-generated graphs to test generalization…

# TODO: Make `graphenv` work not globally but in a class.





# TODO: Maybe, allow `sn.Int`s to have only `1` option, so that it's easy for users to dispatch "events" which can be set as goals. Maybe also turn no-data no-type `.set` calls into such events.
# TODO: Maybe, if `choices_per_cell` is not defined in `sn.info`, it should be treated as `sn.cell_shape[-1]`, and `sn.Int`'s query should get argmax — possibly set as one-hot encodings too… (If ints are not explicitly supported, provide at least some inefficient support of them.) (And, `sn.info['analog']` checks would probably be unnecessary with this.)
# TODO: Also support [mu-law-encoded](https://en.wikipedia.org/wiki/%CE%9C-law_algorithm) (and/or linearly-encoded) floats-in-ints. `IntFloat(*shape, opts=256, mu=mu, bounds=(min,max))`. `mu=0` should be special-cased to be linear.
# TODO: Also support fixed-size strings (with tokenizers) and image-patches. (The most important 'convenience' datatypes.)
# TODO: Maybe, have `.metrics()` on handlers, and have two metrics: cells-per-second (exponentially-moving average) (which doesn't count the time spent on waiting for data) and latency (EMA too) (time from a `.handle` call to when its `feedback` is actually available to us, in seconds).




# TODO: …Do we maybe also want a hyperparam for "how many actions to pick between at unroll-time"?… (…Which we can *technically* allow the model to modify, for better or worse — same as learning-rate and replays-per-step and max-replay-buffer-len… Though maybe it's better to just learn an "expected improvement with this action-count" NN/table.)













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
latent_sz = 16 # For VAE generation.
slow_mode = .05 # Artificial delay per step.

choices_per_cell = 256
clamp = 1. # All inputs/outputs are clipped to `-clamp…clamp`.

lr = 1e-3
dodge_optimizes_params = 1000 # `DODGE` performs better with small direction-vectors.
optimism = .5 # If None, average of dist&smudging is learned; if >=0, pred-targets are clamped to be at most pred-plus-`optimism`.

logging = True
save_load = '' # A filename, if saving/loading occurs.
steps_per_save = 1000

sn.info = {
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
    def __call__(self, query, latent=...):
        """Given a name-only `query` (and possibly `latent`), samples a binary action.

        The result is `(action, logits)`. Use `action` to act; use L2 loss on `logits` and `.target(action)` to predict the action.

        This will advance RNN `State`, so do `State.Episode` shenanigans if that is unwanted."""
        is_ana = self.analog_mask(detach(query)).float()
        i, j, cells = self.start, self.end, query.shape[0]
        input = torch.cat((query, torch.zeros(cells, j - i)), -1)
        name_and_logits = self.fn(input, latent)[0]
        assert len(name_and_logits.shape) == 2 and name_and_logits.shape[1] == i + self.cpc
        name = name_and_logits[:, :i] # (Even for digital cells, names are VAE-generated.)
        analog = name_and_logits[:, :j]
        with torch.no_grad():
            digprob = self.softmax(detach(name_and_logits[:, i : i + self.cpc]))
            indices = digprob.multinomial(1)[..., 0]
            bits = sn.Int.encode_ints(sn, indices, self.bpc)
            digital = torch.cat((name, bits, torch.zeros(cells, j - self.bpc - i)), -1)
        action = is_ana * analog + (1-is_ana) * digital
        logits = is_ana * name_and_logits + (1-is_ana) * torch.cat((name, digprob), -1)
        return action, logits
    def target(self, act, one_hot=1):
        """Converts a presumably-sampled `action` to its `logits`, which can be used to compute L2 loss."""
        # The representation: for analog, just the prediction as-is; for digital, name & the post-softmax probability distribution.
        #   (L2 loss on probabilities ends up copying the probabilities.)
        i, j, cells = self.start, self.end, act.shape[0]
        assert len(act.shape) == 2 and act.shape[1] == j
        act = detach(act)
        with torch.no_grad():
            is_ana = self.analog_mask(act).float()
            analog = torch.cat((act, torch.zeros(cells, self.cpc - j)), -1)
            indices = sn.Int.decode_bits(sn, act[:, i : i + self.bpc]) % self.cpc
            digital = F.one_hot(indices, self.cpc).float() * one_hot
            return is_ana * analog + (1-is_ana) * digital
    @staticmethod
    def goal_mask(frame):
        return frame[:, 0:1] > 0
    @staticmethod
    def analog_mask(frame):
        return frame[:, 1:2] > 0



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
    transition = torch.load(save_load)
except FileNotFoundError:
    transition = nn.Sequential(
        # `(frame, latent) → (name_and_logits_or_values, latent_mean_and_var_log, dist, smudge)`
        #   `frame` is goals and prev-actions and next-observations.
        #   `latent` is for VAEs: sampled from a normal distribution with learned mean & variance-logarithm.
        nn.Linear(sum(cell_shape) + latent_sz, state_sz),
        h(state_sz, state_sz),
        h(state_sz, state_sz),
        h(state_sz, sum(cell_shape[:-1])+choices_per_cell + 2*latent_sz + 1 + 1),
    )
def transition_(x, latent=...):
    """Wraps `transition` to return a tuple of `(name_and_logits, latent_info, distance, smudge)`. `name_and_logits` is for `Sampler`, `latent_info` is for `normal` and `make_normal`."""
    if latent is ...:
        latent = torch.randn(x.shape[0], latent_sz)
    assert len(x.shape) == 2 and len(latent.shape) == 2 and x.shape[0] == latent.shape[0] and latent.shape[1] == latent_sz
    y = transition(torch.cat((x, latent), -1))
    lt = 2*latent_sz
    return y[:, :-(lt+1)], y[:, -(lt+1) : -2], 2 ** y[:, -2:-1], 2 ** y[:, -1:] - 1
dodge = DODGE(transition) # For forward-gradient-feedback potentially-very-long-unrolls RNN training.
optim = torch.optim.Adam(transition.parameters(), lr=lr)
sample = Sampler(transition_, choices_per_cell, sum(cell_shape[:-1]), sum(cell_shape))
# For debugging, print the param-count.
pc = sum(x.numel() for x in transition.parameters())
print(pc/1000000000+'B' if pc>1000000000 else pc/1000000+'M' if pc>1000000 else pc/1000+'K' if pc>1000 else pc, 'params')





# Our interface to multigroup combined-cells goals (OR/AND goals): the last name part is `group_id`, AKA user ID.
#   Via careful engineering here, users can have entirely separate threads of experience that reach any goals in their group (but the NN can of course learn to share data between its threads).
def goal_groups(frame):
    """Returns the `set` of all goal-group IDs contained in the `frame` (a 2D NumPy array), each a byte-objects, suitable for indexing a dictionary with."""
    return np.unique(frame[:, sum(cell_shape[:-2]) : sum(cell_shape[:-1])], axis=0)
# Computable-on-unroll distances.
def local_dists(base: torch.Tensor, goals) -> torch.Tensor:
    """`local_dists(base, goals) → smudges`

    Given sets of cells of observations & goals, computes per-goal-group local-dists (smudging) between them, which is 0 when the `goal` is 100% reached and more when not. (Learned pathfinding should select min-dist actions among min-smudging paths, so that even unreachable goals can have best-effort paths known instead of being out-of-distribution.)"""
    smudges = [local_dist(base, goal, group) for (group, goal, _, _) in goals.values()]
    return torch.stack(smudges) if len(smudges) else torch.tensor([])
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
        same_group = (base[:, sum(cell_shape[:-2]) : sum(cell_shape[:-1])] == group).all(-1, keepdim=True)
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
def goal_cells_override(based_on: torch.Tensor, analog_mask: torch.Tensor, goal_group):
    """`goal_cells_override(based_on, analog_mask, goal_group) → based_on_2`

    Overrides the data needed by goal-cells, namely, goal/analog bits and the goal-group."""
    assert len(based_on.shape) == len(analog_mask.shape) == 2 and analog_mask.shape[1] == 1
    cells = analog_mask.shape[0]
    start, end = sum(cell_shape[:-2]), sum(cell_shape[:-1])
    goalness = torch.ones(cells, 1)
    analogness = analog_mask.float()*2-1
    goal_group = torch.as_tensor(goal_group).expand(cells, end-start)
    return torch.cat((goalness, analogness, based_on[2:start], goal_group, based_on[end:]), -1)
def sample_goal(cells, goal_group, analog_prob=.5):
    """Samples a new goal. Advances RNN state."""
    analog_mask = torch.rand(cells, 1) < analog_prob
    z = torch.zeros(cells, sn.cell_size)
    z = goal_cells_override(z, analog_mask, goal_group)
    z = sample(z).clamp(-clamp, clamp)
    return goal_cells_override(z, analog_mask, goal_group)





# Loss.
def fill_in(target):
    """VAE integration: encode `target` and decode an alternative to it. If called in pre-`target` RNN state, produces target's info and alternative-target's info.

    Result: `((target_dist, target_smudge, target, target_logits), (pred_dist, pred_smudge, pred, pred_logits), regularization_loss)`"""
    assert len(target.shape) == 2
    with State.Episode(start_from_initial=False): # Decoder & distance.
        _, target_latent_info, target_dist, target_smudge = transition_(target)
        target_logits = sample.target(target)
        target_latent = normal(target_latent_info)
        regularization_loss = make_normal(target_latent_info).sum(-1, keepdim=True)
    name_sz = sn.cell_size - cell_shape[-1]
    pred, pred_logits = sample(target[:, : name_sz], target_latent) # Encoder.
    with State.Episode(start_from_initial=False): # Distance.
        _, _, pred_dist, pred_smudge = transition_(pred)
    return ((target_dist, target_smudge, target, target_logits), (pred_dist, pred_smudge, pred, pred_logits), regularization_loss)

def local_loss(frame):
    """Does gated autoencoding, always improving goal-direction actions and goal-suggestions. Advances RNN state; needs to be called in the RNN-context just before `frame`. Returns `(loss, dist_preds, smudge_preds)`; use `global_loss` later to learn the gating.

    (Due to our "non-goals have lower dist, goals have different dist" gating, [generated](http://proceedings.mlr.press/v100/nair20a/nair20a.pdf) goals are neither too hard (AKA when never occuring as real targets) nor too easy (AKA when we know what we can do and then reliably do it, with no dist-error).) (Related to: [AMIGo](https://arxiv.org/pdf/2006.12122.pdf), [AdaGoal](https://arxiv.org/pdf/2111.12045.pdf). Fits tighter than either.)"""
    # Contrast `frame` with our own understanding of it.
    (frame_dist, frame_smudge, frame, frame_logits), (pred_dist, pred_smudge, pred, pred_logits), reg_loss = fill_in(frame)
    # TODO: …Wait: where do we zero-fill the values of `frame`? `sample` in `fill_in`? Really? Is that really a good idea, considering that we need to make goal-cells start from overriden zeros?

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

def per_goal_loss(frame, goals):
    """`per_goal_loss(frame, goals) → (loss, smudges, pred_dist, pred_smudge)`

    Wraps `local_loss` to compute all *per-goal* local metrics, which can later be `torch.stack`ed and put into `global_dists` and then into `global_loss`."""
    # Compute per-group means of predictions, then predict means and return means (which would be subject to `global_loss`).
    #   Also compute `local_dist`s.
    loss, dist_pred, smudge_pred = local_loss(frame)
    smudges, dist_preds, smudge_preds,  = [], [], [], 
    for group, goal, _, _ in goals:
        same_group = (frame[:, sum(cell_shape[:-2]) : sum(cell_shape[:-1])] == group).all(-1, keepdim=True).float()
        cell_count = same_group.sum()
        mean_dist_pred = (same_group * detach(dist_pred)).sum() / cell_count
        mean_smudge_pred = (same_group * detach(smudge_pred)).sum() / cell_count
        dist_preds.append(mean_dist_pred)
        smudge_preds.append(mean_smudge_pred)
        loss = loss + (same_group * (dist_pred - mean_dist_pred).square()).sum()
        loss = loss + (same_group * (smudge_pred - mean_smudge_pred).square()).sum()
        smudges.append(local_dist(frame, goal, group)) # (`same_group` is recomputed inside.)
        # TODO: Remove `local_dists` since this is all that it does, and we're doing its job for it.
    if goals:
        return loss, torch.stack(smudges), torch.stack(dist_preds), torch.stack(smudge_preds)
    else:
        return loss, torch.tensor(()), torch.tensor(()), torch.tensor(())

def global_loss(pred_dist, pred_smudge, target_dist, target_smudge): # TODO: Swap word-order to be consistent with `global_dists`.
    """Learns what to gate the autoencoding by. Returns the loss."""
    target_dist = detach(target_dist.min(pred_dist + optimism) if optimism is not None else target_dist)
    target_smudge = detach(target_smudge.min(pred_smudge + optimism) if optimism is not None else target_smudge)
    #   TODO: Swap word-order to be consistent with `global_dists`.
    dist_loss = (pred_dist.log2() - target_dist.log2()).square().sum()
    smudge_loss = ((pred_smudge+1).log2() - (target_smudge+1).log2()).square().sum()
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
    def update_direction():
        nonlocal direction
        direction = torch.where(torch.rand(dodge.sz) < dodge_optimizes_params / dodge.sz, torch.randn(dodge.sz), torch.zeros(dodge.sz))
        dodge.restart(direction)
    with fw.dual_level():
        with State.Setter(lambda state, to: state.initial*.001 + .999*to): # Soft-reset.
            update_direction()
            with State.Episode() as life:
                with torch.no_grad(): # TODO: This shouldn't apply to the delayed synth-grad-aware backprop-train.
                    prev_q, action, frame = None, None, None
                    goals = {} # goal_group_id → (group, goal, expiration_time_cpu, expiration_time_gpu)
                    time = 0 # Hopefully this doesn't get above 2**31-1.
                    while True:
                        if slow_mode > 0:
                            await asyncio.sleep(slow_mode)

                        obs, query, error = await sn.handle(sn.torch(torch, action))
                        obs   = np.nan_to_num(  obs.clip(-clamp, clamp), copy=False)
                        query = np.nan_to_num(query.clip(-clamp, clamp), copy=False)
                        #   (If getting out-of-memory, might want to chunk data/query processing.)

                        # Bookkeeping for later.
                        obs, query = torch.tensor(obs), torch.tensor(query)
                        frame = torch.cat((action, obs), 0) if action is not None else obs
                        frame_names = np.concatenate((prev_q, obs[:, :prev_q.shape[1]]), 0) if prev_q is not None else obs[:, :query.shape[1]]
                        prev_q = query

                        # Delete/update our `goals` when we think we reached them.
                        for hash, (group, cells, expiration_cpu, expiration_gpu) in goals.copy().items():
                            if time > expiration_cpu:
                                del goals[hash]
                        for group in goal_groups(frame_names):
                            hash = group.tobytes()
                            if hash not in goals:
                                with State.Episode(start_from_initial=False):
                                    goal = sample_goal(random.randint(1, obs.shape[0] or 1), group)
                                expiration_cpu = torch.tensor(time+10000, device='cpu').int()
                                expiration_gpu = torch.tensor(time+10000).int()
                                goals[hash] = (group, goal, expiration_cpu, expiration_gpu)
                                # And change DODGE direction when changing the goal.
                                if random.randint(1, len(goals)) == 1:
                                    update_direction()

                        # TODO: Here is the perfect place to predict `frame`'s data from its zeroed-out-data version, if we DODGE-predict online. (Though, might want to remove the goal-cells.) (Replays can still do prediction, just, one-step.)
                        #   (Gating the prediction by improvement is easy here, because we'll have 2 distances: of prediction and of `frame` a bit below. And even of the `action` from the previous step…)
                        #   `local_loss` (with goal-cells already inside `frame`), and once we have dist-targets, `global_loss`
                        #   Then `log_metrics()`
                        #   Then `dodge.minimize(L);  optim.step();  optim.zero_grad(True)`

                        # Give goals to the RNN.
                        extra_cells = []
                        for group, (cells, expiration_cpu, expiration_gpu) in goals.items():
                            extra_cells.append(cells)
                        if len(extra_cells): frame = torch.cat([*extra_cells, frame], 0)

                        # Give prev-action & next-observation, remember distance estimates, and sample next action.
                        _, _, dist, _ = transition_(frame)
                        #   TODO: Should preserve `dist`, along with smudges… But shouldn't they be preserved in per-goal lists? With correspondences of which number is which goal? How to compute those correspondences?
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
                            action, _ = sample(torch.cat((query, obs[:, :query.shape[1]]), -1)).clamp(-clamp, clamp)
                            #   (Can also do safe exploration / simple planning, by `sample`ing several actions and only using the lowest-dist-sum (inside `with State.Episode(False): ...`) of those.)
                            #     (Could even plot the regret of sampling-one vs sampling-many, by generating N and having a regret-plot for each (should go to 0 with sufficiently-big N), and see if/when it's worth it.)

                        # Save. (Both replay-buffer and disk.)
                        if save_load and time % steps_per_save == 0:
                            torch.save(transition, save_load)

                        time += 1