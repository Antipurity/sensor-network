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












# TODO: During unroll, preserve all dist/smudge predictions (computed when we input prev-act and next-obs stuff into `transition_`) and all (per-goal-group) local per-goal dist/smudge.
#   TODO: Whenever we change a goal, compute all `global_dists` and minimize prediction loss.
#     TODO: …We should rewrite `full_loss` to no longer perform all those extra manipulations on `dst`, right?… TODO: Write down the changes.
#       …Since dist-targets are not known locally, `full_loss` *might* have to be split into `local_loss` (learning gated-by-dist/smudge-predictions of cell contents) and `global_loss` (learning dists/smudge)…
# TODO: In `full_loss` or whenever, keep track of int32 per-cell indices of goal-groups, and use them to index `stack`ed global dists/smudgings.
#   …Or we could use `goal_group_ids` and match goal groups in `frame`…
#   TODO: Learn the min-smudging too, just like distances.
#   TODO: Gate losses by decreasing-dists.
# TODO: Remove `replay_buffer` entirely; no goal-relabeling at all. Don't just learn distances from local information, but instead minimize losses during rollouts: when we save full frames to the replay buffer, get `L = full_loss(…)` inputs and `dodge.minimize(L)`.



# TODO: Stop sampling goals from the replay buffer, and start *generating* goals (much like in http://proceedings.mlr.press/v100/nair20a/nair20a.pdf):
#   TODO: In `full_loss`, get latents of `dst` first, then sample & use those latents to turn 0-filled `dst`-sized cells into `dst` (in the same GPU-call as the rest of the `frame`) if good.
#     TODO: For goals, "good" means "sample-dist is higher" (whereas for actions, "good" means "sample-dist is lower"). (Goals are then neither too hard (AKA when never occuring as real targets) nor too easy (AKA when real-dist is lower than expected).)
#       (This is [AMIGo](https://arxiv.org/pdf/2006.12122.pdf)-inspired. Alternatively, could make goals learn regret rather than dists and maximize regret rather than dists; or even be [AdaGoal](https://arxiv.org/pdf/2111.12045.pdf)-like and have an ensemble of RNNs, and maximize dist-error.)
#       TODO: Possibly, for goals, "good" should mean "sample-dist is not as expected". (So that faster-than-expected paths still have opportunity to be retreated and learned-from more.)
#   TODO: At unroll-time & goal-selection-time, rather than choosing anything from the replay buffer, simply choose a random number of goal-cells to generate (and a `torch.randn` latent) then generate them.



# TODO: Should we generate `dst` cells every step with the same latents (stored per-goal-group) (RNN-input is all-0s but with goal-group-id), and make them all predict the real/first `dst` whenever the condition fits (dist-error is higher (either locally or anywhere-in-the-future) and smudging is lower)? Good idea to make this temporally-coherent, right — since the whole path does have the same `dst` as a potential future?
# TODO: Should we generate a few `dst`-latent (or random-latent?) `src` cells every step with a special input structure (1s in the first number and 0s everywhere else; queries & setting are possible), and predict/autoencode themselves whenever the path has contained any dist-prediction errors (so, same as `dst`'s gating, but for the past instead for the future)? So that `dst` can strive toward non-input unpredictable outcomes?
#   TODO: NEED an env that consists of many buttons, which have to be pushed in a password-sequence that changes rarely, with a bit-indicator for "was the last button correct". With enough buttons, state-based exploration should discover nothing, whereas proper full-past exploration *should* discover the sequence eventually.
#   TODO: POSSIBLY WANT `dst`-generation to be conditioned on whether it can only use `src`-cells, and compute separate dists & smudges for `src`-only learning, and have a hyperparam that makes only `src` cells generatable (envs can still specify input-space goals).
#     TODO: If so, WANT an env that can only be solved with good exploration (a big maze with resetting), to give us the success criterion: the env works both with and without input-space goals.



# TODO: …Wait: `sn.Int(…, goal=True)` is autoregressive and thus can't be specified at every single timestep, because ints are not limited to single timesteps…
#   …The problem is that we *can't* give up the speed advantage of putting many cells through the RNN in parallel, but at the same time, we also can't let `Int`s be spread across multiple timesteps because then big-`Int` goals are impossible…
#     TODO: …Maybe `sn` should explicitly handle within-step dependencies, by making `handle` also return a list/tuple of per-stage end-of-stage (so `[cells]` if without dependencies, `[N, cells]` with 1, etc), and making `pipe` know & handle that, and POSSIBLY making `set`/`query`/`get` be able to accept tuples/lists of datatypes to be put in-sequence (implemented via temporary-internal-state: make each outer global call establish a context in which each inner global call appends to a new stage, which is good-enough because our async-semantics schedule data immediately)?… And possibly have the `sn.List` datatype, known in `_default_typing` (and, default-typable datatypes should mention their own alternative forms in docs)?…
#       …Actually, maybe it's in fact better to generate *everything* that came from a single env autoregressively?… Or is it better to allow envs to decide, which would be easy if we have `sn.List`s… If we allow envs to decide, then we can have envs that compare both ways, in terms of both speed and loss…
#       …And what if there are *inter*-env dependencies? Do we really want to spend this much time on limiting the model's capabilities?
#         TODO: For generality, maybe we should make `sn` blissfully unaware of discrete-sampling difficulties, but only here, generate all analog cells in parallel first and then generate ALL discrete cells autoregressively (very slow but correct)? In the worst case, we can still regain a lot of performance by autodetecting goal-groups and autoregressively generating each group's ints in parallel.






# TODO: …Maybe, have per-SRWM-matrix synthetic gradients via making the synth-grad NNs (possibly even the main RNN) learning to output key & value vectors for each matrix (possibly more than 1 pair), the outer product of which is taken to be the gradient?…
#   …Maybe instead of BPTT with its periodic-stalling, we should do backprop-less DODGE online (even with `dodge_optimizes_params=1`, it should help quite a bit) with ≈1000-step-delayed one-step synth-grad learning following at a distance… Disabling DODGE before that and re-enabling it with the old direction afterward… Possibly even with Reptile or MAML…

# TODO: Do dropout of frame-cells when doing synth-grad replays (forward-unroll with DODGE can see all). (This will make sampling actions-without-next-frame at unroll-time not out-of-distribution, *and* make the net try to minimize dependencies on inputs.)






# TODO: …May want a hyperparam for what to do with digital inputs: leave as binary encodings, VS lookup in a learnable table, VS lookup in a fixed random table…






# TODO: …Maybe also add support for nucleus sampling (like zeroing out all probabilities that are less than 1/40 and re-normalizing), so that we don't very-rarely sample overly-low-probability discrete actions?…






# TODO: Run & fix the copy-task in `env/copy.py`, to test our implementation.

# TODO: An env that has both analog actions and analog goals: make the goal a random image (plus the "at the final-state" bit), then get like 3 images (exposing "NOT at the final-state"), then expose the sum of those actions and the fact that this is the final-state.
# TODO: Is it possible to learn an analog-reward-goal which isn't ever actually encountered, but is set to 1 by the env so that the model always maximizes reward?
#   1st way: detect when a `encountered_dst`-cell has the same name as a `requested_dst`-cell (if it exists) and measure distortion to increase dist by. Needs cell-names to coincide though, which is quadratically-unlikely in general.
#   2nd way: pre-decide distortion in `full_loss` (50% of cases, 0…1 distortion), distort `dst`, and increase dist by it (in log-space, for severe penalties). Probably a good idea; certainly better than waiting for someone to request impossible destinations and SLOWLY learning that they are impossible AFTERWARDS…

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
#   TODO: Use `make_normal` in VAE's loss; use `normal` when sampling VAE's latent variables.

import sensornet as sn

from model.log import log, clear



# Hyperparameters.
cell_shape = (8,8,8,8, 256)
sn.shape(*cell_shape)

state_sz = 256
latent_sz = 16 # For VAE generation.
slow_mode = .05 # Artificial delay per step.

choices_per_cell = 256

lr = 1e-3
dodge_optimizes_params = 1000 # `DODGE` performs better with small direction-vectors.
replays_per_step = 2
max_replay_buffer_len = 1024

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
replay_buffer = []
# Our interface to multigroup combined-cells goals (OR/AND goals): the last name part is `group_id`, AKA user ID.
#   Via carefully engineering the loss, users can have entirely separate threads of experience that reach ALL the goals in their group (but the NN can of course learn to share data between its threads).
def goal_group_ids(frame):
    """Returns the `set` of all goal-group IDs contained in the `frame` (a 2D NumPy array), each a byte-objects, suitable for indexing a dictionary with."""
    return np.unique(frame[:, sum(cell_shape[:-2]) : sum(cell_shape[:-1])], axis=0)



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
    @staticmethod
    def as_goal(frame):
        return torch.cat((torch.ones(frame.shape[0], 1), frame[:, 1:]), -1)



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





# Computable-on-unroll distances.
def local_dist(base: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    """`local_dist(base, goal) → smudge`

    Given sets of cells of observations & goals, computes local-dist (smudging) between them, which is 0 when the `goal` is 100% reached and more when not. (Learned pathfinding should select min-dist actions among min-smudging paths, so that even unreachable goals can have best-effort paths known instead of being out-of-distribution.)

    For efficiency, `goal` may be several stacked goal-groups at once."""
    assert len(base.shape) == 2 and len(goal.shape) >= 2 and base.shape[-1] == goal.shape[-1] == sn.cell_size
    base, goal = detach(base), detach(goal)
    max_smudge = sn.cell_size
    with torch.no_grad():
        base_logits, goal_logits = Sampler.target(base, max_smudge), Sampler.target(goal, max_smudge)
        cross_smudges = (base_logits.unsqueeze(-3) - goal_logits.unsqueeze(-2)).abs().sum(-1)
        return cross_smudges.min(-2)[0].mean(-1)
# TODO: …Wouldn't we also like to have `local_dists` which splits BOTH base & goal into matching goal-groups, then measures per-goal-group smudges? Would free us from having to do this in `full_loss`, right?
#   …Wouldn't we want NumPy names for this?… …The goal-cells list is likely pre-prepared by the unroll, but `base` is not split into goal-groups and we need to do so via NumPy…
def global_dists(smudges: torch.Tensor, final_dist_pred: torch.Tensor, final_smudge_pred: torch.Tensor):
    """`global_dists(smudges, final_dist_pred, final_smudge_pred) → (smudge, dists)`

    On a path to a goal, the min local-dist (`smudge`) is considered the best that we can do to reach it. `dists` before it will count out the steps to reach it, to be used as prediction targets."""
    assert final_dist_pred.shape[-1] == final_smudge_pred.shape[-1] == 1
    smudges, final_dist_pred, final_smudge_pred = detach(smudges), detach(final_dist_pred), detach(final_smudge_pred)
    with torch.no_grad():
        smudge = smudges.min(-1, keepdim=True)[0].min(final_smudge_pred + 1) # TODO: Maybe "smudging" should be copied if we neither reached the goal nor did we claim to, and copied+1 if we're claiming to be close? …But then, it's not enough to have just *final* dist preds, we need to have *all*…
        reached = smudges <= smudge+1
        next_of_reached = torch.cat((torch.zeros(*reached.shape[:-1], 1), reached[..., :-1].float()), -1)
        dists = final_dist_pred + torch.arange(1, reached.shape[-1]+1) # Count-out.
        left_scan = ((dists - 1) * next_of_reached).cummax(-1)[0]
        dists = (dists - left_scan).flip(-1)
        return (smudge, dists)





# Loss.
def fill_in(target):
    """VAE integration: encode `target` and decode an alternative to it. If called in pre-`target` RNN state, produces target's info and alternative-target's info.

    Result: `((target_dist, target, target_logits), (pred_dist, pred, pred_logits), regularization_loss)`"""
    # TODO: Shouldn't this *also* output smudgings? So that we could predict them, and gate by them?
    assert len(target.shape) == 2
    with State.Episode(start_from_initial=False): # Decoder & distance.
        _, target_latent_info, target_dist, _ = transition_(target)
        target_logits = sample.target(target)
        target_latent = normal(target_latent_info)
        regularization_loss = make_normal(target_latent_info).sum(-1, keepdim=True)
    name_sz = sn.cell_size - cell_shape[-1]
    pred, pred_logits = sample(target[:, : name_sz], target_latent) # Encoder.
    with State.Episode(start_from_initial=False): # Distance.
        _, _, pred_dist, _ = transition_(pred)
    return ((target_dist, target, target_logits), (pred_dist, pred, pred_logits), regularization_loss)

def full_loss(prev_ep, frame, dst, timediff, regret_cpu):
    """`full_loss(prev_ep, frame, dst, timediff, regret_cpu)`

    Predicts the `prev_frame`→`frame` transition IF it's closer to the goal than what we can sample.

    - `prev_ep` is the context that knows `prev_frame`.
    - `frame` is the immediately-next prev-action and next-observation cells.
    - `dst` is some faraway goal.
    - `timediff` is the distance to that goal, as a `(frame_cells + dst_cells, 1)`-shaped int32 tensor. Must always be *more* than 0."""

    with prev_ep:
        # Multigroup ("OR") goals, where many threads of experience (with their group IDs) can pursue separate goals.
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

        # Contrast `frame` with our own understanding of it.
        (frame_dist, frame, frame_logits), (pred_dist, pred, pred_logits), reg_loss = fill_in(frame)

        # Critic-regularized regression: if `frame_pred` regrets not being `frame`, make it into `frame`.
        dist_is_better = detach(frame_dist)[:, -1:] < detach(pred_dist)[:, -1:]
        mask = (dist_is_better | ~sample.analog_mask(frame)).float()
        predict_loss = (is_learned * (pred_logits - frame_logits).square() * mask).sum()

        # Remember our regret of not being the `frame`. *Only* remember what we regret.
        regret = (is_learned * (pred_dist[:, -1:] - frame_dist[:, -1:])).sum() / (is_learned.sum() + 1e-2)
        regret_cpu.copy_(detach(regret), non_blocking=True)

        # Critic regression: `dist = sg timediff`, but *optimistic*, meaning, min dist and not mean dist.
        #   (Worst-case, can also try tilted L1 loss AKA distributional RL.)
        dist_limit = frame_dist + .5 # Always be able to increase, but not by too much.
        dist_loss = (is_learned * (frame_dist.log2() - detach(timediff.min(dist_limit)).log2()).square()).sum()

        # GAN-like penalization of ungrounded plans.
        dist_penalty = 1.05
        ungrounded_dist_loss = 0 # (is_learned * (frame_pred_dist - detach(frame_pred_dist) * dist_penalty).square()).sum() # TODO: (*Should* be superseded by our unformed "go to min-local-dist min-global-dist goal-cells" plans, right?…)

        log(0, False, torch, improvement = mask.mean() - (~sample.analog_mask(frame)).float().mean())
        #   TODO: Why is improvement always 0, even though `predict_loss` is not? …Wait, is it `nan` sometimes, from looking at the plots?…
        #     (From looking at the plot, it's `nan` sometimes in the beginning, but not in the end.)
        #     TODO: …Why are we leaving out digital cells anyway? `env/copy.py` is a digital-only env anyway, so doesn't this not make any sense?
        log(1, False, torch, predict_loss=predict_loss, dist_loss=dist_loss, ungrounded_dist_loss=ungrounded_dist_loss, reg_loss=reg_loss)
        n = 2
        for name, env in envs.items():
            if hasattr(env, 'metric'):
                log(n, False, torch, **{name+'.'+k: v for k,v in env.metric().items()})
                n += 1
        # TODO: …How can we possibly fix the loss not decreasing?…

        loss = predict_loss + dist_loss + ungrounded_dist_loss + reg_loss
        return loss

def replay(optim, current_frame, current_time): # TODO: …Don't replay, *maybe*?…
    """Remembers a frame from a distant past, so that the NN can reinforce actions and observations when it needs to go to the present."""
    if len(replay_buffer) < 8: return

    L = 0
    for _ in range(replays_per_step):

        time, ep, frame, regret_cpu = random.choice(replay_buffer)

        # Learn combined-cells ("AND") goals, by omitting some cells randomly.
        dst_is_picked = np.random.rand(current_frame.shape[0]) < (.05+.95*random.random())
        dst = current_frame[dst_is_picked]

        # Learn.
        timediff = torch.full((frame.shape[0] + dst.shape[0], 1), float(current_time - time))
        L = L + full_loss(ep, frame, dst=dst, timediff=timediff, regret_cpu=regret_cpu)

        # If our replay buffer gets too big, leave only max-regret samples.
        if len(replay_buffer) > max_replay_buffer_len:
            replay_buffer.sort(key = lambda sample: sample[3], reverse=True)
            del replay_buffer[(max_replay_buffer_len // 2):]

    # Optimize NN params.
    dodge.minimize(L)
    optim.step();  optim.zero_grad(True)





# Unroll.
@sn.run
async def main():
    direction = None
    def update_direction():
        nonlocal direction
        direction = torch.where(torch.rand(dodge.sz) < dodge_optimizes_params / dodge.sz, torch.randn(dodge.sz), torch.zeros(dodge.sz))
        dodge.restart(direction)
    with fw.dual_level():
        with State.Setter(lambda state, to: state.initial*.001 + .999*to): # Soft-reset.
            update_direction()
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
                        groups = set(name.tobytes() for name in goal_group_ids(frame_names))

                        # Zero-pad `query` to be action-sized.
                        obs, query = torch.tensor(obs), torch.tensor(query)
                        frame = torch.cat((action, obs), 0) if action is not None else obs
                        frame_to_save = frame[not_goal_cells]
                        life_to_save = life.clone(remember_on_exit=False)

                        # Delete/update our `goals` when we think we reached them.
                        #   TODO: Sample `dst` instead of this `replay_buffer` sampling. (Possibly generated along with the rest of the query at every step, with the same latents until the goal is changed.)
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
                                        update_direction()

                        # TODO: Here is the perfect place to predict `frame`'s data from its zeroed-out-data version, if we DODGE-predict online. (Though, might want to remove the goal-cells.) (Replays can still do prediction, just, one-step.)
                        #   (Gating the prediction by improvement is easy here, because we'll have 2 distances: of prediction and of `frame` a bit below. And even of the `action` from the previous step…)
                        #   `gated_generative_loss`

                        # Give goals to the RNN.
                        extra_cells = []
                        for group, (cells, expiration_cpu, expiration_gpu) in goals.items():
                            extra_cells.append(cells)
                        if len(extra_cells): frame = torch.cat([*extra_cells, frame], 0)

                        # Give prev-action & next-observation, remember distance estimates, and sample next action.
                        _, _, dist, _ = transition_(frame)
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
                            action, _ = sample(torch.cat((query, obs[:, :query.shape[1]]), -1))
                            #   (Can also do safe exploration / simple planning, by `sample`ing several actions and only using the lowest-dist-sum (inside `with State.Episode(False): ...`) of those.)
                            #     (Could even plot the regret of sampling-one vs sampling-many, by generating N and having a regret-plot for each (should go to 0 with sufficiently-big N), and see if/when it's worth it.)

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