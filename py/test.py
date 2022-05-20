"""
Testing how far we can push "compression = exploration": compression reduces redundancies to only convey the outliers, while exploration seeks out novel cases, so there must be some deep connection here.

---

[RL is notoriously difficult to train agents with.](https://www.alexirpan.com/2018/02/14/rl-hard.html) It hasn't had its BERT moment yet.

Let's not mistake ambition for wisdom. So if we want to proceed, we'd better make sure that we have an idea of what we're doing. Let's try the Feynman technique: creation by teaching, useful because in good teaching, concepts must be as simple as possible.

# TODO: Re-read. Figure out what went wrong, if anything.

# Unsupervised RL: state is the goal

In Reinforcement Learning without reinforcement, all we have is an environment that gives observations to our agent and receives actions.

Without rewards, goals can only be (dependent on) states, which incorporate observations and produce actions as they are unrolled in time: `goal = ev(state)`. Postulating "goals are `ev`entual states" actually makes things simpler than with reward, for which we'd have to construct a separate differentiable model [(the critic)](https://hal.archives-ouvertes.fr/hal-00756747/file/ivo_smcc12_survey.pdf) to maximize [(the actor)](https://hal.archives-ouvertes.fr/hal-00756747/file/ivo_smcc12_survey.pdf) and [still have problems with properly incorporating the future.](https://arxiv.org/abs/2201.12417)

To prepare for goal-directed behavior, should practice reaching all possible goals, and build a good map of how to get from anywhere to anywhere, for downstream tasks to use.

What can we learn from a dataset of `next: state → state` trajectories, and in particular, from a `src → state → … → state → dst` trajectory? (Note that `next` is a function like in programming, and can also do input/output: non-differentiably, incorporate observations and produce actions.)

# Actions have goals, so infer our goal (actions define goals: critic)

A quick example: say we have an action that means "how hard to press this button", and there are two possible consequences: either it's pressed, or not pressed. A random `next` policy would just emit basically-random numbers as actions, but how do we know what those would lead to, or in other words, compress actions into their goal?

If the goal did direct this trajectory, then it stayed constant, meaning `ev(src) = ev(dst)`. We should extract this invariant through learning, via gradient descent because deep learning performs well.

(The astute among you may have noticed that saying "THE goal" is presumptuous, for there could be many. Meaning that we'd need to model its distribution, not just its average. Could be a direction of future work, or of this one if averaging is not enough in practice.)

Through SGD, we want to ensure: `prev_goal = sg(next_goal)`, or `ev(state) = sg(ev(next(state)))`. (`sg` is for stop-gradient, AKA don't-learn, because goals can't just decide to have had different actions.)

(This can of course be repeated for more than one step, but one-step is simpler.)

# Actions must be goal-directed (goals define actions: actor)

But suppose we do know the goal, such as "press this button"; how to actually plan out a trajectory to it?

The main transition `next: state → state` should know its `ev`entual goal by being `next: (prev_state, goal) → next_state` instead, and learning how to reach that (unchangable) goal.

We want to ensure: `sg(prev_goal) = next_goal`, or `sg(ev(state)) = ev(next(state, ev(state)))`. (`sg` is because actions can't just decide that a goal is not worth going to.)

# Putting it all together

Learning representations end-to-end is good, so let's learn the critic and the actor at the same time.

Summing both options, we get the loss: `ev(state) = ev(next(state, ev(state)))`

We only need to ensure that we don't converge to trivial solutions such as "all-zeroes". For example, could [use cross-correlation](https://arxiv.org/abs/2103.03230): `from model.loss import CrossCorrelationLoss`.

That's the conclusion. If all steps leading to it were correct and well-founded, then the conclusion must be correct and well-founded too, and we can use it to perform Unsupervised RL (URL). (Yeah, URL is a good, funny name for the technique here. Hopefully not too confusing.)

# Wait but how is this related to exploration

Ideally. By extracting invariants, meaningless wandering is cut off, and coherent plans solidify. These plans not only make tricky spots in state-space easier to go to for learned-later plans, but also make sure that the past is not forgotten amid all the novelty-seeking. Which is arguably what exploration is all about. So, ideally, the exploration percentage should just go up.

But if not, we always have contingencies within contingencies. To ensure that futures are sampled uniformly, could have a large batch size (instantiate many pasts at once and [batch-normalize](https://arxiv.org/abs/1502.03167) the goals: be multi-environment) or just allow forking in environments (which `import minienv` here does). To prioritize novelty, could make loss the actual reward and bring in RL for that. No plan survives contact with the enemy, though.

# What else is this related to

In recent years, deep learning has seen the rise of methods for self-supervised learning. Among them, a particular kind of non-contrastive methods for extracting useful representations from images bear strong resemblance to URL.

- [SimSiam](https://arxiv.org/abs/2011.10566): basically `predictor(ev(state1)) = sg(ev(state2))`. Here, `state1` and `state2` are augmented image views put through a neural network.
    - [Bootstrap Your Own Latent](https://arxiv.org/abs/2006.07733): same, but the predicted `ev` on the right is replaced with a slow-moving average of the trained `ev` on the left.

- [Barlow twins](https://arxiv.org/abs/2103.03230): `ev(state1) = ev(state2)` using a cross-correlation loss for soft-whitening.

(Image augmentations include: crop, blur, make it grayscale, change brightness and/or contrast and/or saturation. Models get good by learning to ignore those in their learned representations.)

The main difference from URL is the lack of time in models, followed by the lack of actions. But we could say that `state2 = next(state1, ev(state1))` and `state1 = next(state0, ...)`, meaning that `next` gives another augmentation of the image as an observation and puts it through its neural network again, and to make this match URL's semantics, also connect the other augmentation's representations to one of them through a neural network. This makes one side of the loss useless because it can easily cheat by copying its `ev(state1)` input (can't learn plans if we don't even have actions), but the other is exactly the same as the Barlow twins, so we could claim that as a special case of URL (faster but less general).

Further, the ability to reproduce [the human ability to learn useful representations from interacting with the world](https://xcorr.net/2021/12/31/2021-in-review-unsupervised-brain-models/) can be said to be the main goal of self-supervised learning in computer vision. URL suggests that the structure of the body/environment is usable as data augmentations: for images, we have eyes, which can crop (movement & eyelids), make it grayscale [(eye ](http://hyperphysics.phy-astr.gsu.edu/hbase/vision/rodcone.html)[ro](https://en.wikipedia.org/wiki/Rod_cell)[ds)](https://en.wikipedia.org/wiki/File:Distribution_of_Cones_and_Rods_on_Human_Retina.png), scale and flip and rotate (body movement in 3D), blur (un/focus), adjust brightness (eyelashes). In other domains, it's trivial to envision a "next word" augmentation or "next sound sample after this movement" (and/or implement them with `sensornet`), and URL both extracting invariants and learning to enforce them out of the box.

Nothing forms from nothing, and all ideas have inspirations. In fact, the aforementioned methods were the main inspiration for URL.

Reminder: URL is simply the `ev(state) = ev(next(state, ev(state)))` loss on an RNN.

# TODO: Abandon all hatred. Spread love. No ill-considered approaches like before. Only the well-founded URL. Inspect and support its every step on the way.
"""



# TODO: (This should be our new intro to `test.py`.)
# (Let's assume a discrete model of the world: `universe: state → state`. This describes one boring line/circle through time.)
# (Now, let's consider an agent with bidirectional communication with its world, receiving `obs`ervations and returning `act`ions: `universe: (state, act) → (state, obs)`. Now we have a whole infinite graph of possibilities, with the path selected by `act`ions.)
# (Where that agent ultimately goes can be named its goal. In the sequence of the agent's `obs`ervations, this can be written as `SRC <path> DST <path>: MID0 MID1 MID2 … DST`. This is very [similar to ](https://paperswithcode.com/task/text-infilling)[text ](https://nlp.stanford.edu/pubs/donahue2020infilling.pdf)[infilling](https://beta.openai.com/docs/guides/completion/inserting-text), where `DST` is either in `state`-space or in `obs`-space but variable-length.) (RL writes goals as one-number-utility functions. We follow [upside-down RL](https://arxiv.org/abs/1912.02877) here, but both approaches are general enough to implement each other.)
# (To reach goals most efficiently, the agent should build a map of that graph of possibilities, to run shortest-path queries on. Why *not* use something like a [Transformer](https://arxiv.org/abs/1706.03762) with [RNN-state](https://arxiv.org/abs/2202.05780), and do next-`obs` prediction? Without an ideological agenda, there's no reason to claim that Deep Learning can't succeed at general intelligence, being the most data-driven and successful approach that we have.)
# (`act`ing can actually be exactly the same as next-frame prediction, if actions are included in next-frame observations; then it's easy for human and AI actions to be treated exactly the same. But we do want shortest/best paths and not just any paths, so that prediction has to be [critic-](https://arxiv.org/abs/2006.15134)[regularized](https://arxiv.org/abs/1806.05635); if `obs` are stochastic, the model may become overly optimistic instead of averaging/smudging the predictions.)

#   TODO: …Should we make `DST` `state`-space in our impl too, and to bridge from obs-space to state-space, have a NN from obs to RNN-state, possibly just a single episode of the main NN with just the desired-obs inputted? TODO: But how would this model partial goals, AKA "we don't care about any of the other cells"? By training dists with partial goals?…
#     TODO: Since RNN state is unwieldy to actually condition on, should we learn a separate cell type just for goals, outputted by the main RNN per-cell? How is it learned, exactly? …We either need to store the unroll's goal-cells in replay (and maybe update this on replay), or do a no-DST unroll first then a with-DST unroll… Which approach is better?













# TODO: …What exactly do we implement, then?
# TODO: Have just one RNN, from observations to actions/predictions and prev-dists.
#   TODO: What losses do we want, exactly? Min-filtered critic and critic-filtered regression?
#   TODO: Should we faraway-sample and sort-by-timestamp? Or demand contiguous sequences?
#     TODO: Or both: faraway-sample to determine the goal and distances, then unroll BPTT near the starting point, hoping that meta-learning would pick up what it needs?
#       (…Or are faraway-sequences inherently better for meta-learning…)
#     We can't do faraway-sampling, because next-obs prediction would instantly get broken. We can only faraway-sample the goal, and have to unroll a contiguous sequence.
#   TODO: Do we softly-reset the RNN state when unrolling?
#   TODO: What actions do we search through at unroll-time, exactly? (…Do we need a GAN for potential actions?… But how would it be limited to only not-the-same-future actions?…) (Can we get away with inverting and/or corrupting a single random number per cell?) (Should we do one or a few steps of gradient descent on corrupted actions?)
#     TODO: …Should we maybe output a probability distribution, and treat every number in actions as 0/1?… …No, this assumes that each bit is independent in a cell, which is quite likely to not be true… …But this assumption doesn't really affect anything except for how effective interventions-on-actions are, unless multiple bigger-actions have EXACTLY the same distances; and besides, we can just design envs with bit-actions in mind from the ground up…
#       Kinda starting to like this idea. Because to search, we don't need to *guess* any actions, just sample several times. (Even if interaction-dependencies get screwed up and this search would only do a tiny bit to fix it. Again: envs can be designed to handle this, at least by allowing (sub-)actions to be performed many times, like in painting.)
#       (TODO: Should `sn.handle` also accept the feedback-error, which we can set to `1` here?)
#       TODO: Yes: have `sample(act_prob)→act`, and use that on unroll & replay.
#         …Is there no way to sample autoregressively, or something…?
#           Because this primitive sampling has a good chance of really collapsing the useful combinations, even if just 2 bit-actions are equally good…

# TODO: Only have one RNN net: input prev-act & next-obs, output, per-cell, prev-frame dists and next-frame predictions (since it includes prev-action, this is also the action).
#   TODO: To make next-queries not conflict with prev-queries/actions, pattern-match `sn` tensors via NumPy, and use old indices where names match and create zero-filled 'prev-actions' for new queries. Feedback should read from those indices.
#   TODO: To predict even the first frame of new obs, when storing to the replay buffer, pattern-match the labels of prev & next frames, and insert the missing zero-padded next-frame labels into prev-frame.
#   (Can do safe exploration / simple planning, by `sample`ing several actions and only using the lowest-dist-sum (inside `with State.Episode(False): ...`) of those.)

# TODO: Softly-reset the RNN when unrolling, via the global `with State.Setter(lambda initial, current: initial*.001 + .999*current): ...`.

# TODO: Loss:
#   TODO: Prediction, filtered by advantage (`(dist_replay < dist_policy).float()`). The policy-action is `sample`d.
#   TODO: Min-dist learning: try `next_dist_replay = where(i<j, min(j-i, prev_dist_replay), next_dist*1.1)`. (Worst-case, can try tilted L1 loss.) (Storing `dst` in replays in increasing dist-to-it is *probably* not worth it.)
#   TODO: Maybe, `dist_policy = sg dist_policy*1.1` for GAN-like penalization of ungrounded plans.
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













import asyncio
import random
import torch
import torch.nn as nn

import sensornet as sn
import minienv
from model.log import log, clear



cell_shape = (8,8,8,8, 64)
sn.shape(*cell_shape)

state_sz, goal_sz = 256, 256
max_state_cells = 256

minienv.reset(can_reset_the_world = False, allow_suicide = False, max_nodes=1000)
#   TODO: Should at least make `minienv` work not globally but in a class.



class SkipConnection(nn.Module):
    def __init__(self, *fn): super().__init__();  self.fn = nn.Sequential(*fn)
    def forward(self, x):
        y = self.fn(x)
        return y + x if x.shape == y.shape else y # TODO: Why not slicing if too-big?
class SelfAttention(nn.Module):
    def __init__(self, *args, **kwargs): super().__init__();  self.fn = nn.MultiheadAttention(*args, **kwargs)
    def forward(self, x):
        x = torch.unsqueeze(x, -2)
        y, _ = self.fn(x, x, x, need_weights=False)
        return torch.squeeze(y, -2)
def f(in_sz = state_sz, out_sz = state_sz): # A per-cell transform.
    # TODO: …Why not a one-liner?
    return SkipConnection(
        nn.LayerNorm(in_sz),
        nn.ReLU(),
        nn.Linear(in_sz, out_sz),
    )
def h(in_sz = state_sz, out_sz = state_sz): # A cross-cell transform.
    return SkipConnection(
        nn.LayerNorm(in_sz),
        nn.ReLU(),
        SelfAttention(embed_dim=in_sz, num_heads=2),
        f(in_sz, out_sz), # TODO: …No nested skip-connections, do it properly (it's just 2 one-liners anyway)…
    )
def norm(x, axis=None, eps=1e-5):
    if axis is not None:
        return (x - x.mean(axis, True)) / (x.std(axis, keepdim=True) + eps)
    else:
        return (x - x.mean()) / (x.std() + eps)
class Next(nn.Module): # TODO: …If we don't use `RNN`, then we don't need `Next` either…
    """Incorporate observations & queries, and transition the state. Take feedback directly from the resulting state.

    (For stability, would be a good idea to split `data` & `query` if their concatenation is too long and transition for each chunk, to not forget too much of our internal state: let the RNN learn time-dynamics, Transformer is just a reach-extension mechanism for better optimization. Currently not implemented.)"""
    def __init__(self, embed_data, embed_query, max_state_cells, transition, condition_state_on_goal, goal):
        super().__init__()
        self.embed_data = embed_data
        self.embed_query = embed_query
        self.max_state_cells = max_state_cells
        self.transition = transition
        self.condition_state_on_goal = condition_state_on_goal
        self.goal = goal
    def forward(self, state, data, query):
        data = self.embed_data(torch.as_tensor(data, dtype=torch.float32, device=state.device))
        query = self.embed_query(torch.as_tensor(query, dtype=torch.float32, device=state.device))
        state = torch.cat((state, data, query), 0)[-self.max_state_cells:, :]
        state = self.condition_state_on_goal(torch.cat((state, self.goal(state)), 1))
        state = self.transition(state)
        return state



device = 'cuda' if torch.cuda.is_available() else 'cpu'
embed_data = nn.Sequential( # data → state (to concat at the end)
    nn.Linear(sum(cell_shape), state_sz),
    f(state_sz),
).to(device)
embed_query = nn.Sequential( # query → state (to concat at the end)
    nn.Linear(sum(cell_shape) - cell_shape[-1], state_sz),
    f(state_sz),
).to(device)
condition_state_on_goal = h(state_sz + goal_sz, state_sz).to(device) # TODO: Why does this exist?
transition = nn.Sequential( # state → state; the differentiable part of the RNN transition.
    h(state_sz, state_sz),
    h(state_sz, state_sz),
).to(device)
ev = nn.Sequential( # state → goal.
    h(state_sz, goal_sz),
    h(goal_sz, goal_sz),
).to(device)
next = Next(embed_data, embed_query, max_state_cells, transition, condition_state_on_goal, ev)



state = torch.randn(max_state_cells, state_sz, device=device)
feedback = None
exploration_peaks = [0.]
async def print_loss(data_len, query_len, explored, reachable):
    # TODO: …We should make `log` be able to use this `await sn.torch(torch, x, True)` directly, so that we avoid the awkwardness of having to manually await each item…
    explored = round(explored*100, 2)
    if explored >= exploration_peaks[-1]: exploration_peaks[-1] = explored
    else: exploration_peaks.append(explored)
    if len(exploration_peaks) > 17:
        exploration_peaks[:-16] = [sum(exploration_peaks[:-16]) / len(exploration_peaks[:-16])]
    explored_avg = sum(exploration_peaks) / len(exploration_peaks)
    reachable = round(reachable*100, 2)
    # Ignored: `data_len`, `query_len`, `reachable`
    log(explored=explored, explored_avg=explored_avg)
@sn.run
async def main():
    global state, feedback
    n = 0
    while True:
        n += 1
        if n == 1000 or n == 10000: clear()
        # await asyncio.sleep(.05) # TODO: Remove this to go fast.
        data, query, data_error, query_error = await sn.handle(feedback)
        # import numpy as np;  data = np.concatenate((np.random.randn(10, data.shape[-1]), data)) # TODO: This simple data noise is not a principled approach at all.
        # TODO: (Also may want to chunk `data` and `query` here, and/or add the error as noise.)

        # noise_level = abs(2*(n%500) - n%1000) # Go up, then go down, then repeat.
        # state = state + torch.rand_like(state)*noise_level*2-1
        # state = norm(state) # ...This is even more counterproductive than expected.

        state = next(state, data, query)
        feedback = sn.torch(torch, state[(-query.shape[0] or max_state_cells):, :data.shape[-1]])

        asyncio.ensure_future(print_loss(data.shape[0], query.shape[0], minienv.explored(), minienv.reachable()))

        # if random.randint(1,200) == 1: state = torch.randn_like(state, requires_grad=True) # TODO: Does this help exploration? ...A bit, I guess? Uniform improvement to ~3%, except when representations collapse or something.