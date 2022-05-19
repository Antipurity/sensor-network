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













# TODO: …What exactly do we implement, then? Do we not need any new `model/` files, and should just transfer cont-control lessons to `test.py`?
#   …Do we really still want to do the RNN-of-Transformer architecture…
#     TODO: …Maybe we should use fast weight programmers instead?
#       Maybe even a [self-referential weight matrix](https://arxiv.org/pdf/2202.05780.pdf)? TODO: Yes. A `model/` file. `model/recurrency.py`, perhaps.
#         (This self-attention-like layer would store its initial & current matrices, and allow explicitly changing them. Initial-matrix requires grad.)
#         (To not access layers one-by-one to do replays: need a ctx-manager that resets current-weights to initial, and on exit, restores them.) TODO: Should we extend this idea to generic RNN-state-tensors? …Why, yes.
#           …That 'generic RNN state' should also follow that pattern of "softly-reset toward an initial state", right?
#           TODO: Have the `State` class first, with `with State.New(): ...`.
#           TODO: (…What does "predict next RNN-state using prev RNN-state" mean if we don't really have access to the actual RNN states directly anymore… Does it have any connections with Reptile or something…)
#             (It updates initial-state toward final-state, albeit via SGD rather than polyak-avg. Which is the opposite of soft-reset, in both ways. …What if we did actually do this, via gradient during replay, trying to make each initial-state the average of possible-states? Doesn't it augment, not replace? Make initial-states as non-intrusive as possible?)
#               (Should allow attaching global handlers of state-update, to compute such a loss.)
#         (Updates are additive, so replays can skip huge timespans: faraway-sample and sort-by-timestamp and update in sequence.)
#     (…May also want to do the [Reptile](https://openai.com/blog/reptile/) approximation to learn beyond BPTT, where instead of infinite-horizon backprop, we do 2+ SGD steps (each is a BPTT unroll here), then update all trainable tensors toward their new versions, and still do BPTT at unroll-time.)
#       (Just like SWRM, these updates are also linear, making us able to do faraway-sampling for faraway-BPTT.)
#       (These unroll-time updates face the same problem as SWRM: unroll-time params diverge from what we optimize. Could use the same solution: slowly polyak-avg them toward the initial/optimized params.)
#       (…Would self-referential weight matrices not be enough, making us need to add a second optimizer via Reptile?)
# TODO: …Will we still want to have separate embedders for data & queries, or not? …Why *not* have them, thin if needed?
#   TODO: What NNs do we want, exactly?

# TODO: Assume a full env with bidirectional obs/act communication with our agent (any possible universe can be described as one-step updates): `env(prev,act)→(next,obs)`, where `prev` and `next`s are hidden `state`s.
#   And we have goal(state)→dst for what `act`s want. (After all, all trajectories end up *somewhere*, so might as well label and learn destinations explicitly.) (This description is complete, and with precise-enough goals, no further intervention on trajectories is needed, so we can just find the shortest path.)
#   THIS is the real structure that NNs learn, isn't it?
#   The env forms an infinite graph, and we'd like to find all-to-all paths in it. That's all that agents are.
#   Agents had better learn a full map of the world. For that, these NNs need:
#     - Theoretically, the inverted env `(state, obs) → (state, act)`, but it should be split to better model `env`'s exact inputs/outputs, and to allow tampering with inputs/actions/goals:
#       - `act(src, dst) → act` and `dist(src, act, dst) → dist`.
#         - `update(history, obs) → src`, because the NN is not blind.
#         - `env(src, act) → history`, to round out the RNN.
#         - (`dst`s of `src`s should be provided when storing samples in the replay buffer. Probably zero-out and/or defer the storage if not available immediately.)
#   From this, we can infer several equations/losses (like in Noether's theorem, symmetry/equality implies conservation/zero-difference):
#     - Learning to predict consequences-of-`next_obs` while extracting mutual info (directly, the result of `update`): `history = sg update.copy(history, obs)`, possibly for `next_history` and `next_obs`.
#       - (Not predicting `obs` directly because its non-determinism leads to smearing.) (No need for another NN for `history`, because the prior `env` call could take care of obs-independent time-keeping.)
#       - (Allows [MuZero](https://arxiv.org/abs/1911.08265)-like no-`obs` planning, in addition to safe exploration AKA randomly proposing an action and evaluating whether distance to a 'safe' goal increases too egregiously.)
#       - …Wait: but if we embed observations & queries anyway, then can't we avoid smearing by just predicting those embeddings (with targets being momentum-slowed, exactly like in BYOL)?… (Not inspecting `history` explicitly would mesh with `model/recurrency.py` stuff better too.)
#     - *Maybe*, backward-prediction too, so that single-step updates can still sense the past despite gradient-descent being unconnected: the autoencoder `reverse_update(obscure(history), obs) = sg history`, possibly reversing `prev_act` and `prev_obs` too.
#     - Min-dist actions:
#       - …The standard methods have proven superior: dist-prediction along real trajectories, [self-imitation](https://arxiv.org/pdf/1806.05635.pdf), possibly DDPG, and a discrete search among actions at unroll-time. And synthetic-gradient for simple learning-through-time.
#         - Learn not dist mean/median but its min, likely via quantile regression (tilted L1 loss). (Use distributional RL to mine for rare-but-good actions, such as when random trajectory fluctuation only RARELY finds the goal and we need more search there to refine acts.) (Downside: learning the 1/1000th median is 1000 times slower. Which we'll need, in episodic envs with lots of uselessly-big sampled dists.)
#           - (An alternative didn't work well in a trivial 2D env: gating each next dist-level by either its prev dist-level (dropping targets if more than that) or nothing, so that each is lower. Possibly because that env is a mad chase to first-correctish-dist-map; might work in other envs.)
#           - May also use a stable (non-self-modifying) approximation to Floyd-Warshall algo to learn min-dists (mine for regret to minimize) that we may not have seen directly: first-level targets are `j-i`, and each next dist-level's target is prev dist-level's `d2(a,b) = min(d(a,m) + d(m,b))` where `d(a,a)=0` and `d(a,b) = min(d(a,b), j-i) if i<j else ∞`, with midpoints from a whole minibatch, and from-replay-buffer actions. For goal-space `b`s, `m` must still be full-space.   At least, each level would turn more-than-mean targets into mean; for convergence speed, may want to mult loss by `pred<target ? 2 : .5` or smth similar.
#           - Mine for regret harder: replay-buffer prioritization of max-regret (of `b`: mean/max `dist(a,b) - (j-i)` among `a`) samples for fastest spreading of the influence of discovered largest shortcuts. At unroll, always overwrite the last sample; at replay, sample that plus randoms, and sort by minibatch-regret and write-back that sorted data. (Also good for easily sampling good unroll-time goals, getting more data in most-promising AKA max-regret areas.)
#     - Penalize probably-unconnected components (which could mislead optimization if left untreated, especially with Floyd-Warshall search) like a GAN: at unroll, store `dst`, and on replay, maximize distances to `dst`s. This on-policy loss must be weaker than dist-learning loss.













import asyncio
import random
import torch
import torch.nn as nn

import sensornet as sn
import minienv
from model.rnn import RNN # TODO: …Do we even want *this*? Aren't we kinda tired of BPTT? (And even if we aren't, isn't this trivial to implement?)
from model.loss import CrossCorrelationLoss # TODO: No; don't do this.
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
number_loss = CrossCorrelationLoss( # TODO: No.
    axis=-2,
    decorrelation_strength=.00,
    shuffle_invariant=False, # TODO: ...Wait, why is shuffle-invariance like the magic sauce? ...Is it because it makes optimization impossible... Yep. All our results were complete bullshit.
    also_return_l2=True,
)
cell_loss = CrossCorrelationLoss( # TODO: No.
    axis=-1,
    decorrelation_strength=1,
    also_return_l2=True,
)
CCL_was, L2_was = 0., 0.
def loss_func(prev_state, next_state, *_): # TODO: No.
    global CCL_was, L2_was
    A, B = norm(ev(prev_state)), norm(ev(next_state))
    A, B = A.unsqueeze(-2), B.unsqueeze(-2) # TODO:
    CCL_was, L2_was = cell_loss(A, B) # TODO: Also number_loss, cell_loss. Maybe both at the same time.
    # TODO: ...Wait, if CCL is so low, then how can L2 still be stuck at 60k...? WHAT IS GOING ON
    return CCL_was
model = RNN(
    transition = next,
    loss = loss_func,
    optimizer = lambda p: torch.optim.Adam(p, lr=1e-4),
    backprop_length = lambda: random.randint(2, 4), # TODO:
    trace = False, # TODO: (The loss is still not grad-checkpointed, though. ...Maybe `RNN` could allow returning tuples from transitions, and pass all to loss then discard all non-first results? Then we won't need memory, though print_loss will be a bit out of date... As a bonus, we won't have to compute `ev(prev_state)` twice.)
)



state = torch.randn(max_state_cells, state_sz, device=device)
feedback = None
exploration_peaks = [0.]
async def print_loss(data_len, query_len, explored, reachable, CCL, L2):
    # TODO: …We should make `log` be able to use this `sn.torch` directly, so that we avoid the awkwardness of having to manually await each item…
    CCL, L2 = sn.torch(torch, CCL, True), sn.torch(torch, L2, True)
    CCL, L2 = await CCL, await L2
    explored = round(explored*100, 2)
    if explored >= exploration_peaks[-1]: exploration_peaks[-1] = explored
    else: exploration_peaks.append(explored)
    if len(exploration_peaks) > 17:
        exploration_peaks[:-16] = [sum(exploration_peaks[:-16]) / len(exploration_peaks[:-16])]
    explored_avg = sum(exploration_peaks) / len(exploration_peaks)
    reachable = round(reachable*100, 2)
    # Ignored: `data_len`, `query_len`, `reachable`
    log(explored=explored, explored_avg=explored_avg)
    log(1, False, CCL=CCL)
    log(2, False, L2=L2)
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

        state = model(state, data, query)
        feedback = sn.torch(torch, state[(-query.shape[0] or max_state_cells):, :data.shape[-1]])

        asyncio.ensure_future(print_loss(data.shape[0], query.shape[0], minienv.explored(), minienv.reachable(), CCL_was, L2_was))

        # if random.randint(1,200) == 1: state = torch.randn_like(state, requires_grad=True) # TODO: Does this help exploration? ...A bit, I guess? Uniform improvement to ~3%, except when representations collapse or something.