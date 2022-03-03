"""
Testing how far we can push "compression = exploration": compression reduces redundancies to only convey the outliers, while exploration seeks out novel cases, so there must be some deep connection here.

---

[RL is notoriously difficult to train agents with.](https://www.alexirpan.com/2018/02/14/rl-hard.html) It hasn't had its BERT moment yet.

Let's not mistake ambition for wisdom. So if we want to proceed, we'd better make sure that we have an idea of what we're doing. Let's try the Feynman technique: creation by teaching, useful because in good teaching, concepts must be as simple as possible.

# Unsupervised RL: state is the goal

In Reinforcement Learning without reinforcement, all we have is an environment that gives observations to our agent and receives actions.

Without rewards, goals can only be (dependent on) states, which incorporate observations and produce actions as they are unrolled in time: `goal = ev(state)`. Postulating "goals are basically states" actually makes things simpler than with reward, for which we'd have to construct a separate differentiable model [(the critic)](https://hal.archives-ouvertes.fr/hal-00756747/file/ivo_smcc12_survey.pdf) to maximize [(the actor)](https://hal.archives-ouvertes.fr/hal-00756747/file/ivo_smcc12_survey.pdf) and [have problems with properly incorporating the future.](https://arxiv.org/abs/2201.12417)

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
# TODO: (Also, make logging create and interactively-update plots for all the values. ...In fact, should have the `logging` module that does that...)



import asyncio
import random
import torch
import torch.nn as nn

import sensornet as sn
import minienv
from model.rnn import RNN
from model.momentum_copy import MomentumCopy
from model.loss import CrossCorrelationLoss



cell_shape, part_size = (8, 24, 64), 8
state_sz, goal_sz = 128, 128
sn.shape(cell_shape, part_size)

minienv.reset(can_reset_the_world = False, allow_suicide = False)



class SkipConnection(nn.Module):
    def __init__(self, *fn): super().__init__();  self.fn = nn.Sequential(*fn)
    def forward(self, x):
        y = self.fn(x)
        return y + x if x.shape == y.shape else y
class SelfAttention(nn.Module):
    def __init__(self, *args, **kwargs): super().__init__();  self.fn = nn.MultiheadAttention(*args, **kwargs)
    def forward(self, x):
        x = torch.unsqueeze(x, -2)
        y, _ = self.fn(x, x, x, need_weights=False)
        return torch.squeeze(y, -2)
def f(in_sz = state_sz, out_sz = state_sz):
    return SkipConnection(
        nn.LayerNorm(in_sz),
        nn.ReLU(),
        nn.Linear(in_sz, out_sz),
    )
def h(in_sz = state_sz, out_sz = state_sz):
    return SkipConnection(
        nn.LayerNorm(in_sz),
        nn.ReLU(),
        SelfAttention(embed_dim=in_sz, num_heads=2),
        f(in_sz, out_sz),
    )



device = 'cuda' if torch.cuda.is_available() else 'cpu'
embed_data = nn.Sequential( # data → state (to concat at the end)
    nn.Linear(sum(cell_shape), state_sz),
    SkipConnection(
        nn.ReLU(),
        nn.LayerNorm(state_sz),
        nn.Linear(state_sz, sum(cell_shape)),
    ),
).to(device)
embed_query = nn.Sequential( # query → state (to concat at the end)
    nn.Linear(sum(cell_shape) - cell_shape[-1], state_sz),
    SkipConnection(
        nn.ReLU(),
        nn.LayerNorm(state_sz),
        nn.Linear(state_sz, sum(cell_shape)),
    ),
).to(device)
condition_state_on_goal = h(sum(cell_shape) + goal_sz, sum(cell_shape)).to(device)
next = nn.Sequential( # state → state; RNN.
    h(sum(cell_shape), sum(cell_shape)),
    h(sum(cell_shape), sum(cell_shape)),
).to(device)
ev = nn.Sequential( # state → goal.
    h(sum(cell_shape), goal_sz),
    h(sum(cell_shape), goal_sz),
).to(device)
# TODO: Code up an actual Module that incorporates data & query and does a transition and returns feedback (through a global variable, or its own variable, probably). Use that as `RNN`'s transition. Don't just call `embed_data`/`embed_query`/`condition_state_on_goal` in the loop, that'll consume memory when we don't want it to.
loss = CrossCorrelationLoss(
    axis=-2, # TODO: (Also try the "cross-norm along -1" variant.)
    decorrelation_strength=.1,
    shuffle_invariant=True,
    also_return_l2=True,
)
def optimizer_stepper(p):
    optimizer = torch.optim.Adam([ # TODO: Would be able to use `p` with an actual Module.
        *embed_data.parameters(),
        *embed_query.parameters(),
        *condition_state_on_goal.parameters(),
        *next.parameters(),
        *ev.parameters(),
    ], lr=1e-4)
    def step():
        optimizer.step()
        optimizer.zero_grad(True)
    return step
def loss_func(prev_state, next_state):
    global loss_was
    eps = 1e-5
    A, B = ev(prev_state), ev(next_state)
    A = (A - A.mean(-1)) / (A.std(-1) + eps)
    B = (B - B.mean(-1)) / (B.std(-1) + eps)
    loss_was = loss(A, B)
    return loss_was
model = RNN(
    transition = next,
    loss = loss_func,
    optimizer = optimizer_stepper, # TODO: We *could* just create an actual optimizer, now.
    backprop_length = lambda: random.randint(2, 4), # TODO: (Figure out what's up with the crazy memory usage.)
    trace = False, # TODO:
)



max_state_cells = 1024
state = torch.randn(16, sum(cell_shape), device=device)
feedback = None
loss_was = 0.
exploration_peaks = [0.]
async def print_loss(data_len, query_len, explored, loss, reachable):
    loss = await sn.torch(torch, loss, True)
    explored = round(explored*100, 2)
    if explored >= exploration_peaks[-1]: exploration_peaks[-1] = explored
    else: exploration_peaks.append(explored)
    if len(exploration_peaks) > 2048:
        exploration_peaks[:-1024] = [sum(exploration_peaks[:-1024]) / len(exploration_peaks[:-1024])]
    explored_avg = sum(exploration_peaks) / len(exploration_peaks)
    print(str(data_len).rjust(3), str(query_len).ljust(2), 'explored', str(explored).rjust(5)+'%', ' avg', str(round(explored_avg, 2)).rjust(5)+'%', ' reachable', str(round(reachable*100, 2)).rjust(5)+'%', '  L2', str(loss)) # TODO: Should have a little system where we call a func with keyword args and it measures max-str-len-so-far and prints everything correctly. ...In `model.logging`.
async def main():
    global state, feedback
    while True:
        # (…Might want to also split data/query into multiple RNN updates if we have too much data.)
        #   (Let the RNN learn the time dynamics, a Transformer is more of a reach-extension mechanism.)
        # (…Might also want to do proper GPT-style pre-training, predicting shifted-by-1-to-the-left input, or ensuring that resulting representations stay the same.)
        await asyncio.sleep(.05) # TODO: Remove. It serves no purpose now, other than going slower. (The fan hardly being active sure is nice, though.)
        data, query, data_error, query_error = await sn.handle(feedback)
        data = embed_data(torch.as_tensor(data, dtype=torch.float32, device=device)) # TODO: This input-incorporation should happen in the transition instead. (Would take much less memory.)
        query = embed_query(torch.as_tensor(query, dtype=torch.float32, device=device))
        state = torch.cat((state, data, query), 0)[-max_state_cells:, :]
        state = condition_state_on_goal(torch.cat((state, ev(state)), 1))
        state = model(state)
        feedback = sn.torch(torch, state[(-query.shape[0] or max_state_cells):, :])
        asyncio.ensure_future(print_loss(data.shape[0], query.shape[0], minienv.explored(), loss_was, minienv.reachable()))
asyncio.run(main()) # TODO: ...Maybe, should have the `sn.run` decorator so that users don't have to waste 2 lines on this...