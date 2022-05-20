- `sensornet`: the directory for a Python impl of a sensor network (named numeric inputs/outputs).

---

Theoretically, with an environment that gives inputs/outputs such as `sensornet`, a user should be able to just plug an AI model and make it do anything.

But in practice, even though deep learning enjoys tremendous success in self/supervised learning, *reinforcement* learning is another matter. The ability to learn rich plans with only a dozen actions (outputs) is somewhat pathetic compared any modern non-RL task. A unified algorithm that can seamlessly learn good representations from fully exploring the environment, good enough to simply unleash on the Internet to learn it, is currently, at the very least, just beyond the cutting edge of AI.

So, a sensor network without such an algorithm is just a problem looking for a solution.

---

Here, in `py`, we test candidates:

- `test_board_wanderings.py`: a discrete env with 4 actions, where we try out a hypothesis that exploration is about learning a map from anywhere to anywhere, so that fine-tuning for a specific task can just learn the goal and not how to get to it. Conclusion: a map sure is learned, but there are problems (loss-approximation is hard, actions are too discrete for gradient descent to learn, unable to include past-state and still learn, and the env is specially-constructed for this).

- `test_board_revisit.py`: really, really tried to not have to learn `(source, destination) → dist` quadratically-many distances. Ended up learning locally-[isometric](https://en.wikipedia.org/wiki/Isomap)[ maps](https://en.wikipedia.org/wiki/Johnson%E2%80%93Lindenstrauss_lemma) (embed `source` and `destination`, and make the L2 distance as required), and using faraway timestamp-aware sampling, and combining solved subtasks wherever going through a midpoint is faster.

- `test_continuous_control.py`: after much testing, basic well-known RL techniques have proved their superiority, or at least that these environments are too trivial. Neural-pathfinding is still better than single-number rewards, though: a [zero-shot human-](https://github.com/openai/CLIP)[enabled interface](https://cliport.github.io/), richer data due to goal relabeling, much easier inverse-'RL', and [actual RL is a special case anyway.](https://arxiv.org/abs/1912.02875)

- …

- `test.py` and `minienv`: the eventual goal, using `sensornet` for IO, where we don't just explore, we *learn* to explore, in an env with a variable number of diverse actions. It informs the basic form of a solution: differentiable, continuous, Transformer-based.