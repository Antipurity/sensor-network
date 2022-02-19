Not every action serves a [consistent ](https://deepmind.com/research/publications/2021/The-Difficulty-of-Passive-Learning-in-Deep-Reinforcement-Learning)[goal](https://deepsense.ai/what-is-reinforcement-learning-the-complete-guide/), so to gain future potential, have to *explore*. This minienv is intended as a sanity check of whether a handler is capable of exploration: it essentially exposes a bunch of progressively-less-likely states.

To use it:

```python
import sensornet # Must come first.
import minienv
# And handle `sensornet`'s globally-exposed data.
```

When `minienv.explored() > .987` at times, your handler handles exploration well.

---

Roughly speaking.

Exploration is about making rare events less rare.

This can be seen as either the need to boldly go where we have not been before ([implicit ](http://www.cs.cornell.edu/~helou/IMRL.pdf)[rewards](https://lilianweng.github.io/lil-log/2020/06/07/exploration-strategies-in-deep-reinforcement-learning.html)), or the need to make all probabilities equal ([entropy ](https://arxiv.org/abs/2005.00820)[regularization](https://paperswithcode.com/method/entropy-regularization), [trying ](https://arxiv.org/abs/1806.09605)[every](https://arxiv.org/abs/1504.04909)[thing](https://icml.cc/media/Slides/icml/2019/halla(10-09-15)-10-09-15-4336-recent_advances.pdf), etc).

Prioritizing the unknown is just asking for [catastrophic ](https://en.wikipedia.org/wiki/Catastrophic_interference)[forgetting](https://arxiv.org/abs/1612.00796), so compressing future possibilities to make them equally-likely (if its compressed future features are picked randomly) is where it's at.

In machine learning, [self-](https://en.wikipedia.org/wiki/Unsupervised_learning)[supervised learning](https://en.wikipedia.org/wiki/Self-supervised_learning) is all about compressing data into an AI [model](https://www.tensorflow.org/text/tutorials/transformer). Try its best [met](https://arxiv.org/abs/2006.07733)[hods.](https://paperswithcode.com/method/swav)

---

So what exactly does this environment make its agents do?

Please use general approaches, and do not overfit to minienv's nature. You don't need to know it. (Read the source code if you're curious.)