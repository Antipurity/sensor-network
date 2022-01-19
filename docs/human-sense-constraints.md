Humans have eyes, and muscles, and ears, and throats.

This directly drives the development of RGB displays, and RGB cameras, and speakers, and microphones.

But at the same time, it constrains all UI software in a way that humans cannot really understand without experience with freely creating extra senses. (Additionally, human senses carry hidden semantic meaning: eyes get tired, ears deafen, muscle use slowly drains energy.)

[AI has no ](https://arxiv.org/abs/1706.03762)[such limitations.](https://arxiv.org/abs/2107.14795)

It can be used to, at the very least, learn (and/or download) basic knowledge about an interaction before a human needs to [learn it](https://www.quora.com/How-soon-will-we-be-able-to-download-knowledge-into-our-brain-instead-of-going-to-school): like a movie or a videogame that provides musical cues. Less basically, can be used for streamlining control over large amounts of things, where humans *can* do actions but it's inconvenient and non-parallelizable; examples include: direct control of a robot's joints, [gesture](https://arxiv.org/abs/2111.00038) interfaces, giving commands to units in an RTS, using a Lisp REPL without text (AKA executing functions, given a set of possible inputs and mutable cells), or even deciding how to transform displayed data programmatically (for example, what you're reading has a lot of links that could do with being hidden when editing). Usefulness may depend on how much compute is available to train the AI model.

It's nothing new, all already seen in countless demos. A sensor network makes it in principle interoperable without any special effort, because [implementation matters](https://openreview.net/forum?id=r1etN1rtPB).

So the question is, is collecting demos in one model at all important?

[Y](https://arxiv.org/abs/2106.09017)[e](https://arxiv.org/abs/2005.14165)[s](https://www.cs.uic.edu/~liub/lifelong-learning.html)[.](https://journals.sagepub.com/doi/full/10.1177/20539517211047734)

The only real obstacle is that a typical PC doesn't have the compute to run some massive AI model full-time. But AI-training increases in efficiency rapidly, [2…10 ](https://openai.com/blog/ai-and-efficiency/)[times in ](https://venturebeat.com/2020/06/04/ark-invest-ai-training-costs-dropped-100-fold-between-2017-and-2019/)[a year](https://ark-invest.com/articles/analyst-research/ai-training/), as available [compute ](https://www.researchgate.net/figure/The-exponential-progress-of-computing-power-from-1900-to-2013-with-projections-into_fig1_335422453)[doubles](https://www.weforum.org/agenda/2018/09/end-of-an-era-what-computing-will-look-like-after-moores-law/)[ every two years](https://en.wikipedia.org/wiki/Moore%27s_law): what is today a [multi-million-dollar project](https://syncedreview.com/2020/04/30/ai21-labs-asks-how-much-does-it-cost-to-train-nlp-models/) may be doable on a personal computer in a decade. Phones and other devices that need a lasting battery could connect, and [Internet bandwidth is growing too, at 1.5× a year](https://www.nngroup.com/articles/law-of-bandwidth/).