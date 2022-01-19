## Sensor network

Tools for connecting humans and AI models to arbitrary sensors: differentiable [subscribe/publish](https://en.wikipedia.org/wiki/Publish%E2%80%93subscribe_pattern).

It can represent streams of [any](https://github.com/Antipurity/sensor-network/tree/master/docs/universal-environments.md) named observations and actions.

## Design constraints

- Position-invariance, of cells into which data is divided. This enables hotswappable and user-defined observations/actions, which is how humans [expect ](https://en.wikipedia.org/wiki/Process_(computing))[computers ](https://en.wikipedia.org/wiki/USB)[to operate ](https://en.wikipedia.org/wiki/Internet_of_things)[anyway.](https://en.wikipedia.org/wiki/Internet) In ML, [Transformers are currently dominant anyway.](https://arxiv.org/abs/1706.03762)
- -1…1 values, optionally, [mostly for humans.](https://www.reddit.com/r/NoStupidQuestions/comments/65o0gi/how_loud_is_a_nuclear_explosion_all_noise_is/)
- That's all. A human can use it. AGI can use it.

## You can:

- Collect the [re](https://arxiv.org/abs/2102.06701)[q](https://arxiv.org/abs/1712.00409)[ui](https://arxiv.org/abs/2001.08361)[re](https://arxiv.org/abs/2006.10621)[d](https://arxiv.org/abs/2010.14701) data for machine learning. [From a web page.](https://github.com/Antipurity/sensor-network/tree/master/js-lib)
- [Mix-and-match multimodal tasks of variable size](https://arxiv.org/abs/2106.09017), [with 0 or more objectives](https://arxiv.org/abs/2112.15422).
- [Not worry about efficiency.](https://Antipurity.github.io/sensor-network)
- TODO: Fully (& a link to privacy considerations: real-time interaction data is the most precious data that can reasonably be collected about you, so to protect it from interests that are contrary to yours (& links to ads, greed, politics?? Or https://www.singularityweblog.com/technology-reveals/), we better make damn sure that we get this right. For humans, having sovereignty over your own body is [a right](https://en.wikipedia.org/wiki/Bodily_integrity) so fundamental that it's not worth mentioning; we should treat sensor networks with the same standards, and have at least one open-source implementation that is as good as proprietary alternatives.   This sensor-network implementation allows local processing of all data, and encrypted (& what, a link to WebRTC spec?) Internet communication if needed; it does not [depend on exploitation of user data for it to exist](https://en.wikipedia.org/wiki/Big_Tech).) control (& a link to the test page) your computer interaction data (& a link to a file like "Humans control bodies, and live for decades. In contrast, computer programs typically have constrained interfaces, and are destroyed when they outlive their use. With a sensor network for a computer program, equality can be achieved. Though for the privileged 'human' class, gains in equality can be [seen as a personal attack](https://www.techrepublic.com/article/big-data-privacy-is-a-bigger-issue-than-you-think/); please don't go overboard with privacy, and don't confuse "scarily accurate" with "out to get you" without non-emotional reasons, [which there often are](https://www.prnewswire.com/news-releases/privacy-concerns-rise-as-businesses-report-increased-personal-data-collection-301356043.html). TODO: And do we go toward personal sensor networks and how compute growth could make them viable within our lifetimes?"): collect it (& a link to sensors), store it (& a link to the database), extend it over the Internet (& a link to the Internet handler)
- TODO: Maybe, "easily design your own data collection pipeline, and/or integrate arbitrary pipelines in one AI model at the same time"? What would we link to? The test page?
- TODO: What about funky sounds? It definitely needs to be said somewhere to give a solid grounding of what a human's preferred interface is like; but where and how?

## You should eventually be able to:

- Trust in this implementation's production-ready robustness and efficiency.
- Integrate with any OS-level and device-level data.
- Integrate with any machine learning framework and dataset. Reachable performance should not deteriorate significantly, either time or objective.
- Equalize capabilities of humans and machines:
    - Enhance machines, by making data about human activity available, driving the friction and fundamental differences to zero.
    - Enhance humans, [merging](https://towardsdatascience.com/merging-with-ai-how-to-make-a-brain-computer-interface-to-communicate-with-google-using-keras-and-f9414c540a92)[ ](https://en.wikipedia.org/wiki/Transcendent_Man)[with](https://venturebeat.com/2020/11/23/this-is-how-well-merge-with-ai/)[ ](https://www.ft.com/content/0c4fac58-bd15-11e9-9381-78bab8a70848)[AI](https://en.wikipedia.org/wiki/Lovecraftian_horror) (TODO: A link? To what?): having full control over your body and its senses, doing mind transfer and uploading (self-propagation via being predicted TODO: Also a link to a file that expands on this.) and modification (deciding rewards of others and yourself TODO: A link? To what?), [being analyzed in full](https://github.com/likedan/Awesome-CoreML-Models), and effectively replacing the real world with a rainbow of conceptual possibility.
- TODO: Integrate all your data, and extract knowledge and skill from it (& a link to, what, soft mind uploading, and why it's the only realistic approximation to the sci-fi concept, and exponential improvement in compute will take us ever closer to it?).
- TODO: Use an AI model to translate limited human actions (& a link to a small file detailing how humans have eyes and muscles and ears and throats, which drives the design of RGB displays and RGB cameras and speakers and microphones but simultaneously constrains all UI software in a way that humans cannot really understand without experience with freely creating extra senses, whereas Transformers can handle arbitrary data) into arbitrary computer interactions.
    - TODO: "Expand your consciousness via AI translation"?...
    - TODO: "Create, use, and/or train a personal assistant"?...
- TODO: What about uniting AI models, letting them learn from data and each other (AI as the infinite conceptual rainbow)? Can we relate this to a useful thing? Link to nearest-neighbor search, maybe?
    - TODO: ...Would this perform any better than incorporating a differentiable model directly into your own, through newly-learned interfaces, though?
- TODO: What about retraining your AI model [trained on another data input/output](https://github.com/Antipurity/sensor-network/tree/master/docs/universal-environments.md), simply by encoding input and decoding output via randomly-initialized NNs? Maybe even training on several ways at the same time?
- TODO: Above, WHERE ARE THE LINKS THAT JUSTIFY THE NEED FOR EACH AND EVERY ONE OF THEM
- TODO: What other AI capabilities did we want, exactly?
- TODO: ...What about, "Intelligence consists of learning from what you do. Among all optimization methods, model-construction has proven fundamental, even for RL (easier to optimize a reward model than the reward directly) (& some link/s to the rise of DL or even ML?). A differentiable sensor network is a model of intelligent existence, and when closely integrated with humans, may prove to be just as fundamental because it provides new capabilities, eventually (& some link/s to exponentially-improving compute?)."...
    - TODO: ...Is this even really worth saying?
- Take control of an enemy spaceship in seconds like it's sci-fi. Once we've invented spaceships.

Does this sensor network have to visibly implement AGI and the singularity before you care about its refined generality? It has essentially no design constraints, so, [eventua](https://en.wikipedia.org/wiki/The_Singularity_Is_Near)[lly?](https://www.academia.edu/36810724/The_21st_Century_Singularity_and_its_Big_History_Implications_A_re_analysis)