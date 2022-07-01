Bidirectional-sensor networks for ML: each gathers named numeric data and queries from anywhere, and in a loop, handles that.

Perfectly flexible, [efficient](https://github.com/Antipurity/sensor-network/tree/master/py/sensornet/test.py), [tiny](https://github.com/Antipurity/sensor-network/tree/master/py/sensornet/__init__.py), and [with 100% test coverage](https://github.com/Antipurity/sensor-network/tree/master/py/sensornet/test.py).

Position-invariant and numeric: these constraints free AI models from all concerns about data formats. With only a couple lines of code, you can: learn in completely arbitrary environments, combine digital and analog inputs/outputs, combine tasks at runtime, and perform any machine-learning tasks with a single model (multimodal learning and reinforcement learning and meta-learning and lifelong learning and model distillation; though need a *flawless* implementation that easily handles everything `sn` can bring to bear).

Needs Python 3.5 or newer (mainly for `asyncio` with `await`).

## Installation

```bash
pip install sensornet (TODO: Publish to PyPi.)
```

Or, like, copy this directory, and import it.

Or `pip install -ve .` here, see where your `easy-install.pth` lives, and edit that file to remove the last directory in the path to here.

## Usage

Simple.

First, initialize the handler:

```python
import sensornet as sn
import numpy as np

sn = sn.Handler(8,8,8,8, 64) # Cell-shape: first name-parts, then data.
# OR, simply use the global `sn` as if it's a handler:
sn.cell_shape = (8,8,8,8, 64)
```

Then send/receive data:

```python
async def get():
    sn.set('sensor', 13, 32)
    assert (await sn.get('action', 32)) in range(32)
```

(Simple integer sending/receiving is shown here, but other datatypes are also available.)

And handle it:

```python
@sn.run
async def main():
    fb = None
    while True:
        data, query, error = await sn.handle(fb)
        fb = np.random.rand(query.shape[0], data.shape[1])*2-1
```

This module implements this basic discrete+analog protocol, and does not include anything [else](https://github.com/Antipurity/sensor-network/tree/master/docs/ROADMAP.md) by default, such as string handling or file storage or Internet communication or multiprocessing or integration with ML-libraries apart from PyTorch.

(Implementing a controllable programming language with forking and/or listenable-to data is left as an exercise to the reader.)

---

TODO: A small tour of the datatypes (`sn` just doesn't sound attractive otherwise), hopefully including `@sn.func   async def fn(img: sn.Float(28, 28, dims=2)) -> 10: ...` and/or `sn.Goal()`. And hopefully using an actual ML model to learn.