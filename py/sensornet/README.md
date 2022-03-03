Bidirectional-sensor networks for ML: each gathers named numeric data and queries from anywhere, and in a loop, handles that.

This is the core protocol: as flexible as you can imagine, [as fast as you need](https://github.com/Antipurity/sensor-network/tree/master/py/sensornet/test.py), [as small as is reasonable](https://github.com/Antipurity/sensor-network/tree/master/py/sensornet/__init__.py), and [as well-tested as you can measure](https://github.com/Antipurity/sensor-network/tree/master/py/sensornet/test.py).

Position-invariant and numeric, nothing is fixed: these constraints free AI models from all concerns about data formats. With only a couple lines of code, you can: learn in completely arbitrary environments, combine tasks at runtime, and perform cross-dataset meta-learning and multimodal learning and multi-objective reinforcement learning and lifelong learning and model distillation.

Python 3.5 or newer (for `asyncio` with `await`).

## Installation

```bash
pip install sensornet
```

TODO: Publish to PyPi.

Or, like, copy this directory.

Or `pip install -ve .` here, see where your `easy-install.pth` lives, and edit that file to remove the last directory in the path to here.

## Usage

Simple.

First, initialize the handler:

```python
import sensornet as sn
import numpy as np

h = sn.Handler((8, 24, 64), 8) # See `sn.Namer` for discussion of cell shapes.
# OR, simply use the global `sn` as if it's a handler:
sn.shape((8, 24, 64), 8)
h = sn
```

Then send/receive data:

```python
def set():
    h.data('name', np.random.rand(32)*2-1)

async def get():
    nums = await h.get('name', 32)
    assert nums.shape == (32,)
```

And handle it:

```python
@sn.run
async def main():
    fb = None
    while True:
        data, query, data_error, query_error = await h.handle(fb)
        fb = np.random.rand(query.shape[0], data.shape[1])*2-1
```

This module implements this basic protocol, and does not include anything [else](https://github.com/Antipurity/sensor-network/tree/master/docs/ROADMAP.md) by default, such as string/image handling or file storage or multiprocessing or Internet communication or integration with ML-libraries.

(Implementing a controllable language with forking and/or listenable-to data, and training an AI model that does something useful there, is left as an exercise to the reader.)

---

## Integrations

- PyTorch: given `import torch`, `sn.torch(torch, tensor)` should be used as the argument to `h.handle(...)`.