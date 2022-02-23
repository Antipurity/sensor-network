"""
[Attention is all you need](https://arxiv.org/abs/1706.03762), allegedly.

# TODO: Since we only need self-attention in our models now, kill this file, and use the built-in `torch.nn.MultiheadAttention`.
"""



import torch
import torch.nn as nn



class Attention(nn.Module):
    """
    Simple, quadratic-time quadratic-memory dot-product [attention](https://arxiv.org/abs/1706.03762): `softmax(q @ k.T / sqrt(q_size), -1) @ v`.

    Position-invariant, so encode 'position' data (names) numerically.

    (PyTorch has [`torch.nn.MultiheadAttention`](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html), but it's not only too bloated, but also not bloated enough, since its query size and output size have to be the same.)

    Constructor args:
    - `kv_size`: the size of cells from which keys & values are computed.
    - `q_size = kv_size`: the size of query cells.
    - `output_size = q_size`: the size of resulting cells, one result per query.
    - `heads = 1`: how many attentions to execute in parallel, since just one may not provide enough resolution. Must divide `q_size` and `output_size`.
    - `kv_to_v = torch.nn.Linear(kv_size, output_size, bias=False)`
    - `kv_to_k = torch.nn.Linear(kv_size, q_size, bias=False)`

    Call args:
    - `kv`: the vectors from which keys & values are computed, sized `? × kv_size`.
    - `q = kv`: the query, sized `?? × q_size`, to fill in with values. If not specified, this is self-attention.
    """
    def __init__(self, kv_size, q_size = None, output_size = None, heads = 1, kv_to_v=None, kv_to_k=None):
        if q_size is None: q_size = kv_size
        if output_size is None: output_size = q_size
        assert isinstance(kv_size, int) and isinstance(q_size, int) and isinstance(output_size, int)
        assert q_size % heads == output_size % heads == 0
        super().__init__()
        self.kv_size = kv_size
        self.q_size = q_size
        self.output_size = output_size
        self.heads = heads
        self.qk_mult = 1. / q_size ** .5
        self.kv_to_v = kv_to_v or nn.Linear(kv_size, output_size, bias=False)
        self.kv_to_k = kv_to_k or nn.Linear(kv_size, q_size, bias=False)
    def forward(self, kv, q = None):
        # Transform keys & values from the initial `kv` vector. `q` should be transformed before this if needed.
        if q is None: q = kv
        assert kv.shape[-1] == self.kv_size # These cause warnings with tracing.
        assert q.shape[-1] == self.q_size
        v = self.kv_to_v(kv) # N×kv_size → N×output_size
        k = self.kv_to_k(kv) # N×kv_size → N×q_size
        v = self._pre_multihead(v)
        k = self._pre_multihead(k)
        q = self._pre_multihead(q)
        return self._post_multihead(torch.matmul(nn.functional.softmax(torch.matmul(q, torch.transpose(k, -2, -1)) * self.qk_mult, -1), v))
    def _pre_multihead(self, x):
        if self.heads == 1: return x
        x = x.reshape(*x.shape[:-1], self.heads, int(x.shape[-1]) // self.heads)
        if len(x.shape) >= 3:
            x = torch.transpose(x, -2, -3)
        return x
    def _post_multihead(self, y):
        if self.heads == 1: return y
        if len(y.shape) >= 3:
            y = torch.transpose(y, -2, -3)
        return y.reshape(*y.shape[:-2], y.shape[-2] * y.shape[-1])



if __name__ == '__main__':
    # Test.
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    kv = torch.randn(51, 96, device=dev)
    q = torch.randn(21, 32, device=dev)
    result = torch.randn(21, 96, device=dev)
    model = Attention(kv_size=96, q_size=32, output_size=96, heads=2).to(dev)
    optim = torch.optim.SGD(model.parameters(), lr=1e-3)
    for _ in range(1000):
        output = model(kv, q)
        L = (output - result).square().sum()
        print('L2', L.cpu().detach().numpy())
        L.backward()
        optim.step();  optim.zero_grad(True)
    print('OK' if (L.cpu().detach().numpy() < 1e-7).all() else 'BAD')