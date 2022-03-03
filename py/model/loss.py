"""
Losses like L2 (`(A - B).square().sum()`) or cross-entropy (`(A * B.log()).sum()`) are nice and all.

But even solved problems are never solved. [Evidence that some losses could be better is starting to accumulate.](https://arxiv.org/abs/2103.03230)
"""



import torch
import torch.nn as nn



class CrossCorrelationLoss(nn.Module):
    """
    `CrossCorrelationLoss(axis = -1, decorrelation_strength = 1., ...)(x, y)`

    By default, basically `(x.norm(-2) @ y.norm(-2).t() / x.shape[-2] - eye(x.shape[-2])).square().mean(-2).sum()`.

    A differentiable dissimilarity metric that makes [normalized cross-](https://github.com/rogerberm/pytorch-ncc)[correlation](https://en.wikipedia.org/wiki/Cross-correlation) (or pairwise [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) if done on matrices) as close to the identity matrix as possible when minimized (`.backward()`).

    Why: non-pairwise losses like `(x-y).abs().sum()` can't enforce the lack of redundancy, which makes predicting your own output just turn it into a constant. But this loss does enforce that, [making it usable for self-distillation.](https://arxiv.org/abs/2103.03230)

    PyTorch-specific.

    Parameters:
    - `axis = -1`: the dimension to compute similarity over. Inputs are put through `x → x.transpose(axis, -2)` before the matrix-multiplication.
    - `decorrelation_strength = .1`: the loss target is separated into `1`s and `0`s, and the loss to `0`s is multiplied by this. The closer to `0`, the more important actual prediction is.
    - `prediction = 'l2'`: [either `'l1'` or `'l2'`.](https://machine-learning-note.readthedocs.io/en/latest/basic/loss_functions.html)
    - `shuffle_invariant = False`: could be useful if the inputs were computed by a Transformer. If `True`, the target cross-correlation of a sequence of random vectors with itself (an identity matrix) is computed with its shuffle instead, such that it is closest to the actual cross-correlation.
    - `also_return_l2 = False`: if `True`, each call's result is not the scalar `loss` but the tuple-of-scalars `(loss, L2)`. Useful for debugging: since the actual cross-correlation loss often gets stuck at seemingly-high values while learning is still happening, printing the closest L2 post-normalization loss instead may be a better performance metric. (This correctly handles `shuffle_invariant=True`.)
    - `eps = 1e-5`: prevents division-by-0 in normalization, which is `x → (x - x.mean()) / (x.std() + eps)`.

    Notes:
    - `decorrelation_strength > 0` and `prediction = 'l1'`: learning hardly happens, because gradients self-interfere too much.
    - Performs best when both `x` and `y` are differentiable, not one of them.
    - The more data, the better it learns (L2 loss of normalized vectors can get lower).
    - Optimizing with Adam performs much better than with SGD or RMSprop. Gradient magnitude is typically small.
    """
    def __init__(self, axis = -1, decorrelation_strength = .1, prediction = 'l2', shuffle_invariant = False, also_return_l2 = False, eps = 1e-5):
        assert prediction == 'l1' or prediction == 'l2'
        super().__init__()
        self.axis = axis
        self.decorrelation_strength = decorrelation_strength
        self.l2 = prediction == 'l2'
        self.shuffle_invariant = shuffle_invariant
        self.also_return_l2 = also_return_l2
        self.eps = eps
    def forward(self, x, y):
        # Not particularly efficient. But it works.
        assert x.shape == y.shape
        assert x.shape[self.axis] > 1
        axis = self.axis if self.axis >= 0 else self.axis + len(x.shape)
        while len(x.shape) < 2:
            x = x.unsqueeze(0);  y = y.unsqueeze(0);  axis += 1
        x = x.transpose(axis, -2);  y = y.transpose(axis, -2)
        x = (x - x.mean(-2, True)) / (x.std(-2, keepdim=True) + self.eps)
        y = (y - y.mean(-2, True)) / (y.std(-2, keepdim=True) + self.eps)
        cc = torch.matmul(x, y.transpose(-2, -1)) / x.shape[-2]
        mask = self.target_for(cc)
        target = (mask - mask.mean())
        cc = cc - target
        cc = cc.square() if self.l2 else cc.abs()
        L = (mask * cc + self.decorrelation_strength * (1 - mask) * cc).sum()
        if not self.also_return_l2: return L
        indices = mask.argmax(-1)
        while len(indices.shape) > 1: # Just don't deal with 3D+ cc because selection is a bit hard.
            import random # (It's just for debugging, so it shouldn't matter.)
            i = random.randrange(indices.shape[0])
            x, y, indices = x[i], y[i], indices[i]
        y = torch.index_select(y, -2, indices)
        return (L, (x - y).square().sum())
    def target_for(self, cc):
        """Given a cross-correlation matrix, returns the 0|1 target that it should predict. (This is an approximate solution, but it works well enough. Needs `O(N^2*log(N))` time. Might not handle non-power-of-2 sizes as well.)"""
        # Do several max-swaps to approximate global-optimum.
        #   Probably not optimal, but good enough, and deterministic.
        #     (Besides, more optimal solutions seem to change quicker between epochs, making learning more challenging.)
        mask = torch.eye(cc.shape[-2], cc.shape[-1], device = cc.device)
        while len(mask.shape) < len(cc.shape): mask = mask.unsqueeze(0)
        if not self.shuffle_invariant: return mask
        def swap(a, k):
            """Each column in `a` is swapped with its cyclically-shifted-by-`k`-indices-down column."""
            N = a.shape[-1]
            down = torch.cat(list(reversed(torch.split(a, (k, N-k), -1))), -1)
            up = torch.cat(list(reversed(torch.split(a, (N-k, k), -1))), -1)
            which = torch.div(torch.arange(0,N, device=a.device), k, rounding_mode='trunc') % 2 == 0
            return torch.where(which, down, up)
        def max_swap(a, k, cc):
            """Swaps columns of `a` with those shifted by `k`, but only wherever this minimizes the distance from cross-correlation `cc`."""
            # (Minimizing L2 loss seems to result in slightly more instability than maximizing `cc*a`.)
            b = swap(a, k)
            sumA = swap(cc * a, k).sum(-2, True)
            sumA = sumA + swap(sumA, k)
            sumB = swap(cc * b, k).sum(-2, True)
            sumB = sumB + swap(sumB, k)
            return torch.where(sumA > sumB, a, b)
        with torch.no_grad():
            # Halve the offset until we're fine-tuning.
            k = cc.shape[-1] // 2
            while k > 0:
                mask = max_swap(mask, k, cc)
                k = (k+1) // 2
                if k <= 1: break
            return mask



if __name__ == '__main__': # Tests.
    from torch import randn
    from torch.optim import SGD
    CCL = CrossCorrelationLoss

    cells, out_sz = 16, 128
    input, output, loss = randn(cells, out_sz), randn(cells, out_sz), CCL(
        axis=-1,
        decorrelation_strength=.1,
        shuffle_invariant=True,
        also_return_l2=True,
    )
    ccls, l2s = [], []
    fn = nn.Sequential(
        nn.Linear(out_sz, 128),
        nn.ReLU(),
        nn.Linear(128, out_sz),
    )
    fn2 = nn.Sequential(
        nn.Linear(out_sz, 128),
        nn.ReLU(),
        nn.Linear(128, out_sz),
    )
    opt = SGD([*fn.parameters(), *fn2.parameters()], 1e-2)
    for _ in range(200):
        A, B = fn(input), fn2(output)
        L, L2 = loss(A, B)
        L.backward();  opt.step();  opt.zero_grad()

        ccl, l2 = L.detach().cpu().numpy(), L2.detach().cpu().numpy()
        print('norm CCL', str(ccl).ljust(11), '    norm L2', l2)
        ccls.append(float(ccl))
        l2s.append(float(l2))
    import matplotlib.pyplot as plt
    plt.clf()
    plt.subplot(2,1,1, ylabel='Cross-correlation')
    plt.plot(ccls)
    plt.subplot(2,1,2, ylabel='L2 loss')
    plt.plot(l2s)
    plt.pause(1)
    plt.show()