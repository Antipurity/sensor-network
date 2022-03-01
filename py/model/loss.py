"""
Losses like L2 (`(A - B).square().sum()`) or cross-entropy (`(A * B.log()).sum()`) are nice and all.

But even solved problems are never solved. [Evidence that some losses could be better is starting to accumulate.](https://arxiv.org/abs/2103.03230)
"""



import torch
import torch.nn as nn



import random # TODO:
import matplotlib.pyplot as plt # TODO:
IM = None # TODO:
class CrossCorrelationLoss(nn.Module):
    """
    `CrossCorrelationLoss(axis = -1, decorrelation_strength = 1., prediction = 'l2', shuffle_invariant = False)(x, y)`

    By default, basically `(x.norm(-2) @ y.norm(-2).t() / x.shape[-2] - eye(x.shape[-2])).square().mean(-2).sum()`.

    A differentiable dissimilarity metric that makes [normalized cross-](https://github.com/rogerberm/pytorch-ncc)[correlation](https://en.wikipedia.org/wiki/Cross-correlation) (or pairwise [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) if done on matrices) as close to the identity matrix as posssible when minimized (`.backward()`).

    Why: non-pairwise losses can't enforce the lack of redundancy, which makes predicting your own output just turn it into a constant. But this loss does enforce that, [making it usable for self-distillation.](https://arxiv.org/abs/2103.03230)

    If printing the loss, be advised, it is frequently almost-stuck at `.9786` or similar, but learning still happens.

    Parameters:
    - `axis = -1`: the dimension to compute similarity over. Inputs are put through `x → x.transpose(axis, -2)` before the matrix-multiplication.
    - `decorrelation_strength = .1`: the loss target is separated into `1`s and `0`s, and the loss to `0`s is multiplied by this. The closer to `0`, the more important actual prediction is.
    - `prediction = 'l2'`: [either `'l1'` or `'l2'`.](https://machine-learning-note.readthedocs.io/en/latest/basic/loss_functions.html)
    - `shuffle_invariant = False`: if `True`, the target identity matrix (cross-correlation of a sequence with itself) is replaced with its closest shuffle. Useful if the inputs were computed by a Transformer.

    Notes:
    - `decorrelation_strength > 0` and `prediction = 'l1'`: learning hardly happens, because gradients self-interfere too much.
    - Performs best when both `x` and `y` are differentiable.
    - The more data, the better it learns (L2 loss of normalized vectors can get lower).
    - Optimizing with Adam seems to perform much better than with SGD or RMSprop.
    """
    def __init__(self, axis = -1, decorrelation_strength = .1, prediction = 'l2', shuffle_invariant = False):
        assert prediction == 'l1' or prediction == 'l2'
        super().__init__()
        self.axis = axis
        self.decorrelation_strength = decorrelation_strength
        self.l2 = prediction == 'l2'
        self.shuffle_invariant = shuffle_invariant
    def forward(self, x, y):
        # Not particularly efficient.
        assert x.shape == y.shape
        assert x.shape[self.axis] > 1
        axis = self.axis if self.axis >= 0 else self.axis + len(x.shape)
        while len(x.shape) < 2:
            x = x.unsqueeze(0);  y = y.unsqueeze(0);  axis += 1
        x = x.transpose(axis, -2);  y = y.transpose(axis, -2)
        eps = 1e-5
        x = (x - x.mean(-2, True)) / (x.std(-2, keepdim=True) + eps)
        y = (y - y.mean(-2, True)) / (y.std(-2, keepdim=True) + eps)
        cc = torch.matmul(x, y.transpose(-2, -1)) / x.shape[-2]
        if not self.shuffle_invariant:
            mask = torch.eye(cc.shape[-2], cc.shape[-1], device = x.device)
        else:
            with torch.no_grad(): # Try to find the closest shuffle.
                # Taking max along one dim may cause repetition and missing entries along the other (which shuffles don't have). So approximate fixing it, iteratively. # TODO: So use a heuristic to quickly fix it. ...Impossible.
                min, max_cand = cc.min(), cc - cc.max()
                mask = (cc - cc.max(-2, True)[0]).sign()+1.
                for i in range(10): # TODO: ...What we want is to find points that are maxima in *both* dimensions, not one of them... Is it correct?... Is it possible to do that directly?...
                    dim, dim2 = -1 if i%2 == 0 else -2, -2 if i%2 == 0 else -1
                    first_solutions_1 = (1. - (mask.cumsum(dim) - mask).sign())
                    first_solutions_2 = (1. - (mask.cumsum(dim2) - mask).sign())
                    mask = mask * first_solutions_1 * first_solutions_2 # Deduplicate.
                    f = mask + (1.-mask) * max_cand # Put fallbacks into the mask.
                    mask = (f - f.max(dim, True)[0]).sign()+1. # Fill.
                    mask = mask.maximum((f - f.max(dim2, True)[0]).sign()+1.) # Fill.
                # TODO: Wait, how did we want to fill in the missing entries, again... Replace 0s with max_cand, and find the max along the dim.
                # TODO: Why is this broken? Can we make it not-broken?
                #   (After all, Transformers are unordered, so can we really just impose ordered CCL on their outputs?)
                # TODO: ...Why does the mask still contain clear duplications along the most-recently-deduplicated dimension?... (Or maybe the other dim.) Is always picking the first max a too-strong prior; but how would we pick a random max?...
        target = (mask - mask.mean())
        if random.randint(1,1) == 1: # TODO: Remove this, after we've debugged shuffle.
            global IM
            if IM is None:
                IM = plt.imshow(mask.detach().cpu().numpy()) # TODO:
            else:
                IM.set_data(mask.detach().cpu().numpy()) # TODO: Looks very wrong. Fix it.
            plt.pause(.01)
        cc = cc - target
        part1 = mask * (cc.square() if self.l2 else cc.abs())
        part2 = self.decorrelation_strength * (1 - mask) * (cc.square() if self.l2 else cc.abs())
        return (part1 + part2).mean(-2).sum()



if __name__ == '__main__': # Tests.
    from torch import randn
    from torch.optim import Adam
    CCL = CrossCorrelationLoss

    cells, out_sz = 16, 128
    input, fn, fn2, output, loss, losses = randn(cells, out_sz), nn.Sequential(
        nn.Linear(out_sz, 128),
        nn.ReLU(),
        nn.Linear(128, out_sz),
    ), nn.Sequential(
        nn.Linear(out_sz, 128),
        nn.ReLU(),
        nn.Linear(128, out_sz),
    ), randn(cells, out_sz), CCL(axis=-1, decorrelation_strength=.1, shuffle_invariant=True), [] # TODO: .1
    opt = Adam([*fn.parameters(), *fn2.parameters()], 1e-4)
    for _ in range(50000): # TODO: 5000
        A, B = fn(input), fn2(output)
        L = loss(A, B)
        A = (A - A.mean(-1, True)) / (A.std(-1, keepdim=True) + 1e-5)
        B = (B - B.mean(-1, True)) / (B.std(-1, keepdim=True) + 1e-5)
        print('norm CCL', str(L.detach().cpu().numpy()).ljust(11), '    norm L2', (A-B).square().sum().detach().cpu().numpy())
        L.backward();  opt.step();  opt.zero_grad()

    # TODO: Test that shuffle-invariance always converges to a `target` matrix with .sum() being x.shape[-2]. (& fix.)