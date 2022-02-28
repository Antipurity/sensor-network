"""
Losses like L2 (`(A - B).square().sum()`) or cross-entropy (`(A * B.log()).sum()`) are nice and all.

But even solved problems are never solved. [Evidence that some losses could be better is starting to accumulate.](https://arxiv.org/abs/2103.03230)
"""



import torch
import torch.nn as nn



class CrossCorrelationLoss(nn.Module):
    """
    `CrossCorrelationLoss(axis=0, decorrelation_strength=1., prediction='l2')(x, y)`

    By default, `(x.norm(-2) @ y.norm(-2).t() - eye(x.shape[-2])).square().mean(-2).sum()`.

    A differentiable dissimilarity metric that makes [normalized cross-](https://github.com/rogerberm/pytorch-ncc)[correlation](https://en.wikipedia.org/wiki/Cross-correlation) (or pairwise [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) if done on matrices) as close to the identity matrix as posssible when minimized (`.backward()`).

    Why: non-pairwise losses can't enforce the lack of redundancy, which makes predicting your own output just turn it into a constant. But this loss does enforce that, [making it usable for self-distillation.](https://arxiv.org/abs/2103.03230)

    Parameters:
    - `axis = 0`: the dimension to compute similarity over. Inputs are put through `x → x.transpose(axis, -2)` before the matrix-multiplication.
    - `decorrelation_strength = 1`: the loss target is separated into `1`s and `0`s, and the loss to `0`s is multiplied by this. The closer to `0`, the more important actual prediction is.
    - `prediction = 'l2'`: [either `'l1'` or `'l2'`.](https://machine-learning-note.readthedocs.io/en/latest/basic/loss_functions.html)
    - TODO:
    """
    def __init__(self, axis=0, decorrelation_strength = 1., prediction = 'l2'):
        # TODO: The shuffle-invariant mode (which makes targets 1s wherever they are the max on both axes).
        assert prediction == 'l1' or prediction == 'l2'
        super().__init__()
        self.axis = axis
        self.decorrelation_strength = decorrelation_strength
        self.l2 = prediction == 'l2'
    def forward(self, x, y):
        assert x.shape == y.shape
        while len(x.shape) < 2: x = x.unsqueeze(0);  y = y.unsqueeze(0)
        x = x.transpose(self.axis, -2);  y = y.transpose(self.axis, -2)
        x = (x - x.mean(-2)) / (x.std(-2) + 1e-5)
        y = (y - y.mean(-2)) / (y.std(-2) + 1e-5)
        cc = torch.matmul(x, y.transpose(-2, -1))
        target = torch.eye(cc.shape[-2], cc.shape[-1], device = x.device)
        part1 = target * ((cc - 1).square() if self.l2 else (cc - 1).abs())
        part2 = self.decorrelation_strength * (1 - target) * (cc.square() if self.l2 else cc.abs())
        return (part1 + part2).mean(-2).sum()



if __name__ == '__main__':
    pass
    # TODO: Tests of all parameter configurations, especially that a shuffle-invariant loss always converges to a shuffled identity matrix.
    # TODO: Test decorrelation_strength.
    # TODO: Test L1/L2.