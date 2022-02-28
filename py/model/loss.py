"""
Losses like L2 (`(A - B).square().sum()`) or cross-entropy (`(A * B.log()).sum()`) are nice and all.

But even solved problems are never solved. [Evidence that some losses could be better is starting to accumulate.](https://arxiv.org/abs/2103.03230)
"""



import torch
import torch.nn as nn



class CrossCorrelationLoss(nn.Module):
    """
    `CrossCorrelationLoss(axis=0)(x, y)`

    Basically `(x.norm(axis) @ y.norm(axis).t() - eye(x.shape[axis])).square().mean(axis).sum()`.

    A differentiable dissimilarity metric that makes [normalized cross-](https://github.com/rogerberm/pytorch-ncc)[correlation](https://en.wikipedia.org/wiki/Cross-correlation) (or pairwise [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) if done on matrices) as close to the identity matrix as posssible when minimized (`.backward()`).

    Why: non-pairwise losses can't enforce the lack of redundancy, which makes predicting your own output just turn it into a constant. But this loss does enforce that, [making it usable for self-distillation.](https://arxiv.org/abs/2103.03230)

    Parameters:
    - `axis = 0`: TODO:
    - TODO:
    """
    def __init__(self, axis=0): # TODO: What args do we want?
        # TODO: The correlation-removal-strength coefficient.
        # TODO: The shuffle-invariant mode (which makes targets 1s wherever they are the max on both axes).
        # TODO: A way to choose between L1 and L2 loss.
        super().__init__()
        self.axis = axis
    def forward(self, x, y):
        assert x.shape == y.shape
        while len(x.shape) < 2: x = x.unsqueeze(0);  y = y.unsqueeze(0)
        x = x.transpose(self.axis, -2);  y = y.transpose(self.axis, -2)
        x = (x - x.mean(-2)) / (x.std(-2) + 1e-5)
        y = (y - y.mean(-2)) / (y.std(-2) + 1e-5)
        cc = torch.matmul(x, y.t())
        return (cc - torch.eye(cc.shape[-2], cc.shape[-1])).square().mean(-2).sum()



if __name__ == '__main__':
    pass
    # TODO: Tests of all parameter configurations, especially that a shuffle-invariant loss always converges to a shuffled identity matrix.