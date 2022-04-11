"""
Contains the class `GAN` that can generatively model data, or maximize a learned reward signal.
"""



import torch
import torch.nn as nn



def L2(a,b): return (a - b).square().sum()
def _cat(*a, dim=-1): return torch.cat(a, dim)



class GAN(nn.Module):
    """
    (Conditional) Generative Adversarial Network: a generative model, where a *discriminator* predicts how realistic a sample is, and a *generator* maximizes discriminator's output. Notoriously hard to train.

    Args:
    - `generator`, `discriminator`: neural nets, from a concatenation of `args` and noise/sample to sample/number.
    - `noise_sz = 16`: size of noise, in the last dimension.
    - `loss = L2`: the loss for the learned loss (`discriminator`), [L2 by default](https://arxiv.org/abs/1611.04076).

    Methods:
    - `.forward(*args) → sample` or just call: generate. Needs to be conditioned on at least 1 arg, to at least establish the batch size.
    - `.pred(*args, sample, reward=1) → loss`: updates the discriminator.
    - `.max(*args, sample, reward=1) → loss`: updates the generator.

    A full GAN update would do `sample = gan(*args)`, then `loss = gan.pred(*args, real, reward=1) + gan.pred(*args, sample, reward=0) + gan.max(*args, sample, reward=1)`. Distributions of real & fake samples must always overlap, so that gradient descent knows where to go.

    Neural nets often generalize poorly outside of training distribution, especially if decision boundaries are sparse there and linear behavior is allowed to drift. You can use a trick from DDPG: have 2 `discriminator`s and combine them by returning the min value.
    """
    def __init__(self, generator, discriminator, noise_sz = 16, loss = L2):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.noise_sz = noise_sz
        self.loss = loss
    def forward(self, *args, noise = ...):
        """Generates a sample; pass it to `.max(…)`."""
        if noise is ...:
            shape = [*args[0].shape[:-1], self.noise_sz]
            noise = torch.randn(shape, device=args[0].device)
        return self.generator(_cat(*args, noise))
    def pred(self, *args, reward=1):
        """Simply updates the discriminator to output `reward`. `.detach()` inputs if they should not be updated."""
        return self.loss(self.discriminator(_cat(*args)), reward)
    def max(self, *args, reward=1):
        """Freezes the discriminator, then makes it output `reward` by adjusting the args."""
        for p in self.discriminator.parameters(): p.requires_grad_(False)
        L = self.loss(self.discriminator(_cat(*args)), reward)
        for p in self.discriminator.parameters(): p.requires_grad_(True)
        return L



if __name__ == '__main__':
    # GANs on a super-simple dataset.
    #   (Using the "min distance to any data point" loss is most-efficient;
    #     learning that loss is second-most-efficient;
    #     and the GAN's way of "we're not there yet" is often unstable but often works.)
    input_sz, noise_sz, N = 4, 4, 4
    g = nn.Sequential(
        # (LayerNorm *after* ReLU improves performance of GANs, compared to *before*.)
        nn.Linear(input_sz + noise_sz, 4*N), nn.LeakyReLU(), nn.LayerNorm(4*N),
        nn.Linear(4*N, N),
    )
    d = nn.Sequential(
        nn.Linear(input_sz + N, 4*N), nn.LeakyReLU(), nn.LayerNorm(4*N),
        nn.Linear(4*N, 1),
    )
    gan = GAN(g,d, noise_sz=noise_sz)

    # 2 groups (input) with 3 examples each (output).
    input = torch.randn(2, 1, input_sz).expand(2, 3, input_sz).reshape(6, input_sz)
    output = torch.randn(6, N)
    opt = torch.optim.Adam(gan.parameters(), lr=1e-3, weight_decay=.0)

    def min_L2(sample, output, groups=2):
        sample = sample.reshape((groups, sample.shape[0]//groups, *sample.shape[1:]))
        output = output.reshape((groups, output.shape[0]//groups, *output.shape[1:]))
        return (output.unsqueeze(2) - sample.unsqueeze(1)).square().sum(-1).min(1)[0].sum(0)
    from log import log, finish

    for iter in range(20001):
        sample = gan(input) # (We seem to usually collapse into generating a single output, though.)
        with torch.no_grad():
            log(0, False, L2=min_L2(sample, output).sum().cpu().numpy())
        l1 = gan.pred(input, output, reward=0)
        # l2 = gan.pred(input, sample, reward=min_L2(sample, output))
        l2 = gan.pred(input, sample, reward=1)
        l3 = gan.max(input, sample, reward=0)
        # l1, l2, l3 = 0, min_L2(sample, output).sum(), 0
        (l1 + l2 + l3).backward()
        if iter % 5000 == 0:
            print('     ', output[..., 0].detach().cpu().numpy(), sample[..., 0].detach().cpu().numpy())
        opt.step();  opt.zero_grad()
    finish()