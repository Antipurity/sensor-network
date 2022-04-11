"""
TODO:
"""



import torch
import torch.nn as nn



def L2(a,b): return (a - b).square().sum()
def _cat(*a, dim=-1): return torch.cat(a, dim)



class GAN(nn.Module):
    """
    (Conditional) Generative Adversarial Network: a generative model, where a *discriminator* predicts how realistic a sample is, and a *generator* maximizes discriminator's output.

    Args:
    - `generator`, `discriminator`: neural nets, from a concatenation of `args` and noise/sample to sample/number.
    - `noise_sz = 16`: size of noise, in the last dimension.
    - `loss = L2`: the loss for the learned loss (`discriminator`).

    Methods:
    - `.forward(*args) → sample` or just call: generate. Needs to be conditioned on at least 1 arg, to at least establish the batch size.
    - `.pred(*args, sample, reward=1) → loss`: updates the discriminator.
    - `.max(*args, sample, reward=1) → loss`: updates the generator.

    A full GAN update would do `sample = gan(*args)`, then `loss = gan.pred(real, *args, reward=1) + gan.pred(fake, *args, reward=0) + gan.max(fake, *args, reward=1)`.

    Neural nets often generalize poorly outside of training distribution, especially if decision boundaries are sparse there and linear behavior is allowed to drift. You can use a trick from DDPG: have 2 `discriminator`s and combine them by returning the min value.
    """
    def __init__(self, generator, discriminator, noise_sz = 16, loss = L2):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.noise_sz = noise_sz
        self.loss = loss
    def forward(self, *args):
        """Generates a sample; pass it to `.max(…)`."""
        shape = [*args[0].shape, self.noise_sz]
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



if __file__ == '__main__':
    pass # TODO: How to test the GAN?