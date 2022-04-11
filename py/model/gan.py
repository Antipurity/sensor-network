"""
TODO:
"""



import torch
import torch.nn as nn



class GAN(nn.Module):
    """
    (Conditional) Generative Adversarial Network: a generative model, where a *discriminator* predicts how realistic a sample is, and a *generator* maximizes discriminator's output.

    Methods:
    - `.forward(*args) → sample` or just call: generate. Needs to be conditioned on at least 1 arg.
    - `.pred(*args, sample, reward=1) → loss`: updates the discriminator.
    - `.max(*args, sample, reward=1) → loss`: updates the generator.

    A full GAN update would do `sample = gan(*args)`, then `loss = gan.pred(*args, real, reward=1) + gan.pred(*args, fake, reward=0) + gan.max(*args, fake, reward=1)`.

    Neural nets often generalize poorly outside of training distribution, especially if decision boundaries are sparse there and linear behavior is allowed to drift. You can use a trick from DDPG: have 2 discriminators and combine them by returning the min value.
    """
    def __init__(self, generator, discriminator, noise_sz = 16):
        # TODO: Do we also want the loss for training `discriminator`, L2 by default?
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.noise_sz = noise_sz
    def forward(self, *args):
        """Generates a sample; pass it to `.max(…)`."""
        shape = [*args[0].shape, self.noise_sz]
        noise = torch.randn(shape, device=args[0].device)
        return self.generator(_cat(*args, noise))
    # TODO: `.pred(*args, sample, reward=1)→loss` which makes the GAN want to generate this more.
    # TODO: `.max(*args, sample, reward=1)→loss` which gives gradient to `sample` without adjusting the discriminator.



def _cat(*a, dim=-1): return torch.cat(a, dim)



if __file__ == '__main__':
    pass # TODO: How to test the GAN?