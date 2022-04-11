"""
Contains the class `GAN` that can generatively model data, or maximize a learned reward signal.
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

    A full GAN update would do `sample = gan(*args)`, then `loss = gan.pred(*args, real, reward=1) + gan.pred(*args, sample, reward=0) + gan.max(*args, sample, reward=1)`.

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
    input_sz, noise_sz, N = 4, 4, 1
    g = nn.Sequential(
        nn.Linear(input_sz + noise_sz, 4*N), nn.LayerNorm(4*N), nn.LeakyReLU(),
        nn.Dropout(),
        nn.Linear(4*N, N),
    )
    d = nn.Sequential(
        nn.Linear(input_sz + N, 4*N), nn.LayerNorm(4*N), nn.LeakyReLU(),
        nn.Linear(4*N, 1),
    )
    gan = GAN(g,d, noise_sz=noise_sz)

    # 2 groups (input) with 3 examples each (output).
    input = torch.randn(2, 1, input_sz).expand(2, 3, input_sz).reshape(6, input_sz)
    output = torch.randn(6, N)
    TEST = torch.randn(6, N, requires_grad=True) # TODO: Use `sample` again after this works (and this *has* to work, right?…).
    opt = torch.optim.SGD([TEST, *gan.parameters()], lr=1e-3, weight_decay=.001)

    for iter in range(10001):
        # TODO: …Try noising the output, maybe?…
        #   …Why does nothing work…
        sample = gan(input)
        TEST = sample
        import random
        l1 = gan.pred(input, output + torch.randn_like(output) * (1 - iter/10000), reward=1)
        l2 = gan.pred(input, TEST, reward=0)
        l3 = gan.max(input, TEST, reward=1)
        # print(TEST.mean().detach().cpu().numpy(), '\t', TEST.std().detach().cpu().numpy()) # TODO:
        # p = random.random();  l4 = gan.pred(input, p*output + (1-p)*sample, reward=p) # TODO: …What if we try learning linear mixes too?… No change…
        l = l1 + l2 + l3
        print(l1.detach().cpu().numpy(), '\t', l2.detach().cpu().numpy(), '\t', l3.detach().cpu().numpy())
        l.backward() # TODO: Why is l3 so bad? Why isn't the GAN learning? What to do when discriminator gets too good?
        if iter % 1000 == 0:
            print('     ', output[..., 0].detach().cpu().numpy(), TEST[..., 0].detach().cpu().numpy()) # TODO: Clearly not learning anything.
        #   …Are GANs *supposed* to be unable to learn super-simplified stuff (since our prior test with diffusion models also failed)?…
        # TODO: How to measure same-group closeness between `sample` and `output`?
        opt.step();  opt.zero_grad()
        # TODO: Run & fix.