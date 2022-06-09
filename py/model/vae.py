"""
A few very simple utilities for making variational autoencoders (VAEs), namely, `normal` for sampling latents and `make_normal` for regularizing latents.
"""



import torch



def make_normal(mean: torch.Tensor, var: torch.Tensor):
    """Given generative mean & log-of-variance, returns the loss that makes them 0-mean 1-variance over time, so that the distribution can be sampled-from via `.randn_like(x)`. `.sum().backward()` the result.

    VAE's regularization."""
    return mean.square() + var.exp() - var
def normal(mean: torch.Tensor, var: torch.Tensor):
    """Simply samples from a normal distribution with a given `mean` and log-of-`var`iance."""
    return mean + (.5 * var).exp() * torch.randn_like(mean) # More or less.



if __name__ == '__main__':
    N, sz = 16, 32
    data = torch.rand(N, sz) # 0…1
    import torch.nn as nn
    net = nn.Sequential(
        nn.LayerNorm(sz), nn.ReLU(), nn.Linear(sz, sz),
        nn.LayerNorm(sz), nn.ReLU(), nn.Linear(sz, 2*sz),
    )
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(100):
        out = net(data) + 1 # `+1` to show the effects of regularization better.
        mean, var = out[:, :sz], out[:, sz:]
        make_normal(mean, var).sum().backward()
        opt.step();  opt.zero_grad(True)
        with torch.no_grad():
            print('mean mean', mean.mean().cpu().numpy(), 'var mean', var.exp().mean().cpu().numpy())
            #   (0-mean 1-variance. Obviously. Why did this even need to be a test?)