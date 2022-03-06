"""
Pretraining for all possible goals is like learning a map of how to get from anywhere to anywhere.

We have a square board here, and 1 agent that can only walk it. If we can't learn a map here, how can we expect to learn a map in more complex environments?
"""



# TODO: First, get a handle on diffusion models, because time and again, we've faced the need to make internal representations smooth (or model distributions instead of averages) and were found wanting: have a denoising model (first try without the noise level as an input), and a dataset of like 16 points, and a func for noise-corrupting a point (into like 100 levels, 0 is the original, 100 is normal-noise), and a func for running the denoising model and predicting the prev-level (all levels at once, shifting the corruptions), and a func for going from noise to a generated sample, and a func for printing the L2 loss from a generated sample to its nearest dataset neighbor.
#   TODO: Second, understand whether we can impose a denoising loss not on data points directly, but on functions from them.
# Generative modeling `datapoints` via diffusion models.
import torch
import torch.nn as nn
device = 'cuda' if torch.cuda.is_available() else 'cpu'
sz, hidden_sz = 32, 128
datapoints = torch.randn(8, sz, device=device)



import torchvision
import torchvision.transforms as transforms
transform = transforms.ToTensor()
loader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR10(root='./TEMP', train=True, download=True, transform=transform),
    batch_size=64,
    shuffle=True,
)



def norm(x, axis=None, eps=1e-5):
    if axis is not None:
        return (x - x.mean(axis, True)) / (x.std(axis, keepdim=True) + eps)
    else:
        return (x - x.mean()) / (x.std() + eps)
class SkipConnection(nn.Module):
    def __init__(self, fn): super().__init__();  self.fn = fn
    def forward(self, x): return x + self.fn(x)
denoise = nn.Sequential(
    # SkipConnection(nn.Sequential(
        nn.Linear(sz, hidden_sz), # First trying without the noise level as an input. # TODO:
        nn.LayerNorm(hidden_sz),
        nn.Softsign(),
        nn.Linear(hidden_sz, hidden_sz),
        nn.LayerNorm(hidden_sz),
        nn.Softsign(),
        nn.Linear(hidden_sz, sz),
    # )),
    # nn.LayerNorm(sz), # TODO:
).to(device)
opt = torch.optim.Adam(denoise.parameters(), lr=1e-3)



def noise(datapoints, future_levels=1, levels = 1000):
    """Returns `(samples_with_noise, samples_with_less_noise)`; predict the latter from the former to denoise."""
    # TODO: ...This doesn't seem to be working great... So maybe we should try generating noise step-by-step?...
    indices = torch.arange(0, levels - future_levels + 1, device=device)
    noise_strength = ((indices + future_levels) / levels - 1).exp() # TODO: ...Uhh, is this the correct amount, really?
    denoise_strength = (indices / levels - 1).exp()
    while len(noise_strength.shape)-1 < len(datapoints.shape):
        noise_strength = noise_strength.unsqueeze(-1)
        denoise_strength = denoise_strength.unsqueeze(-1)
    noise = torch.randn(indices.shape[0], *datapoints.shape, device=device)
    noised = (1-noise_strength) * datapoints + noise_strength * noise
    denoised = (1-denoise_strength) * datapoints + denoise_strength * noise
    return noised, denoised
    # TODO: ...What if we trained not one-transition-at-a-time, but generate()=datapoints... This would make denoising-speed be able to be learned, if it even works...
    #   ...Maybe even lowest_distance(generate(datapoints.shape[0]), datapoints)=0?
def corrupt(datapoints, denoise, levels=1000):
    L = 0
    noise = datapoints
    for i in range(levels):
        prev, noise = noise, (levels-1)/levels*noise + torch.randn_like(noise)/levels
        denoised = norm(noise)
        for j in range(i+1): denoised = denoise(denoised)
        L = L + (datapoints - denoised).square().sum()
        # TODO: ...Maybe also try ensuring that many-denoisings converge to the target image, in O(n^2)?... (Because sampling doesn't *seem* to be working out well, possibly because the errors accumulate.)
    return L
def generate(n=1, levels = 1000):
    noise = torch.randn(n, sz, device=device)
    # indices = torch.arange(0, levels, device=device)
    # noise_strength = ((indices + 1) / levels - 1).exp()
    for i in range(levels):
        # noise = denoise(torch.cat((noise, noise_strength[levels-i-1].expand(*noise.shape[:-1], 1)), -1)) # Noise level as an input.
        noise = denoise(noise)
    return noise
def lowest_distance(sample, datapoints):
    return (sample - datapoints).square().sum(-1).min(-1)[0]
def closest_datapoint(sample, datapoints):
    return (sample - datapoints).square().sum(-1).argmin(-1)
# for images, labels in loader:
#     noised, denoised_target = noise(datapoints, 10)
#     denoised = denoise(noised)
#     loss = (denoised_target - denoised).square().sum()
#     loss.backward()
#     opt.step();  opt.zero_grad()
#     with torch.no_grad():
#         sample = generate(1, 10)
#         gen_loss = lowest_distance(sample, datapoints) # TODO: ...Why isn't this going down... What did we do wrong?
#         #   ...Do we really need to input that noise level...
#         #   ...Why are we still failing...
#         print('L2', str(loss.detach().cpu().numpy()).ljust(11), 'sample', str(gen_loss.detach().cpu().numpy()).ljust(11), datapoints[closest_datapoint(sample, datapoints)].detach().cpu().numpy(), sample.detach().cpu().numpy())
for iter in range(50000):
    import random
    # noised, denoised_target = noise(datapoints, random.randint(1,10), 10)
    # denoised = denoise(noised)
    # loss = (denoised_target - denoised).square().sum()
    loss = corrupt(torch.cat((datapoints, datapoints, datapoints, datapoints), 0), denoise, 10)
    # samples = generate(datapoints.shape[0], random.randint(1, 10)).unsqueeze(-2)
    # loss = lowest_distance(samples, datapoints).sum() # TODO: ...But this method collapses diversity... How to not?...
    loss.backward()
    opt.step();  opt.zero_grad()
    with torch.no_grad():
        sample = generate(1, 10)
        gen_loss = lowest_distance(sample, datapoints) # TODO: ...Why isn't this going down... What did we do wrong?
        #   ...Do we really need to input that noise level...
        #   ...Why are we still failing...
        # print('L2', str(loss.detach().cpu().numpy()).ljust(11), 'sample', str(gen_loss.detach().cpu().numpy()).ljust(11), datapoints[closest_datapoint(sample, datapoints)].detach().cpu().numpy(), sample.detach().cpu().numpy())
        print('L2', str(loss.detach().cpu().numpy()).ljust(11), 'sample', str(gen_loss.detach().cpu().numpy()).ljust(11))