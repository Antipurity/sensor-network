"""
This file investigates the connection between compression and exploration: whether extracting potential futures in an AI model, preferably well enough to compress them maximally and thus make its representations uniformly-distributed, makes the pasts select those futures more uniformly.

There are 2 possible observations, one much likelier than another. The displayed "picked" observation should become equally-divided, else no exploration is happening.

Although we haven't found a fully-stable gradient-only solution, a BYOL-like solution appears to show some promise: given a `transition: state1 → state2` RNN, have `future_fn: state → future` and `next_future: future → future` functions, and make `next_future(future_fn(state))` predict an updated-via-momentum `future_fn(transition(state))` (important to normalize both). Basically, materialize those futures in a differentiable form, and make their distribution uniform via gradient.

At least, it doesn't fail all the time like next-observation-prediction does. But it is of course impossible to tell whether it's any good in this extremely-limited environment. It's either on the way to a real solution, or just very unstable.
"""



import torch
import torch.nn as nn



if __name__ == '__main__':
    iterations, batch_size, unroll_length = 10000, 64, 6
    dev = 'cpu' # 'cuda' if torch.cuda.is_available() else 'cpu'
    sz, fut_sz, obs_sz, hidden_sz = 2, 1, 1, 32
    obs = torch.randn(2, obs_sz, device=dev)
    obs -= obs.mean()
    obs /= obs.std()
    picked_0, picked_1 = 0, 0
    losses, picked = [], []
    def obs_of(state):
        global picked_0, picked_1
        mask = (state[..., 0] > 1).long()
        with torch.no_grad():
            picked_0 += (1 - mask).sum()
            picked_1 += mask.sum()
        return torch.gather(obs, 0, mask.unsqueeze(-1).expand(mask.shape[0], obs.shape[-1]))

    class SkipConnection(nn.Module):
        def __init__(self, fn):
            super(SkipConnection, self).__init__()
            self.fn = fn
        def forward(self, x): return self.fn(x) + x

    transition = nn.Sequential( # (state, obs) → state
        nn.Linear(sz+obs_sz, hidden_sz),
        nn.ReLU(),
        nn.LayerNorm(hidden_sz),
        nn.Linear(hidden_sz, sz),
    ).to(dev)

    class MomentumCopy(nn.Module):
        def __init__(self, f, momentum = .99):
            import copy
            super().__init__()
            self.f = f
            self.g = copy.deepcopy(f)
            self.fp, self.gp = list(self.f.parameters()), list(self.g.parameters())
            for p in self.gp: p.requires_grad_(False)
            self.momentum = momentum
        def forward(self, x):
            with torch.no_grad():
                for i in range(len(self.fp)):
                    self.gp[i][:] = self.gp[i] * self.momentum + (1-self.momentum) * self.fp[i]
                return self.g(x) # Cannot backprop through it because the in-place modification above does not play nice with backprop.

    class CumSum(nn.Module):
        def __init__(self): super().__init__()
        def forward(self, x): return x.cumsum(-1)

    future = nn.Sequential( # state → future
        nn.Linear(sz, hidden_sz),
        nn.ReLU(),
        nn.LayerNorm(hidden_sz),
        # nn.Linear(hidden_sz, hidden_sz),
        # nn.ReLU(),
        # nn.LayerNorm(hidden_sz),
        nn.Linear(hidden_sz, fut_sz),
        CumSum(), # Doesn't appear to help.
    ).to(dev)
    future_slowed_down = MomentumCopy(future, .999)
    next_future = SkipConnection(nn.Sequential( # future → future
        nn.Linear(fut_sz, hidden_sz),
        nn.ReLU(),
        nn.LayerNorm(hidden_sz),
        # nn.Linear(hidden_sz, hidden_sz),
        # nn.ReLU(),
        # nn.LayerNorm(hidden_sz),
        nn.Linear(hidden_sz, fut_sz),
    )).to(dev)

    optim = torch.optim.SGD([*transition.parameters(), *future.parameters(), *next_future.parameters()], lr=1e-3, weight_decay=0)
    for iteration in range(iterations+1):
        state = torch.randn(batch_size, sz, device=dev)
        loss = 0
        prev_fut = torch.randn(batch_size, obs_sz, device=dev)
        picked_0, picked_1 = 0, 0
        for unroll in range(unroll_length):
            next_fut = obs_of(state)
            prev_state = state
            state = transition(torch.cat((state, prev_fut), -1))
            A = next_future(future(prev_state))
            A = A - A.mean()
            A = A / (A.std()+1e-5)
            with torch.no_grad():
                B = future_slowed_down(state.detach())
                B -= B.mean()
                B /= (B.std()+1e-5)
            loss += (A - B).square().mean(0).sum()
            prev_fut = next_fut
        losses.append(loss.detach())
        picked.append(picked_0.detach())
        N = 1000
        if iteration % N == 0:
            I1, I2 = max(0, iteration-N), iteration+1
            P = torch.mean(torch.tensor(picked[I1:I2], dtype=torch.float32))
            P0 = P.cpu().detach().numpy()
            P1 = (batch_size*unroll_length - P).cpu().detach().numpy()
            print(str(iteration).rjust(6), 'picked:', str(P0).rjust(10)+(' < ' if (P0<P1).all() else ' > ')+str(P1).ljust(10), '\tL2:', torch.mean(torch.tensor(losses[I1:I2])).cpu().detach().numpy())
        loss.backward()
        optim.step()
        optim.zero_grad(True)
    import matplotlib.pyplot as plt
    plt.subplot(1,2,1)
    plt.plot([x.cpu().detach().numpy() for x in losses])
    plt.plot([x.cpu().detach().numpy() for x in picked])
    with torch.no_grad():
        # Show what happens with all the different (2) inputs.
        #   (If we plot the predicted eventual-future against RNN inputs, then it should ideally be equal-area for all potential obs, meaning that the RNN is maximally sensitive to its initial state and thus can explore well in future-space I guess.)
        plt.subplot(1,2,2)
        def fn(xy):
            return future((xy))[..., 0]
        imresolution = 256
        xy = torch.stack([torch.tensor([x, y], dtype=torch.float32)/imresolution*4-2 for y in range(imresolution) for x in range(imresolution)])
        plt.imshow(fn(xy).reshape((imresolution, imresolution)).cpu().numpy(), cmap='hot', extent=(-2,2,-2,2))
    plt.show()