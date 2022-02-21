"""
This investigates the connection between compression and exploration: whether extracting potential futures in an AI model, preferably well enough to compress them maximally and thus make its representations uniformly-distributed, makes the pasts select those futures more uniformly.

The displayed "picked" observation should become equally-divided, else no exploration is happening.

Not successful, at least so far.
"""



import torch
import torch.nn as nn



if __name__ == '__main__':
    # TODO: The picked ratio doesn't change at all... Why?...
    #   Is next-frame prediction not enough?
    #   Did see something like future-regularization with weight_decay=1e-3. Usually has sudden jumps. But it's extremely fragile: weight-decay has to be 1/10 of the learning-rate (which is best 1e-2), and unroll_length has to be 6. (But at least we might possibly have *some* lead.)
    #     Is this a coincidence, or is next-frame prediction just that inefficient at extracting futures?
    #     Either way, it's so non-smooth that it's pretty much unusable for actual exploration.
    #     Having done more visualizations & runs, pretty sure it's just spurious correlation.
    # TODO: Cumulative sum?
    # TODO: BYOL?
    # TODO: Mixup?
    batch_size, unroll_length = 64, 6
    dev = 'cpu' # 'cuda' if torch.cuda.is_available() else 'cpu'
    sz, fut_sz, obs_sz, hidden_sz = 2, 1, 1, 64
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
    predict_next_obs = nn.Sequential( # state → obs
        nn.Linear(sz, hidden_sz),
        nn.ReLU(),
        nn.LayerNorm(hidden_sz),
        nn.Linear(hidden_sz, obs_sz),
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

    future = nn.Sequential( # state → future
        nn.Linear(sz, hidden_sz),
        nn.ReLU(),
        nn.LayerNorm(hidden_sz),
        nn.Linear(hidden_sz, hidden_sz),
        nn.ReLU(),
        nn.LayerNorm(hidden_sz),
        nn.Linear(hidden_sz, fut_sz),
    ).to(dev)
    future_slowed_down = MomentumCopy(future, .99)
    next_future = SkipConnection(nn.Sequential( # future → future
        nn.Linear(fut_sz, hidden_sz),
        nn.ReLU(),
        nn.LayerNorm(hidden_sz),
        nn.Linear(hidden_sz, hidden_sz),
        nn.ReLU(),
        nn.LayerNorm(hidden_sz),
        nn.Linear(hidden_sz, fut_sz),
    )).to(dev)
    # TODO: What about predicting a 'future' representation at each state, and making that 'future' predict its own future one/many steps in the future (after `next_future`) (normalized, to hopefully make futures equally-spaced)?

    optim = torch.optim.SGD([*transition.parameters(), *predict_next_obs.parameters(), *future.parameters(), *next_future.parameters()], lr=1e-3, weight_decay=1e-4)
    for iteration in range(5001): # TODO:
        state = torch.randn(batch_size, sz, device=dev)
        loss = 0
        prev_fut = torch.randn(batch_size, obs_sz, device=dev)
        picked_0, picked_1 = 0, 0
        for unroll in range(unroll_length):
            next_fut = obs_of(state)
            prev_state = state
            state = transition(torch.cat((state, prev_fut), -1))
            # fut_prediction = predict_next_obs(state) # TODO: Maybe try on the previous state, not the next state? ...Definitely much worse. ...Or maybe it's just having only 1 number for state making it much worse...
            # loss += (next_fut - fut_prediction).square().mean(0).sum()
            # TODO: Below is basically-BYOL. Find out whether we can make it work in any way.
            A = next_future(future(prev_state))
            A = A - A.mean() # TODO: Why does removing this (for both A and B) make loss explode?
            #   TODO: ...Even with just `A` unnormalized, the predicted values are in hundreds-of-millions (at least the dividing lines are centered at 0,0)... How could this happen, even? Is the RNN transition to blame?...
            A = A / (A.std()+1e-5)
            with torch.no_grad():
                B = future_slowed_down(state.detach())
                B -= B.mean() # TODO: Maybe just leave this in for B, but not for A?
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
        #   (If we plot the predicted eventual-obs against RNN inputs, then it should ideally be equal-area for all potential obs.)
        #   ...Currently, basically random. Meaning that we have a chance of improving it?
        #     (...And, currently, variations in state basically never lead to changing the predicted obs.)
        plt.subplot(1,2,2)
        def fn(xy):
            return future((xy))[..., 0]
        imresolution = 256
        xy = torch.stack([torch.tensor([x, y], dtype=torch.float32)/imresolution*4-2 for y in range(imresolution) for x in range(imresolution)])
        plt.imshow(fn(xy).reshape((imresolution, imresolution)).cpu().numpy(), cmap='hot', extent=(-2,2,-2,2))
    plt.show()