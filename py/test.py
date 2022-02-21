"""
This investigates the connection between compression and exploration: whether extracting potential futures in an AI model, preferably well enough to compress them maximally and thus make its representations uniformly-distributed, makes the pasts select those futures more uniformly.

The displayed "picked" future should become equally-divided, or no exploration happens.

Not successful.
"""



import torch
import torch.nn as nn



if __name__ == '__main__':
    # TODO: The picked ratio doesn't change at all... Why?...
    #   Is next-frame prediction not enough?
    #   Did see something like future-regularization with weight_decay=1e-3. Usually has sudden jumps. But it's extremely fragile: weight-decay has to be 1/10 of the learning-rate (which is best 1e-2), and unroll_length has to be 6. (But at least we might possibly have *some* lead.)
    #     Is this a coincidence, or is next-frame prediction just that inefficient at extracting futures?
    #     Either way, it's so non-smooth that it's pretty much unusable for actual exploration.
    # TODO: Cumulative sum?
    # TODO: BYOL?
    # TODO: Mixup?
    batch_size, unroll_length = 64, 6
    dev = 'cpu' # 'cuda' if torch.cuda.is_available() else 'cpu'
    sz, fut_sz, hidden_sz = 2, 1, 8
    futures = torch.randn(2, fut_sz, device=dev)
    picked_0, picked_1 = 0, 0
    losses, picked = [], []
    def future_of(state):
        global picked_0, picked_1
        mask = (state[..., 0] > 1).long()
        with torch.no_grad():
            picked_0 += (1 - mask).sum()
            picked_1 += mask.sum()
        return torch.gather(futures, 0, mask.unsqueeze(-1).expand(mask.shape[0], futures.shape[-1]))
    transition = nn.Sequential(
        nn.Linear(sz+fut_sz, hidden_sz),
        nn.ReLU(),
        nn.LayerNorm(hidden_sz),
        nn.Linear(hidden_sz, sz),
    ).to(dev)
    predict_next_future = nn.Sequential(
        nn.Linear(sz, hidden_sz),
        nn.ReLU(),
        nn.LayerNorm(hidden_sz),
        nn.Linear(hidden_sz, fut_sz),
    ).to(dev)
    optim = torch.optim.SGD([*transition.parameters(), *predict_next_future.parameters()], lr=1e-2, weight_decay=1e-3)
    for iteration in range(50001):
        state = torch.randn(batch_size, sz, device=dev)
        loss = 0
        prev_fut = torch.randn(batch_size, fut_sz, device=dev)
        picked_0, picked_1 = 0, 0
        for unroll in range(unroll_length):
            next_fut = future_of(state)
            state = transition(torch.cat((state, prev_fut), -1))
            fut_prediction = predict_next_future(state)
            loss += (next_fut - fut_prediction).square().mean(0).sum()
            prev_fut = next_fut
        losses.append(loss.detach())
        picked.append(picked_0.detach())
        N = 1000
        if iteration % N == 0:
            I1, I2 = max(0, iteration-N), iteration+1
            P = torch.mean(torch.tensor(picked[I1:I2], dtype=torch.float32))
            print(str(iteration).rjust(6), 'picked:', str(P.cpu().detach().numpy()).rjust(10)+' | '+str((batch_size*unroll_length - P).cpu().detach().numpy()).ljust(10), '\tL2:', torch.mean(torch.tensor(losses[I1:I2])).cpu().detach().numpy())
        loss.backward()
        optim.step()
        optim.zero_grad(True)
    import matplotlib.pyplot as plt
    plt.plot([x.cpu().detach().numpy() for x in losses])
    plt.plot([x.cpu().detach().numpy() for x in picked])
    plt.show()