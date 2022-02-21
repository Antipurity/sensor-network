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
    #     Having done more visualizations & runs, pretty sure it's just spurious correlation.
    # TODO: Cumulative sum?
    # TODO: BYOL?
    # TODO: Mixup?
    batch_size, unroll_length = 64, 6
    dev = 'cpu' # 'cuda' if torch.cuda.is_available() else 'cpu'
    sz, fut_sz, hidden_sz = 1, 1, 64
    futures = torch.randn(2, fut_sz, device=dev)
    futures -= futures.mean()
    futures /= futures.std()
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
        # nn.LayerNorm(hidden_sz),
        nn.Linear(hidden_sz, sz),
    ).to(dev)
    predict_next_future = nn.Sequential(
        nn.Linear(sz, hidden_sz),
        nn.ReLU(),
        # nn.LayerNorm(hidden_sz),
        nn.Linear(hidden_sz, fut_sz),
    ).to(dev)
    # TODO: What about predicting a 'future' representation at each state, and making that 'future' predict its own future one/many steps in the future?
    #   For stability, do we want the target-future to be computed via a momentum-net?
    #   Do we want to artificially constrain the number of possible futures to 2, then think of how a soft constraint can be done?
    #   What other things do we want to include to maybe make this work? Maybe a "past-future → future-future" NN predictor (which would kinda allow us to extract the variance of the "go to next step" augmentation, and would pretty much make this method into BYOL)?
    #     Knowing the future allows us to affect it, and make its distribution uniform... What mechanism do we make the distribution uniform with? Do we make it 1-number and sharpen it or smth, or subtract mean and divide by stdev?...
    #   ...With these 'future' predictors, why do we need a separate RNN? Don't these predictors form an RNN already?
    optim = torch.optim.SGD([*transition.parameters(), *predict_next_future.parameters()], lr=1e-3, weight_decay=1e-4)
    for iteration in range(50001): # TODO:
        state = torch.randn(batch_size, sz, device=dev)
        loss = 0
        prev_fut = torch.randn(batch_size, fut_sz, device=dev)
        picked_0, picked_1 = 0, 0
        for unroll in range(unroll_length):
            next_fut = future_of(state)
            state = transition(torch.cat((state, prev_fut), -1)) # TODO: ...Try without `state`, like that's going to change anything?... (It should theoretically make it not work at all.)
            fut_prediction = predict_next_future(state) # TODO: Maybe try on the previous state, not the next state? ...Definitely much worse. ...Or maybe it's just having only 1 number for state making it much worse...
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
    plt.subplot(1,2,1)
    plt.plot([x.cpu().detach().numpy() for x in losses])
    plt.plot([x.cpu().detach().numpy() for x in picked])
    with torch.no_grad():
        # Show what happens with all the different (2) inputs.
        #   (If we plot the predicted eventual-future against RNN inputs, then it should ideally be equal-area for all potential futures.)
        #   ...Currently, basically random. Meaning that we have a chance of improving it?
        #     (...And, currently, variations in state basically never lead to changing the predicted future.)
        plt.subplot(1,2,2)
        def fn(xy):
            return predict_next_future(transition(xy))
        imresolution = 256
        xy = torch.stack([torch.tensor([x, y], dtype=torch.float32)/imresolution*4-2 for y in range(imresolution) for x in range(imresolution)])
        plt.imshow(fn(xy).reshape((imresolution, imresolution)).cpu().numpy(), cmap='hot', extent=(-2,2,-2,2))
    plt.show()