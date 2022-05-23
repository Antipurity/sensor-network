"""
Printing metrics, with well-performing plotting at the same time.

For example: `log(loss = .53242, whatever = 1.2)`.

Install `matplotlib` for plots.

Has optional (`torch` is an arg to `log`) integration with PyTorch, for logging its tensors.
"""



try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None



import time



def cpu(x, torch):
    """PyTorch integration, providing GPU→CPU async transfer, usable as `await cpu(x)`. (Since PyTorch doesn't make this easy.)"""
    if torch is None: return x
    if not isinstance(x, torch.Tensor) or not x.is_cuda:
        x = x.detach().numpy() if isinstance(x, torch.Tensor) else x
        return x
    with torch.no_grad():
        assert x.numel() == 1
        # https://discuss.pytorch.org/t/non-blocking-device-to-host-transfer/42353/2
        result = torch.zeros_like(x, layout=torch.strided, device='cpu', memory_format=torch.contiguous_format)
        result.copy_(x, non_blocking=True)
        result = result.numpy()
        event = torch.cuda.Event()
        event.record()
        _events.append(event)
        return result



_past = {} # For plots.
_subplots = {} # For plots.
_widths = {} # For printing.
_allow_printing_at = 0. # matplotlib is slow, so don't take up more than 5% CPU time.
_events = [] # For async GPU→CPU transfer.
def _wait_for_events():
    for ev in _events:
        while not ev.query():
            time.sleep(.003)
    _events.clear()
def _new_list():
    example = None
    for values in _past.values():
        if isinstance(values, list):
            example = values
            break
    return [None] * len(example) if example is not None else []
def log(subplot=0, do_print=True, torch=None, **metrics):
    """Prints numeric values of keyword arguments, both in console (if `do_print` is unspecified or `True`) and as plots.

    Use this to track metrics of an infinite loop, such as training loss.

    Install `matplotlib` to see those plots.

    Instead of a numeric value, can log a function such as `lambda plt, key: plt.imshow(x)`."""
    assert isinstance(subplot, int) and (subplot == 0 or (subplot-1) in _subplots)
    strings = []
    if subplot not in _subplots: _subplots[subplot] = set()
    for k in metrics:
        metrics[k] = cpu(metrics[k], torch)
        if k not in _past or not callable(metrics[k]) and not isinstance(_past[k], list):
            _past[k] = _new_list() # Should be the first in this subplot, not the first everywhere.
        if not callable(metrics[k]):
            v = ('%.2f' % metrics[k]) if isinstance(metrics[k], float) else str(metrics[k])
            if k not in _widths or _widths[k] < len(v): _widths[k] = len(v)
            strings.append(k+' '+v.rjust(_widths[k]))
        for subplot_keys in _subplots.values(): subplot_keys.discard(k)
        _subplots[subplot].add(k)
    if do_print and len(strings): print('  '.join(strings))
    for k in _subplots[subplot]:
        if k not in metrics:
            _past[k].append(0.)
        elif not callable(metrics[k]):
            _past[k].append(float(metrics[k]) if k in metrics else None)
        else:
            _past[k] = metrics[k]
    start = time.monotonic()
    global _allow_printing_at
    if plt is not None and start > _allow_printing_at:
        finish(False)
        dur = time.monotonic() - start
        _allow_printing_at = start + dur*19 # 19 = 100%/5% - 1
    if plt is not None:
        plt.pause(.0001)

def finish(final = True):
    """Updates the plots, and allows users to inspect it for as long as they want."""
    _wait_for_events()
    if plt is None: return
    for i, ks in _subplots.items():
        plt.subplot(len(_subplots), 1, i+1)
        plt.cla()
        for k in ks:
            if not callable(_past[k]): plt.plot(_past[k], label=k)
            else: _past[k](plt, k)
        if any(not callable(_past[k]) for k in ks): plt.legend()
    plt.pause(.001)
    if final: plt.show()

def clear(max_past_samples = 0):
    """Clears past plots. Optionally, pass in how many most-recent samples to preserve."""
    if max_past_samples:
        for k in _past:
            if callable(_past[k]): _past[k] = _new_list()
            else: _past[k] = _past[k][-max_past_samples:]
    else:
        for k in _past:
            if callable(_past[k]): _past[k] = []
            else: _past[k].clear()



if __name__ == '__main__': # Test.
    import torch
    torch.set_default_tensor_type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)

    import random
    log(a=3, b=4)
    log(a=.3, b=.4)
    log(a=3, b=4)
    for iter in range(100):
        if iter < 30 and random.randint(1,2)==1:
            log(0, False, a=random.random(), b=random.random())
            log(1, False, torch, d=random.random()*100, e=torch.tensor(random.random())*100)
        else:
            log(0, False, a=random.random(), c=random.random())
            log(1, False, torch, d=random.random()*100, f=torch.tensor(random.random())*100)
    finish()