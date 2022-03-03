"""
Printing metrics, with well-performing plotting at the same time.

For example: `log(loss = .53242, whatever = 1.2)`.

Install `matplotlib` for plots.
"""



try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None



import time



_past = {} # For plots.
_widths = {} # For printing.
_allow_printing_at = 0. # matplotlib is slow, so don't take up more than 5% CPU time.
def log(**metrics):
    """Prints numeric values of keyword arguments, both in console and as plots.

    Use this to track metrics of an infinite loop, such as training loss.

    Install `matplotlib` to see those plots."""
    strings = []
    for k in metrics.keys():
        if k not in _past: _past[k] = [None] * len(list(_past.values())[0]) if _past else []
        v = ('%.2f' % metrics[k]) if isinstance(metrics[k], float) else str(metrics[k])
        if k not in _widths or _widths[k] < len(v): _widths[k] = len(v)
        strings.append(k+' '+v.rjust(_widths[k]))
    if len(strings): print('  '.join(strings))
    for k in _past:
        _past[k].append(float(metrics[k]) if k in metrics else None)
    start = time.monotonic()
    global _allow_printing_at
    if plt is not None and start > _allow_printing_at:
        finish(False)
        dur = time.monotonic() - start
        _allow_printing_at = start + dur*19 # 19 = 100%/5% - 1

def finish(final = True):
    """Updates the plots, and allows users to inspect it for as long as they want."""
    if plt is None: return
    plt.cla()
    for k in _past: plt.plot(_past[k], label=k)
    plt.legend()
    plt.pause(.001)
    if final: plt.show()

def clear(max_past_samples = 0):
    """Clears past plots. Optionally, pass in how many most-recent samples to preserve."""
    if max_past_samples:
        for k in _past: _past[k] = _past[k][-max_past_samples:]
    else:
        for k in _past: _past[k].clear()



if __name__ == '__main__': # Test.
    import random
    log(a=3, b=4)
    log(a=30, b=.4)
    log(a=3, b=4)
    for iter in range(100):
        if iter < 30 and random.randint(1,2)==1:
            log(a=random.random(), b=random.random())
        else:
            log(a=random.random(), c=random.random())
    finish()