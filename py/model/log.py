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
_subplots = {} # For plots.
_widths = {} # For printing.
_allow_printing_at = 0. # matplotlib is slow, so don't take up more than 5% CPU time.
def log(subplot=0, do_print=True, **metrics):
    """Prints numeric values of keyword arguments, both in console and as plots.

    Use this to track metrics of an infinite loop, such as training loss.

    Install `matplotlib` to see those plots."""
    assert isinstance(subplot, int) and (subplot == 0 or (subplot-1) in _subplots)
    strings = []
    if subplot not in _subplots: _subplots[subplot] = set()
    for k in metrics:
        if k not in _past: _past[k] = [None] * len(list(_past.values())[0]) if _past else [] # Should be the first in this subplot, not the first everywhere.
        v = ('%.2f' % metrics[k]) if isinstance(metrics[k], float) else str(metrics[k])
        if k not in _widths or _widths[k] < len(v): _widths[k] = len(v)
        strings.append(k+' '+v.rjust(_widths[k]))
        for subplot_keys in _subplots.values(): subplot_keys.discard(k)
        _subplots[subplot].add(k)
    if do_print and len(strings): print('  '.join(strings))
    for k in _subplots[subplot]:
        _past[k].append(float(metrics[k]) if k in metrics else None)
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
    if plt is None: return
    for i, ks in _subplots.items():
        plt.subplot(len(_subplots), 1, i+1)
        plt.cla()
        for k in ks:
            plt.plot(_past[k], label=k)
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
    # TODO: Test subplots.
    import random
    log(a=3, b=4)
    log(a=.3, b=.4)
    log(a=3, b=4)
    for iter in range(100):
        if iter < 30 and random.randint(1,2)==1:
            log(a=random.random(), b=random.random())
            log(1, False, d=random.random()*100, e=random.random()*100)
        else:
            log(a=random.random(), c=random.random())
            log(1, False, d=random.random()*100, f=random.random()*100)
    # for iter in range(100): # TODO: Why does intertwining them not work?
    #     if iter < 30 and random.randint(1,2)==1:
    #         log(1, False, d=random.random()*100, e=random.random()*100)
    #     else:
    #         log(1, False, d=random.random()*100, f=random.random()*100)
    finish()