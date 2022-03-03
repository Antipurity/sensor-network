"""
TODO:
"""



try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


_widths = {}
def log(**metrics):
    """Prints keyword arguments, both in console and as plots. Use this to track metrics, such as training loss."""
    strings = []
    for k in metrics.keys():
        v = ('%.2f' % metrics[k]) if isinstance(metrics[k], float) else str(metrics[k])
        if k not in _widths or _widths[k] < len(v): _widths[k] = len(v)
        strings.append(k+' '+v.rjust(_widths[k]))
    # TODO: How to update a relevant matplotlib plot if `plt is not None`? (Only occasionally too, to not strain resources.)
    if len(strings): print('  '.join(strings))



if __name__ == '__main__':
    import random
    log(a=3, b=4)
    log(a=30, b=.4)
    log(a=3, b=4)
    for _ in range(100):
        log(a=random.random(), b=random.random())
    pass
# TODO: Also want a basic test of what this can do.