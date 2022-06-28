import random



class Env:
    """
    The simplest 2D torus board: `N×N` board-cells and `4` actions, up/down/left/right. Sometimes, the goal board-cell switches.

    Due to `sn` (robust communication), actions are asynchronous, so models also have to learn to control that. No batch-size, for realism.
    """
    N = 0
    xy = (0,0)
    goal_xy = (0,0)
    def __init__(self, N=16): self.N = N;  self.history = []
    def __call__(self, sn):
        sn.set('xy', self.xy, [self.N] * 2)
        sn.set('xy', self.goal_xy, sn.Goal([self.N] * 2))
        sn.run(self.handle_feedback, sn.query('act', [2,2]))

        if random.random() < 1 / self.N / 5: # Switch goals sometimes.
            self.goal_xy = (random.randrange(self.N), random.randrange(self.N))
    async def handle_feedback(self, fb):
        """Up/down, left/right."""
        direction, which = await fb
        x,y = self.xy
        if which == 0: X = x + direction*2-1
        if which == 1: Y = y + direction*2-1
        self.xy = (X % self.N, Y % self.N)
        self.history.append((x, y, X, Y))
        if len(self.history) > self.N: del self.history[0]
    def dist(self, xy1, xy2):
        """L1 distance with torus-wrapping."""
        (x1,y1), (x2,y2), N = xy1, xy2, self.N
        dx = min(abs(x1 - x2 - N), abs(x1 - x2), abs(x1 - x2 + N))
        dy = min(abs(y1 - y2 - N), abs(y1 - y2), abs(y1 - y2 + N))
        return dx + dy
    def metric(self):
        return {'dist ↓': self.dist(self.xy, self.goal_xy), 'history': self.plot_history}
    def plot_history(self, plt, key, plot_length):
        """The most-recent trajectory."""
        assert key == 'history'
        hsize, vsize = plot_length('↓ dist'), self.N
        N, H = self.N, self.history
        x = [x / N * hsize for x,y,X,Y in H]
        y = [y / N * vsize for x,y,X,Y in H]
        u = [(X-x) / N * hsize for x,y,X,Y in H]
        v = [(Y-y) / N * vsize for x,y,X,Y in H]
        plt.quiver(x, y, u, v, scale=1, scale_units='xy', angles='xy', units='xy')