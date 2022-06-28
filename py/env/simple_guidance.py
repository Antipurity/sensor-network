import random



class Env:
    """
    The simplest 2D torus physics: input position, output acceleration; velocity fades quickly.
    """
    N = 0
    xy = (0,0)
    veloc = (0,0)
    accel = (0,0)
    goal_xy = (0,0)
    def __init__(self): self.history = []
    def __call__(self, sn):
        self.update(self.accel)
        sn.set('xy', self.xy, sn.RawFloat(2))
        sn.set('xy', self.goal_xy, sn.Goal(sn.RawFloat(2)))
        sn.run(self.handle_feedback, sn.query('act', sn.RawFloat(2)))

        if random.random() < 1 / 64: # Switch goals sometimes.
            self.goal_xy = (random.random(), random.random())
    async def handle_feedback(self, fb): self.accel = await fb
    def update(self, accel):
        """Handles acceleration & velocity, and updates history."""
        from math import sqrt
        prev_xy = self.xy
        ddx, ddy = accel
        accel_mult = 1e-2 / (max(1., sqrt(ddx*ddx + ddy*ddy)) + 1e-5)
        self.veloc = tuple(max(-2., min(v + a * accel_mult, 2.)) * .9 for v,a in zip(self.veloc, accel))
        self.xy = tuple((p + v) % 1 for p,v in zip(self.xy, self.veloc))

        self.history.append((*prev_xy, *self.xy))
        if len(self.history) > 64: del self.history[0]
    def dist(self, xy1, xy2):
        """L1 distance with torus-wrapping."""
        (x1,y1), (x2,y2) = xy1, xy2
        dx = min(abs(x1 - x2 - 1), abs(x1 - x2), abs(x1 - x2 + 1))
        dy = min(abs(y1 - y2 - 1), abs(y1 - y2), abs(y1 - y2 + 1))
        return dx + dy
    def metric(self):
        return {'↓ dist': self.dist(self.xy, self.goal_xy), 'history': self.plot_history}
    def plot_history(self, plt, key, plot_length):
        """The most-recent trajectory."""
        assert key == 'history'
        hsize, vsize = plot_length('↓ dist'), 2
        H = self.history
        x = [x * hsize for x,y,X,Y in H]
        y = [y * vsize for x,y,X,Y in H]
        u = [(X-x) * hsize for x,y,X,Y in H]
        v = [(Y-y) * vsize for x,y,X,Y in H]
        plt.quiver(x, y, u, v, scale=1, scale_units='xy', angles='xy', units='xy')