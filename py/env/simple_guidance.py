import random



class Env:
    """
    The simplest spaceship-on-a-torus physics: input position, output acceleration; velocity fades quickly.

    If you must know how it feels, use this simple HTML page:

    ```html
    <div id="target" style="position:absolute; background-color: red; width:5px; height:5px; border-radius:50%"></div>
    <script>
    target = document.getElementById('target')
    let mousex=0, mousey=0
    onmousemove = evt => { mousex = evt.clientX / innerWidth, mousey = evt.clientY / innerHeight }

    let x=0, y=0, dx=0, dy=0
    setInterval(() => {
        let mx = mousex-x, my=mousey-y, m = Math.hypot(mx, my)
        if (m) mx /= m/1e-2, my /= m/1e-2

        let ddx = mx, ddy = my
        dx+=ddx, dy+=ddy, x+=dx, y+=dy
        dx*=.9, dy*=.9, x=(x%1+1)%1, y=(y%1+1)%1
        target.style.left = (x * innerWidth) + 'px'
        target.style.top = (y * innerHeight) + 'px'
    }, 50) // 20 FPS
    </script>
    ```
    """
    def __init__(self, dims=2):
        assert isinstance(dims, int) and dims > 0
        self.history = []
        self.D = dims
        self.xy = tuple([0]*dims)
        self.veloc = tuple([0]*dims)
        self.accel = tuple([0]*dims)
        self.goal_xy = tuple([0]*dims)
    def __call__(self, sn):
        self.update(self.accel)
        sn.set('xy', self.xy, sn.Float(self.D))
        sn.set('xy', self.goal_xy, sn.Goal(sn.Float(self.D)))
        sn.run(self.handle_feedback, sn.query('act', sn.Float(self.D)))

        if random.random() < 1 / 32 / self.D: # Switch goals sometimes.
            self.goal_xy = tuple([random.random() for _ in range(self.D)])
    async def handle_feedback(self, fb): self.accel = await fb
    def update(self, accel):
        """Handles acceleration & velocity, and updates history."""
        from math import sqrt
        prev_xy = self.xy
        accel_mult = 1e-2 / (max(1., sqrt(sum(a*a for a in accel)) * 2 / self.D) + 1e-5)
        self.veloc = tuple(max(-2., min(v + a * accel_mult, 2.)) * .9 for v,a in zip(self.veloc, accel))
        self.xy = tuple((p + v) % 1 for p,v in zip(self.xy, self.veloc))

        self.history.append((prev_xy, self.xy))
        if len(self.history) > 64: del self.history[0]
    def dist(self, xy1, xy2):
        """L1 distance with torus-wrapping."""
        (x1,y1), (x2,y2) = xy1, xy2
        dx = min(abs(x1 - x2 - 1), abs(x1 - x2), abs(x1 - x2 + 1))
        dy = min(abs(y1 - y2 - 1), abs(y1 - y2), abs(y1 - y2 + 1))
        return dx + dy
    def metric(self):
        return {'dist ↓': self.dist(self.xy, self.goal_xy), 'history': self.plot_history}
    def plot_history(self, plt, key, plot_length):
        """The most-recent trajectory, as a projection onto a 2D plane."""
        assert key == 'history'
        hsize, vsize = plot_length('↓ dist'), self.D
        H = self.history
        x = [x * hsize for (x,y,*_),(X,Y,*_) in H]
        y = [y * vsize for (x,y,*_),(X,Y,*_) in H]
        u = [(X-x) * hsize for (x,y,*_),(X,Y,*_) in H]
        v = [(Y-y) * vsize for (x,y,*_),(X,Y,*_) in H]
        plt.quiver(x, y, u, v, scale=1, scale_units='xy', angles='xy', units='xy')