"""
Can RL be done not via approximating the loss (discounted sum of future distances here, AKA return), but via BPTT?

This 2D env has 1 spaceship and a repulsor in the middle; the actions control acceleration. All smooth, so grad-min should be able to find best actions. Effective exploration would be able to cover a lot of the torus.
"""



import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'



def env_init(batch_size): # → state, hidden_state
    """Generate initial positions & velocities of the agent."""
    return torch.rand(batch_size, 2, device=device), torch.zeros(batch_size, 2, device=device)
def env_step(posit, veloc, accel): # → state, hidden_state
    """`env_step(state, hidden_state, action)`

    Add acceleration to velocity, and add velocity to position.

    Max action-acceleration is `1e-3`.

    There's a repulsor at `(.5, .5)` that adds a force proportional to `3e-5 / (pos - (.5, .5)).square()` to acceleration, to make exploration not as trivial. There's friction that multiplies velocity (hidden state) by `0.99` each step.

    If you must know how it feels, use this simple HTML page:

    ```html
    <div id="target" style="position:absolute; background-color: red; width:5px; height:5px"></div>
    <script>
    target = document.getElementById('target')
    let mousex=0, mousey=0
    onmousemove = evt => { mousex = evt.clientX / innerWidth, mousey = evt.clientY / innerHeight }

    let x=0, y=0, dx=0, dy=0
    setInterval(() => {
    let mx = mousex-x, my=mousey-y, m = Math.hypot(mx, my)
    if (m) mx /= m/1e-3, my /= m/1e-3

    let flen = ((x-.5)**2 + (y-.5)**2 + 1e-5), f = 3e-5 / flen
    let fx = f * (x-.5)/flen, fy = f * (y-.5)/flen

    let ddx = mx+fx, ddy = my+fy
    dx+=ddx, dy+=ddy, x+=dx, y+=dy
    dx*=.99, dy*=.99, x=(x%1+1)%1, y=(y%1+1)%1
    target.style.left = (x * innerWidth) + 'px'
    target.style.top = (y * innerHeight) + 'px'
    }, 50) // 20 FPS
    </script>
    ```"""
    accel = accel[..., :2]
    accel = accel * 1e-3 / 2
    accel = accel / (accel.square().sum(-1, keepdim=True).sqrt().max(torch.tensor(1., device=device)))
    force_center = torch.ones(posit.shape[0], 2, device=device)/2
    force_len = (posit - force_center).square() + 1e-5
    force = 3e-5 / force_len
    accel = accel + force * (posit - force_center) / force_len
    veloc = (veloc + accel) * .99
    posit = torch.remainder(posit + veloc, 1.)
    return posit, veloc



# TODO: …Implement an RNN that minimizes the distance between goal-conditioned paths and goals…
#   TODO: Have a replay buffer already.
#   TODO: …Come up with some metric of board-coverage by the replay buffer, and try to log that over time.




# TODO: Use `model.rnn.RNN` to predict the next observation.
# TODO: Try to learn a map in it via BPTT (given an RNN with an input→output skip connection, with a small multiplier on the added branch for discounting; minimize the distance from RNN-goal-space to ), to empirically verify (or contradict) that RL can really be replaced by pointwise minimization.
#   TODO: During unrolling, try sampling `next`-goals and distance-minimized goals independently, from the replay buffer. (In expectation, equivalent to distance-minimizing to the mean of all goals, so this couldn't be right.)
#   TODO: During unrolling, try sampling per-step `next`'s and distance-minimized goals.
#   TODO: During unrolling, try re-sampling the goal ONLY between BPTT steps.
#   TODO: Also try joint embedding, since prediction blurs frames: ensure that embeddings of consecutive frame-states are the same (but distinct over time) (with an extra NN to signify next-step), and minimize future-distance of embeddings by actions; either use CCL between big vectors everywhere, or BYOL (with a target-conditioned-predictor?).
# TODO: Try to learn a map in it via RL.

# TODO: Gotta get back, back to the past:
#   TODO: In `test.py`, implement self-targeting RL (with dist-bootstrapping and `next`-dist-min and self-imitation) and self-targeting BPTT (with `next`-dist-min and a skip connection), and try to not just explore one graph but *learn* to explore `minienv`'s graphs. (I don't think any RL exploration method can *learn* to explore, only explore. So if it works, it's cool.)