import random



name = 'trivial_copy_task'



class Env:
    """
    The simplest copy-task: most bit-inputs have to be ignored, some bits have to be remembered and recalled when requested, which results in -1|1 reward.
    """
    bit = 0
    reward = 0
    def __call__(self, sn):
        p = random.random()

        # TODO: …Should all the `name`s put `name` at the very end, so that our interface to goals (group IDs) can actually run multiple distinct envs seamlessly?
        #   TODO: Or should `sn` have a context manager that modifies names inside it? …Or maybe a decorator, since all these envs are intended to be sensors…
        #     …But how would that interact with async sensors/listeners?…
        # Give reward & strive for it.
        if self.reward != 0:
            sn.data((name, 'reward'), [self.reward])
            self.reward = 0
        sn.data((name, 'reward', 'goal', None), [1])
        #   TODO: …How to allow sensitivity to `len(sn.cell_shape)`, so that this isn't specialized to 4 name parts?…
        #     …A decorator of sensors really would be best, huh…

        if p < .9: # Observe rubbish.
            sn.data((name, 'rubbish'), [random.randint(0,1)])
        elif p < .97: # Observe an important bit.
            self.bit = random.randint(0,1)
            sn.data((name, 'remember'), [self.bit])
        else: # Recall the previous important bit.
            bit = bool(self.bit)
            def callback(fb, sn):
                self.reward = 1 if (fb[0] > 0) == bit else -1
            sn.query((name, 'recall'), 1, callback=callback)