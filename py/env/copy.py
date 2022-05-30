import random



class Env:
    """
    The simplest copy-task: most bit-inputs have to be ignored, some bits have to be remembered and recalled when requested, which results in -1|1 reward.
    """
    bit = 0
    reward = 0
    reward_metric = 0.
    def __call__(self, sn):
        p = random.random()

        # Give reward & strive for it.
        if self.reward != 0:
            sn.data('reward', [self.reward])
            self.reward = 0
        sn.data(('reward', 'goal'), [1])

        if p < .9: # Observe rubbish.
            sn.data('rubbish', [random.randint(0,1)])
        elif p < .97: # Observe an important bit.
            self.bit = random.randint(0,1)
            sn.data('remember', [self.bit])
        else: # Recall the previous important bit.
            bit = bool(self.bit)
            def callback(fb):
                self.reward = 1 if (fb[0] > 0) == bit else -1
                self.reward_metric = self.reward_metric * .99 + .01 * self.reward
            sn.query('recall', 1, callback=callback)
    def metric(self):
        return {'reward': self.reward_metric}