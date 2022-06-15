import random



class Env:
    """
    The simplest copy-task: most bit-inputs have to be ignored, but some bits have to be remembered and recalled when requested, which results in -1|1 reward.
    """
    bit = 0
    reward = 0
    reward_metric = 0.
    def __call__(self, sn):
        p = random.random()

        # Give reward & strive for it.
        if self.reward != 0:
            sn.set('reward', self.reward, 2)
            self.reward = 0
        sn.set('reward', 1, sn.Goal(sn.Int(2)))

        if p < .9: # Observe rubbish.
            sn.set('rubbish', random.randint(0,1), 2)
        elif p < .97: # Observe an important bit.
            self.bit = random.randint(0,1)
            sn.set('remember', self.bit, 2)
        else: # Recall the previous important bit.
            bit = self.bit
            async def callback(fb):
                self.reward = int((await fb) == bit)
                self.reward_metric = self.reward_metric * .99 + .01 * self.reward
            sn.run(callback, sn.query('recall', 2))
    def metric(self):
        return {'reward': self.reward_metric}