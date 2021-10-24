class MCMCSampler(object):
    def __init__(self):
        pass

    def initialize_sample(self):
        raise NotImplementedError

    def gibbs_chain(self, T_burnin, T):
        self.t = 0
        sample = self.initialize_sample()

        for step in range(T_burnin + T):
            sample = self.gibbs_move(sample)
            if step >= T_burnin:
                yield (True, sample)
            else:
                yield (False, sample)

    def gibbs_move(self, sample):
        raise NotImplementedError
