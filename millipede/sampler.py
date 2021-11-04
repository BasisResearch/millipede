class MCMCSampler(object):
    def __init__(self):
        pass

    def initialize_sample(self):
        raise NotImplementedError

    def mcmc_chain(self, T_burnin, T):
        self.t = 0
        self.T_burnin = T_burnin
        sample = self.initialize_sample()

        for step in range(T_burnin + T):
            sample = self.mcmc_move(sample)
            if step >= T_burnin:
                yield (True, sample)
            else:
                yield (False, sample)

    def mcmc_move(self, sample):
        raise NotImplementedError
