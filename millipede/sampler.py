class MCMCSampler(object):
    """
    Base class for all MCMC samplers.
    """
    def initialize_sample(self, seed=None):
        raise NotImplementedError

    def mcmc_chain(self, T_burnin, T, seed=None):
        self.t = 0
        self.T_burnin = T_burnin
        sample = self.initialize_sample(seed=seed)

        for step in range(T_burnin + T):
            sample = self.mcmc_move(sample)
            if step >= T_burnin:
                yield (True, sample)
            else:
                yield (False, sample)

    def mcmc_move(self, sample):
        raise NotImplementedError
