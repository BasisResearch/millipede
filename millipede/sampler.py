from types import SimpleNamespace


class Sample(SimpleNamespace):
    """
    Extends SimpleNamespace to require shape immutability for robustness.
    In other words once the shape of a sample site has been set, that
    shape is required to remain unchanged.
    """

    def __setattr__(self, name, newval):
        if name in self.__dict__:
            val = self.__getattribute__(name)
            both_shaped = hasattr(val, "shape") and hasattr(newval, "shape")
            if both_shaped and val.shape != newval.shape:
                s = "Expected {} to have shape {} but got {}."
                raise ValueError(s.format(name, val.shape, newval.shape))
            elif not hasattr(val, "shape") and hasattr(newval, "shape"):
                s = "Expected {} to be shapeless but got {}."
                raise ValueError(s.format(name, newval.shape))
            elif hasattr(val, "shape") and not hasattr(newval, "shape"):
                s = "Expected {} to have shape {} but it's shapeless."
                raise ValueError(s.format(name, val.shape))
        super().__setattr__(name, newval)


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
