MCMC Samplers
=========================

For most users it suffices to use the predefined Bayesian variable selection classes
like :class:`millipede.NormalLikelihoodVariableSelector`.
However advanced users also have the option of using the MCMC samplers directly.
Note that the samplers are only partially documented. For example usage see
`the source code <https://github.com/broadinstitute/millipede/blob/master/millipede/selection.py>`__.

NormalLikelihoodSampler
-----------------------
.. autoclass:: millipede.normal.NormalLikelihoodSampler


CountLikelihoodSampler
-----------------------
.. autoclass:: millipede.binomial.CountLikelihoodSampler
