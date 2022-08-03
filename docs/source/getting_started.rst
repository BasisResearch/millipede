Getting Started
===============

What is millipede?
------------------

millipede is a `PyTorch <https://pytorch.org/>`__-based library for Bayesian variable selection in generalized linear models
that can be run on both CPU and GPU and that can handle datasets with numbers of data points and covariates 
in the tens of thousands or more.

What is Bayesian variable selection?
------------------------------------

Bayesian variable selection is a model-based approach for identifying parsimonious explanations of observed data.
In the context of generalized linear models with :math:`P` covariates :math:`\{ X_1, ..., X_P \}` and responses :math:`Y`,
Bayesian variable selection can be used to identify *sparse* subsets of covariates (i.e. far fewer than :math:`P`)
that are sufficient for explaining the observed responses in terms of a linear function of the covariates.

In more detail, Bayesian variable selection is formulated as a model selection problem in which we consider
the space of :math:`2^P` models in which some covariates are included and the rest are excluded.
For example, for continuous-valued responses one particular model might take the form :math:`Y = \beta_3 X_3 + \beta_9 X_9` 
with (non-zero) coefficients :math:`\beta_3` and :math:`\beta_9`. 
A priori we assume that models with fewer included covariates are more likely than those with more included covariates.
The set of parsimonious models best supported by the data then emerges from the posterior distribution over the space of models.

What's especially appealing about Bayesian variable selection is that it provides an interpretable score
called the PIP (posterior inclusion probability) for each covariate :math:`X_p`.
The PIP is a true probability and so it satisfies :math:`0 \le \rm{PIP} \le 1` by definition.
Covariates with large PIPs are good candidates for being explanatory of the response :math:`Y`.

Being able to compute PIPs is particularly useful for high-dimensional datasets with large :math:`P`.
For example, we might want to select a small number of covariates to include in a predictive model (i.e. feature selection).
Alternatively, in settings where it is implausible to subject all :math:`P` covariates to
some expensive downstream analysis (e.g. a laboratory experiment),
Bayesian variable selection can be used to select a small number of covariates for further analysis.


Requirements
-------------

millipede requires Python 3.8 or later and the following Python packages: 
`PyTorch <https://pytorch.org/>`__, 
`pandas <https://pandas.pydata.org>`__, and
`polyagamma <https://github.com/zoj613/polyagamma>`__. 

Note that if you wish to run millipede on a GPU you need to install PyTorch with CUDA support. 
In particular if you run the following command from your terminal it should report True:

::

    python -c 'import torch; print(torch.cuda.is_available())'


Installation instructions
-------------------------

Install directly from GitHub:

::

    pip install git+https://github.com/broadinstitute/millipede.git

Install from source:

::

    git clone git@github.com:broadinstitute/millipede.git
    cd millipede
    pip install .


Basic usage
-----------

Using millipede is easy:

::

    # import millipede 
    from millipede import NormalLikelihoodVariableSelector
    
    # create a VariableSelector object appropriate to your datatype
    selector = NormalLikelihoodVariableSelector(dataframe,  # pass in the data
                                                'Response', # indicate the column of responses
                                                S=1,        # specify the expected number of covariates to include a priori
                                               )

    # run the MCMC algorithm to compute posterior inclusion probabilities 
    # and other posterior quantities of interest
    selector.run(T=1000, T_burnin=500)

    # inspect the results
    print(selector.summary)

See the Jupyter notebooks in the `notebooks <https://github.com/broadinstitute/millipede/tree/master/notebooks>`__ directory for detailed example usage.


Supported data types
--------------------

The covariates :math:`X` are essentially arbitrary and can be continuous-valued, binary-valued, a mixture of the two, etc.
Currently the response :math:`Y` can be any of the following:

* continuous-valued => use `NormalLikelihoodVariableSelector <https://millipede.readthedocs.io/en/latest/selection.html#normallikelihoodvariableselector>`__
* binary-valued => use `BernoulliLikelihoodVariableSelector <https://millipede.readthedocs.io/en/latest/selection.html#bernoullilikelihoodvariableselector>`__
* bounded counts => use `BinomialLikelihoodVariableSelector <https://millipede.readthedocs.io/en/latest/selection.html#binomiallikelihoodvariableselector>`__
* unbounded counts => use `NegativeBinomialLikelihoodVariableSelector <https://millipede.readthedocs.io/en/latest/selection.html#negativebinomiallikelihoodvariableselector>`__

Scalability
-----------

Roughly speaking, the cost of the MCMC algorithms implemented in millipede is proportional
to :math:`N \times P`, where :math:`N` is the total number of data points and :math:`P` is the total number of covariates.
For an **approximate** guide to hardware requirements please consult the following guidelines:

* If :math:`N \times P \lesssim 10^7` use a CPU
* If :math:`10^7 \lesssim N \times P \lesssim 10^9` use a GPU
* If :math:`10^9 \lesssim N \times P` you may be out of luck


FAQ
---

* How many MCMC iterations do I need for good results?

It's hard to say. Generally speaking, difficult regimes with highly-correlated covariates or a large number of
covariates are expected to require more iterations. Similarly, datasets with count-based responses are expected to require
more iterations than those with continuous-valued responses (because the underlying inference problem is more difficult).
The best way to determine if you need more MCMC iterations is to run millipede twice with different random number seeds.
If the results for both runs are not similar, you probably want to increase the number of iterations.
As a general rule of thumb, it's probably good to aim for at least :math:`10^4-10^5` samples if doing so is feasible.
Also, you probably want at least 1000 burn-in iterations.


Contact information
-------------------

Martin Jankowiak: mjankowi@broadinstitute.org


References
----------

* Jankowiak, M., 2022. `Bayesian Variable Selection in a Million Dimensions <https://arxiv.org/abs/2208.01180>`__ arXiv preprint arXiv:2208.01180.

* Zanella, G. and Roberts, G., 2019. `Scalable importance tempering and Bayesian variable selection <https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/rssb.12316>`__. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 81(3), pp.489-517.

Citations
---------

If you use millipede please consider citing:

::

@article{jankowiak2022bayesian,
      title={Bayesian Variable Selection in a Million Dimensions},
      author={Martin Jankowiak},
      journal={arXiv preprint arXiv:{2208.01180},
      year={2022},
      eprint={2208.01180},
      archivePrefix={arXiv},
      primaryClass={stat.ME}
}
