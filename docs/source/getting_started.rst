Getting Started
===============

What is Bayesian variable selection?
------------------------------------

Bayesian variable selection is a model-based approach for identifying parsimonious explanations of observed data.
In the context of generalized linear models with `P` covariates `{X_1, ..., X_P}` and responses `Y`, 
Bayesian variable selection can be used to identify *sparse* subsets of covariates (i.e. far fewer than `P`) 
that are sufficient for explaining the observed responses.

In more detail, Bayesian variable selection can be understood as a model selection problem in which we consider 
the space of `2^P` models in which some covariates are included and the rest are excluded.
A priori we assume that models with fewer included covariates are more likely than those with more included covariates.
The models best supported by the data are encoded as a posterior distribution over the space of models.

What's especially appealing about Bayesian variable selection is that it provides us with an interpretable score
called the PIP (posterior inclusion probability) for each covariate `X_p`. 
The PIP is a true probability and so it satisfies `0 <= PIP <= 1` by definition.
Covariates with large PIPs are good candidates for being explanatory of the response `Y`.

Being able to compute PIPs is particularly useful for high-dimensional datasets with large `P`.
For example, we might want to select a small number of covariates to include in a predictive model (i.e. feature selection). 
Alternatively, in settings where it is implausible to subject all `P` covariates to 
some expensive downstream analysis (e.g. a lab experiment),
Bayesian variable selection can be used to select a small number of covariates for further analysis. 
  

Requirements
-------------

millipede requires Python 3.8 and the following Python packages: 
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

    # create a VariableSelector object appropriate to your datatype
    selector = NormalLikelihoodVariableSelector(dataframe,  # pass in the data
                                                'Response', # indicate the column of responses
                                                S=1,        # specify the expected number of covariates to include a priori
                                               )

    # run the MCMC algorithm to compute posterior compusion probabilities and other posterior quantities of interest
    selector.run(T=1000, T_burnin=500)

    # inspect the results
    print(selector.summary)

See the Jupyter notebooks in the `notebooks <https://github.com/broadinstitute/millipede/tree/master/notebooks>`__ directory for detailed example usage.
