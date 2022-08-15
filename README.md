[![Build Status](https://github.com/BasisResearch/millipede/workflows/CI/badge.svg)](https://github.com/BasisResearch/millipede/actions)
[![Documentation Status](https://readthedocs.org/projects/millipede/badge/?version=latest)](https://millipede.readthedocs.io/en/latest/?badge=latest)
      

# millipede: A library for Bayesian variable selection
```
                                        ..    ..
                           millipede      )  (
                      _ _ _ _ _ _ _ _ _ _(.--.)
                     {_{_{_{_{_{_{_{_{_{_( '_')
                     /\/\/\/\/\/\/\/\/\/\ `---
```

millipede is a [PyTorch](https://pytorch.org/)-based library for Bayesian variable selection in generalized
linear models that can be run on both CPU and GPU and that
can handle datasets with numbers of data points and covariates in the tens of thousands or more.

 
## What is Bayesian variable selection?

Bayesian variable selection is a model-based approach for identifying parsimonious explanations of observed data.
In the context of generalized linear models with `P` covariates `{X_1, ..., X_P}` and responses `Y`, 
Bayesian variable selection can be used to identify *sparse* subsets of covariates (i.e. far fewer than `P`) 
that are sufficient for explaining the observed responses in terms of a linear function of the covariates.

In more detail, Bayesian variable selection is formulated as a model selection problem in which we consider 
the space of `2^P` models in which some covariates are included and the rest are excluded.
For example, for continuous-valued responses one particular model might take the form `Y = beta_3 X_3 + beta_9 X_9` 
with (non-zero) coefficients `beta_3` and `beta_9`.
A priori we assume that models with fewer included covariates are more likely than those with more included covariates.
The set of parsimonious models best supported by the data then emerges from the posterior distribution over the space of models.

What's especially appealing about Bayesian variable selection is that it provides an interpretable score
called the PIP (posterior inclusion probability) for each covariate `X_p`. 
The PIP is a true probability and so it satisfies `0 <= PIP <= 1` by definition.
Covariates with large PIPs are good candidates for being explanatory of the response `Y`.

Being able to compute PIPs is particularly useful for high-dimensional datasets with large `P`.
For example, we might want to select a small number of covariates to include in a predictive model (i.e. feature selection). 
Alternatively, in settings where it is implausible to subject all `P` covariates to 
some expensive downstream analysis (e.g. a laboratory experiment),
Bayesian variable selection can be used to select a small number of covariates for further analysis. 
  

## Requirements

millipede requires Python 3.8 or later and the following Python packages: [PyTorch](https://pytorch.org/), [pandas](https://pandas.pydata.org/), and [polyagamma](https://github.com/zoj613/polyagamma). 

Note that if you wish to run millipede on a GPU you need to install PyTorch with CUDA support. 
In particular if you run the following command from your terminal it should report True:
```
python -c 'import torch; print(torch.cuda.is_available())'
```


## Installation instructions

Install directly from GitHub:

```pip install git+https://github.com/BasisResearch/millipede.git```

Install from source:
```
git clone git@github.com:BasisResearch/millipede.git
cd millipede
pip install .
```

## Basic usage

Using millipede is easy:
```python
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
```

See the Jupyter notebooks in the [notebooks](https://github.com/BasisResearch/millipede/tree/master/notebooks) directory for detailed example usage.


## Supported data types 

The covariates `X` are essentially arbitrary and can be continuous-valued, binary-valued, a mixture of the two, etc.
Currently the response `Y` can be any of the following:

| Response type     | Selector class 
| ------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| continuous-valued | [NormalLikelihoodVariableSelector](https://millipede.readthedocs.io/en/latest/selection.html#millipede.selection.NormalLikelihoodVariableSelector)       |
| binary-valued     | [BernoulliLikelihoodVariableSelector](https://millipede.readthedocs.io/en/latest/selection.html#millipede.selection.BernoulliLikelihoodVariableSelector) |
| bounded counts    | [BinomialLikelihoodVariableSelector](https://millipede.readthedocs.io/en/latest/selection.html#binomiallikelihoodvariableselector)                       |
| unbounded counts  | [NegativeBinomialLikelihoodVariableSelector](https://millipede.readthedocs.io/en/latest/selection.html#negativebinomiallikelihoodvariableselector)       |


## Scalability

Roughly speaking, the cost of the MCMC algorithms implemented in millipede is proportional
 to `N x P`, where `N` is the total number of data points and `P` is the total number of covariates. 
For an **approximate** guide to hardware requirements please consult the following table:

| Regime                 | Expectations                            |
| -----------------------|-----------------------------------------|
| `N x P < 10^7`         | Use a CPU                               |
| `10^7 < N x P < 10^8`  | Use a GPU                               |
| `10^8 < N x P < 10^10` | Use a GPU with the subset_size argument |
| `10^10 < N x P`        | You may be out of luck                  |


## Documentation

Read the docs [here](https://millipede.readthedocs.io/en/latest/).


## FAQ

- How many MCMC iterations do I need for good results?

It's hard to say. Generally speaking, difficult regimes with highly-correlated covariates or a large number of
covariates are expected to require more iterations. Similarly, datasets with count-based responses are expected to require
more iterations than those with continuous-valued responses (because the underlying inference problem is more difficult).
The best way to determine if you need more MCMC iterations is to run millipede twice with different random number seeds.
If the results for both runs are not similar, you probably want to increase the number of iterations.
As a general rule of thumb, it's probably good to aim for at least `10^4-10^5` samples if doing so is feasible. 
Also, you probably want at least 1000 burn-in iterations.


## Contact information

Martin Jankowiak: martin@basis.ai


## References

Jankowiak, M., 2022. [Bayesian Variable Selection in a Million Dimensions](https://arxiv.org/abs/2208.01180). arXiv preprint arXiv:2208.01180.

Zanella, G. and Roberts, G., 2019. [Scalable importance tempering and Bayesian variable selection](https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/rssb.12316). Journal of the Royal Statistical Society: Series B (Statistical Methodology), 81(3), pp.489-517.

## Citing millipede

If you use millipede please consider citing:
```
@article{jankowiak2022bayesian,
      title={Bayesian Variable Selection in a Million Dimensions},
      author={Martin Jankowiak},
      journal={arXiv preprint arXiv:{2208.01180},
      year={2022},
      eprint={2208.01180},
      archivePrefix={arXiv},
      primaryClass={stat.ME}
}
```
