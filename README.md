[![Build Status](https://github.com/broadinstitute/millipede/workflows/CI/badge.svg)](https://github.com/broadinstitute/millipede/actions)

```
                                 ..    ..
                    millipede      )  (
               _ _ _ _ _ _ _ _ _ _(.--.)
              {_{_{_{_{_{_{_{_{_{_( '_')
              /\/\/\/\/\/\/\/\/\/\ `---
```

# millipede: a library for bayesian variable selection


## Requirements

[PyTorch](https://pytorch.org/), [pandas](https://pandas.pydata.org/), and [polyagamma](https://github.com/zoj613/polyagamma).


## Installation instructions

Install directly from GitHub:

```pip install git+https://github.com/broadinstitute/millipede.git```

Install from source:
```
git clone git@github.com:broadinstitute/millipede.git
cd millipede
pip install .
```

## Usage

Using millipede is easy:
```python
# create a VariableSelector object appropriate to your datatype
selector = NormalLikelihoodVariableSelector(dataframe,  # pass in the data
                                            'Response', # indicate the column of responses
                                            S=1,        # specify the expected number of covariates to include a priori
                                           )

# run the MCMC algorithm to compute posterior compusion probabilities and other posterior quantities of interest
selector.run(T=1000, T_burnin=500)

# inspect the results
print(selector.summary)
```

See the Jupyter notebooks in the `notebooks/` directory for detailed example usage.


## Contact information

Martin Jankowiak: mjankowi@broadinstitute.org


## References

Zanella, G. and Roberts, G., 2019. [Scalable importance tempering and Bayesian variable selection](https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/rssb.12316). Journal of the Royal Statistical Society: Series B (Statistical Methodology), 81(3), pp.489-517.

Jankowiak, M., 2021. [Fast Bayesian Variable Selection in Binomial and Negative Binomial Regression](https://arxiv.org/abs/2106.14981). arXiv preprint arXiv:2106.14981.
