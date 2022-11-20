# [OVFM:Online Learning in Variable Feature Space With Mixed Data](http://bojianhou.com/files/OVFM.pdf)
### [The IEEE International Conference on Data Mining (ICDM-2021)](https://icdm2021.auckland.ac.nz/)

![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

## abstract
This paper explores a new online learning problem where the data streams are generated from an over-time varying feature space, in which the random variables are of mixed data types including Boolean, ordinal, and continuous. The crux of this setting lies in how to establish the relationship among features, such that the learner can enjoy 1) reconstructed information of the missed-out old features and 2) a jump-start of learning new features with educated weight initialization.  Unfortunately, existing methods mainly assume a linear mapping relationship among features or that the multivariate joint distribution could be modeled as a Gaussian, imiting their applicability to the mixed data streams. To fill the gap, we in this paper propose to leverage Gaussian copula to model the complex joint distribution of the mixed data, we in this paper propose to model the complex joint distribution underlying mixed data with Gaussian copula, where the observed features with arbitrary marginals are mapped onto a latent normal space. The feature correlation is approximated in the latent space through an online EM process. Two base learners trained on the observed and latent features are ensembled to expedite convergence, thereby minimizing prediction risk in an online setting. Theoretical and empirical studies substantiate the effectiveness of our proposal.

## Requiremnt

The code was tested on:

- scipy
- statsmodels
- numpy
- sklearn
- pandas
- math
- matplotlib

To install requirement:
```
# install requirement
pip install -r requirements
```

## Run
Move to source code directory:
```
cd source
```
Run main.py 
```
python cap_main.py 
```
## Q&A
If you have any questions about the program or the paper, please feel free to contact us directly at *zhuosd@e.gzhu.edu.cn*
