# Global and Preference-based Optimization with Mixed Variables using Piecewise Affine Surrogates (PWAS/PWASp)


# Contents

* [Package description](#description)

* [Installation](#install)

* [Basic usage](#basic-usage)

* [Contributors](#contributors)

* [Citing PWAS](#bibliography)

* [License](#license)

<a name="description"></a>
## Package description

We propose a novel surrogate-based global optimization algorithm, called PWAS, based on constructing a piecewise affine surrogate of the objective function over feasible samples. We introduce two types of exploration functions to efficiently search the feasible domain via mixed integer linear programming (MILP) solvers. We also provide a preference-based version of the algorithm, called PWASp, which can be used when only pairwise comparisons between samples can be acquired while the objective function
remains unquantified. For more details on the method, please read our paper [Global and Preference-based Optimization with Mixed Variables using Piecewise Affine Surrogates](http://arxiv.org/abs/2302.04686). 

<a name="cite-ZB23"><a>
> [1] M. Zhu and A. Bemporad, "[Global and preference-based optimization with mixed variables using piecewise aﬃne surrogates](http://arxiv.org/abs/2302.04686)," *Submitted for publication*, 2023. [[bib entry](#ref1)]

<a name="install"></a>
## Installation

~~~code
pip install pwasopt
~~~


### Dependencies:
* python 3
* numpy
* scipy
* pulp
* sklearn
* [pyparc](https://pypi.org/project/pyparc/)
* [pyDOE](https://pythonhosted.org/pyDOE/)
* [pycddlib](https://pypi.org/project/pycddlib/)


<a name="basic-usage"></a>
## Basic usage
We show an example using PWAS/PWASp to optimize the parameters of the [`xgboost` algorithm](https://xgboost.readthedocs.io/en/stable/) for [`MNIST` classification](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html) task. 

### Problem discription
The 0.7/0.3 stratified train/test split ratio is applied. 
The `xgboost` package is used on `MNIST` classification. 
The optimization variables in this problem are the parameters of the `xgboost` algorithm.
Specifically, the continuous variables $x_1$, $x_2$, $x_3$, and $x_4$ refer to the following parameters in `xgboost`, 
respectively: `learning_rate`, `min_split_loss`, `subsample` , and `reg_lambda`. 
The integer variable $y$ stands for the `max_depth`. As for the categorical variables, $n_{d1}$ indicates the booster type in 
`xgboost` where $n_{d1} = \{0, 1\}$ corresponding to {`gbtree`, `dart`}. $n_{d2}$ represents the `grow_policy`, 
where $n_{d2} = \{0, 1\}$ corresponding to {`depthwise`, `lossguide`}. 
$n_{d3}$ refers to the `objective`, where $n_{d3} = \{0, 1\}$ corresponding to {`multi:softmax`, `multi:softprob`}.

### Use PWAS
~~~python
from pwasopt.main_pwas import PWAS

~~~

### Examples
Examples of benchmark testing using PWAS/PWASp can be found in the `examples` folder:
* `mixed_variable_benchmarks.py`: benchmark testing on constrained/unconstrained mixed-variable problems
  * Test results are reported in the [paper](http://arxiv.org/abs/2302.04686)
* `other_benchmarks.py`: various NLP, MIP, INLP, MIP Benchmarks tested with PWAS/PWASp
  * Test results are reported in [test_results_on_other_benchmarks.pdf](https://github.com/mjzhu-p/PWAS/blob/main/examples/test_results_on_other_benchmarks.pdf) under the `examples` folder 

 


<a name="contributors"><a>
## Contributors

This package was coded by Mengjia Zhu with supervision from Prof. Alberto Bemporad.


This software is distributed without any warranty. Please cite the above papers if you use this software.

<a name="bibliography"><a>
## Citing PWAS/PWASp

<a name="ref1"></a> 

Please cite ur paper if you would like to use the code.


<a name="ref1"></a>

```
@article{ZB23,
    author={M. Zhu, A. Bemporad},
    title={Global and Preference-based Optimization with Mixed Variables using Piecewise Affine Surrogates},
    journal={arXiv preprint arXiv:2302.04686},
    year=2023
}
```

<a name="license"><a>
## License

Apache 2.0

(C) 2021-2023 M. Zhu, A. Bemporad

 
