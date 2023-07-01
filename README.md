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

~~~python
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

 
