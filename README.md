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

We propose a novel surrogate-based global optimization algorithm, called PWAS, based on constructing a **p**iece**w**ise **a**ffine **s**urrogate of the objective function over feasible samples. We introduce two types of exploration functions to efficiently search the feasible domain via mixed integer linear programming (MILP) solvers. We also provide a preference-based version of the algorithm, called PWASp, which can be used when only pairwise comparisons between samples can be acquired while the objective function
remains unquantified. For more details on the method, please read our paper [Global and Preference-based Optimization with Mixed Variables using Piecewise Affine Surrogates](http://arxiv.org/abs/2302.04686). 

<a name="cite-ZB23"><a>
> [1] M. Zhu and A. Bemporad, "[Global and preference-based optimization with mixed variables using piecewise aﬃne surrogates](http://arxiv.org/abs/2302.04686)," *Submitted for publication*, 2023. [[bib entry](#ref1)]

<a name="install"></a>
## Installation

~~~code
pip install pwasopt
~~~


### Dependencies:
* python >=3.7
* numpy >=1.24.3
* scipy >=1.11.1
* pulp >=2.8.0
* scikit-learn >=1.3.0
* threadpoolctl >=3.1.0 (for `KMeans` from `scikit-learn` to run properly)
* [pyparc](https://pypi.org/project/pyparc/) >=2.0.4
* [pyDOE](https://pythonhosted.org/pyDOE/) >=0.3.8
* [pycddlib](https://pypi.org/project/pycddlib/) >=2.1.7, <3.0.0

### External dependencies:
MILP solver:
- PWAS/PWASp use `GUROBI` as the default solver to solve the MILP problem of acquisition optimization, 
which is found to be the most robust during benchmark testing. Alternatively, we also include `GLPK`, which may introduce
errors occasionally depending on the test problem and initial samples. User can also switch to another MILP solver by editing the
relevant codes in `acquisition.py` and `sample.py`. Check the compatability of the MILP solver with `pulp` (the LP modeler) 
at the [project webpage](https://pypi.org/project/PuLP/).
- `GUROBI`: [academic licenses](https://www.gurobi.com/academia/academic-program-and-licenses/)
  - [configure GUROBI with python](https://support.gurobi.com/hc/en-us/articles/360044290292-How-do-I-install-Gurobi-for-Python-)
  - [step-by-step explanation for the installation of Gurobi](https://support.gurobi.com/hc/en-us/articles/14799677517585-Getting-Started-with-Gurobi-Optimizer)
- `GLPK`: [project webpage](https://www.gnu.org/software/glpk/)
  - [step-by-step explanation for the installation on Stack Overflow](https://stackoverflow.com/questions/17513666/installing-glpk-gnu-linear-programming-kit-on-windows)


<a name="basic-usage"></a>
## Basic usage

### Examples
Examples of benchmark testing using PWAS/PWASp can be found in the `examples` folder:
* `mixed_variable_benchmarks.py`: benchmark testing on constrained/unconstrained mixed-variable problems
  * Test results are reported in the [paper](http://arxiv.org/abs/2302.04686)
  * _Note_: to test benchmark `NAS-CIFAR10`
    * download the data from its [GitHub repository](https://storage.googleapis.com/nasbench/nasbench_only108.tfrecord)
    * indicate the `data_path` in `mixed_variable_benchmarks.py`
    * since the dataset is compiled with `TensorFlow` version 1.x, **python version < 3.8** is  required (with `TensorFlow` < 2.x)
* `other_benchmarks.py`: various NLP, MIP, INLP, MIP Benchmarks tested with PWAS/PWASp
  * Test results are reported in [test_results_on_other_benchmarks.pdf](https://github.com/mjzhu-p/PWAS/blob/main/examples/test_results_on_other_benchmarks.pdf) under the `examples` folder 

### Case studies
Experimental design with PWAS: [ExpDesign](https://github.com/MolChemML/ExpDesign)
* Optimization of reaction conditions for Suzuki–Miyaura cross-coupling (fully categorical)
* Optimization of crossed-barrel design to augment mechanical toughness (mixed-integer)
* Solvent design for enhanced Menschutkin reaction rate (mixed-integer and categorical with linear constraints)


### Illustrative example
Here, we show a detailed example using PWAS/PWASp to optimize the parameters of the [`xgboost` algorithm](https://xgboost.readthedocs.io/en/stable/) for [`MNIST` classification](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html) task. 


### Problem discription

**_Objective_**:
Maximize the classification accuracy on test data. 
Note PWAS/PWASp optimizes the problem using **minimization**, and therefore we minimize the negative of classification accuracy.

**_Optimization variables_**:
$n_c = 4$ (number of continuous variables), $n_{\rm int} = 1$ (number of integer variables, **ordinal**), 
and $n_d = 3$ (number of categorical variables, **non-ordinal**) with $n_{i} = 2$, for $i = 1, 2, 3$. 
Each categorical variable ($n_{di}$) can be either 0 or 1. 
The bounds are $\ell_x = [10^{-6}\ 10^{-6}\ 0.001\ 10^{-6}]'$, $u_x = [1\  10\  1\  5]'$; $\ell_y = 1$, $u_y = 10$.

**_Notes_**:
The 0.7/0.3 stratified train/test split ratio is applied. 
The `xgboost` package is used on `MNIST` classification. 
The optimization variables in this problem are the parameters of the `xgboost` algorithm.
Specifically, the continuous variables $x_1$, $x_2$, $x_3$, and $x_4$ refer to the following parameters in `xgboost`, 
respectively: `learning_rate`, `min_split_loss`, `subsample` , and `reg_lambda`. 
The integer variable $y$ stands for the `max_depth`. As for the categorical variables, $n_{d1}$ indicates the booster type in 
`xgboost` where $n_{d1} = {0, 1}$ corresponding to {`gbtree`, `dart`}. $n_{d2}$ represents the `grow_policy`, 
where $n_{d2} = {0, 1}$ corresponding to {`depthwise`, `lossguide`}. 
$n_{d3}$ refers to the `objective`, where $n_{d3} = {0, 1}$ corresponding to {`multi:softmax`, `multi:softprob`}.


### Problem specification in Python
~~~python
import xgboost
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np

# info of optimization variables 
nc = 4  # number of continous variables
nint = 1 # number of integer variables, ordinal
nd = 3  # number of categorical variables, non-ordinal
X_d = [2, 2, 2]  # possible number of classes for each categorical variables

lb_cont_int = np.array([1e-6, 1e-6, 0.001, 1e-6, 1])  # lower bounds for continuous and integer variables
ub_cont_int = np.array([1, 10, 0.99999, 5, 10])  # upper bounds for continuous and integer variables
lb_binary = np.zeros((nd))  # lower bounds for categorical variables, note the dimension is the same as nd, it will be updated within the code
ub_binary = np.array([1, 1, 1]) # upper bounds for categorical variables, note it is (the number of classes-1) (since in the one-hot encoder, the counter started at 0)
lb = np.hstack((lb_cont_int, lb_binary)) # combined lower and upper bounds for the optimization variables
ub = np.hstack((ub_cont_int, ub_binary))

# load dataset
# example code: https://github.com/imrekovacs/XGBoost/blob/master/XGBoost%20MNIST%20digits%20classification.ipynb
mnist = load_digits()  
X, y = mnist.data, mnist.target
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, stratify=y,
                                                    random_state=1)  # random_state used for reproducibility
dtrain = xgboost.DMatrix(X_train, label=y_train)
dtest = xgboost.DMatrix(X_test, label=y_test)

# define the objective function, x collects all the optimization variables, ordered as [continuous, integer, categorical]
def fun(x):  
    xc = x[:nc]  # continuous variables
    xint = x[nc:nc + nint]  # integer variables
    xd = x[nc + nint:]  # categorical variables

    if xd[0] == 0:
        mnist_booster = 'gbtree'
    else:
        mnist_booster = 'dart'

    if xd[1] == 0:
        mnist_grow_policy = 'depthwise'
    else:
        mnist_grow_policy = 'lossguide'

    if xd[2] == 0:
        mnist_obj = 'multi:softmax'
    else:
        mnist_obj = 'multi:softprob'
    param = {
        'booster': mnist_booster,
        'grow_policy': mnist_grow_policy,
        'objective': mnist_obj,
        'learning_rate': xc[0],
        'min_split_loss': xc[1],
        'subsample': xc[2],
        'reg_lambda': xc[3],
        'max_depth': int(round(xint[0])),
        'num_class': 10  # the number of classes that exist in this datset
    }

    bstmodel = xgboost.train(param, dtrain)

    y_pred = bstmodel.predict(
        dtest)  # somehow predict gives probability of each class instead of which class it belongs in...

    try:
        acc = metrics.accuracy_score(y_test, y_pred)

    except:
        y_pred = np.argmax(y_pred, axis=1)
        acc = metrics.accuracy_score(y_test, y_pred)

    return -acc  # maximize the accuracy, minimze the -acc

# Specify the number of maximum number of evaluations (including initial sammples) and initial samples
maxevals = 100
n_initil = 20

# default setting for the benchmarks
isLin_eqConstrained = False  # specify whether linear equality constraints are present
isLin_ineqConstrained = False  # specify whether linear inequality constraints are present
Aeq = np.array([])  # linear equality constraints
beq = np.array([])
Aineq = np.array([])  # linear inequality constraints
bineq = np.array([])

~~~

### Solve use PWAS

One can solve the optimization problem either by explicitly passing the function handle `fun` to PWAS, or by passing 
the evaluation of `fun` step-by step.

#### Solve by explicitly passing the function handle
~~~python
from pwasopt.main_pwas import PWAS  

key = 0
np.random.seed(key)  # rng default for reproducibility

delta_E = 0.05  # trade-off hyperparameter in acquisition function between exploitation of surrogate and exploration of exploration function
acq_stage = 'multi-stage'  # can specify whether to solve the acquisition step in one or multiple stages (as noted in Section 3.4 in the paper [1]. Default: `multi-stage`
feasible_sampling = True  # can specify whether infeasible samples are allowed. Default True
K_init = 20  # number of initial PWA partitions

# initialize the PWAS solver
optimizer1 = PWAS(fun, lb, ub, delta_E, nc, nint, nd, X_d, nsamp, maxevals,  # pass fun to PWAS
                 feasible_sampling= feasible_sampling,
                 isLin_eqConstrained=isLin_eqConstrained, Aeq=Aeq, beq=beq,
                 isLin_ineqConstrained=isLin_ineqConstrained, Aineq=Aineq, bineq=bineq,
                 K=K_init, categorical=False,
                 acq_stage=acq_stage)

xopt1, fopt1 = optimizer1.solve()
X1 = np.array(optimizer1.X)
fbest_seq1 = optimizer1.fbest_seq
~~~

#### Solve by passing the function evaluation step-by step
~~~python
optimizer2 = PWAS(fun, lb, ub, delta_E, nc, nint, nd, X_d, nsamp, maxevals,  # here, fun is a placeholder passed to PWAS, not used
                 feasible_sampling= feasible_sampling,
                 isLin_eqConstrained=isLin_eqConstrained, Aeq=Aeq, beq=beq,
                 isLin_ineqConstrained=isLin_ineqConstrained, Aineq=Aineq, bineq=bineq,
                 K=K_init, categorical=False,
                 acq_stage=acq_stage)

x2 = optimizer2.initialize()
for k in range(maxevals):
    f = fun(x2)  # evaluate fun
    x2 = optimizer2.update(f) # feed function evaluation step by step to PWAS
X2 = np.array(optimizer2.X[:-1])  # it is because in prob.update, it will calculate the next point to query (the last x2 is calculated at max_evals +1)
xopt2 = optimizer2.xbest
fopt2 = optimizer2.fbest
X2 = np.array(optimizer2.X)
fbest_seq2 = optimizer2.fbest_seq

~~~
Below we show the best values `fbest_seq1` found by PWAS. 

<p align = "center">
<img src="https://github.com/mjzhu-p/PWAS/blob/main/figures/PWAS_XG-MNIST.png" alt="drawing" width=60%/>
</p>


### Solve use PWASp
When solve with PWASp, instead of using the function evaluations, we solve a preference-based optimization problem 
with preference function $\pi(x_1,x_2)$, $x_1,x_2\in\mathbb{R}^n$
within the finite bounds `lb` $\leq x\leq$ `ub` (see Section 4 of [[1]](#ref1)).

Similarly to PWAS, one can solve the optimization problem either by 
explicitly passing the preference indicator/synthetic preference function to PWASp, or by passing 
the expressed preference `pref_eval` step-by step.

_Note_: for this example, we use `fun` as a **synthetic decision maker** (`synthetic_dm = True`) to express preferences. However, the explicit evaluation of `fun` is unknow to PWASp.

When solve by **explicitly** passing the preference indicator:
- If `synthetic_dm = True`, we have included two preference indicator functions `pref_fun.py` and `pref_fun1.py` 
to provide preferences based on function evaluations and constraint satisfaction.
- If `synthetic_dm = False`, one need to pass a `fun` such that given two decision vectors,
output -1, 0, or 1 depending on the expressed preferences.

#### Solve by explicitly passing the preference indicator 

~~~python
from pwasopt.main_pwasp import PWASp 

key = 0
np.random.seed(key)  # rng default for reproducibility

delta_E = 1  # trade-off hyperparameter in acquisition function between exploitation of surrogate and exploration of exploration function
optimizer1 = PWASp(fun, lb, ub, delta_E, nc, nint, nd, X_d, nsamp, maxevals, feasible_sampling= feasible_sampling,  
                 isLin_eqConstrained=isLin_eqConstrained, Aeq=Aeq, beq=beq,
                 isLin_ineqConstrained=isLin_ineqConstrained, Aineq=Aineq, bineq=bineq,
                 K=K_init, categorical=False,
                 acq_stage=acq_stage, synthetic_dm = True)  

xopt1 = optimizer1.solve()
X1 = np.array(optimizer1.X)
fbest_seq1 = list(map(fun, X1[optimizer1.ibest_seq]))  # for synthetic problems, we can obtain the function evaluation for assessment of the solver
fbest1 = min(fbest_seq1)
~~~

#### Solve by passing the expressed preference step-by step
~~~python
from pwasopt.pref_fun1 import PWASp_fun1  # import the preference indicator function
from pwasopt.pref_fun import PWASp_fun

optimizer2 = PWASp(fun, lb, ub, delta_E, nc, nint, nd, X_d, nsamp, maxevals, feasible_sampling= feasible_sampling,  # fun is a placeholder here
                 isLin_eqConstrained=isLin_eqConstrained, Aeq=Aeq, beq=beq,
                 isLin_ineqConstrained=isLin_ineqConstrained, Aineq=Aineq, bineq=bineq,
                 K=K_init, categorical=False,
                 acq_stage=acq_stage, synthetic_dm = True)

comparetol = 1e-4
if isLin_ineqConstrained or isLin_eqConstrained:
    pref_fun = PWASp_fun1(fun, comparetol, optimizer2.prob.Aeq, optimizer2.prob.beq, optimizer2.prob.Aineq, optimizer2.prob.bineq)  # preference function object
else:
    pref_fun = PWASp_fun(fun, comparetol)
pref = lambda x, y, x_encoded, y_encoded: pref_fun.eval(x, y, x_encoded, y_encoded)
pref_fun.clear()

xbest2, x2, xsbest2, xs2 = optimizer2.initialize()  # get first two random samples
for k in range(maxevals-1):
    pref_eval = pref(x2, xbest2, xs2, xsbest2)  # evaluate preference
    x2 = optimizer2.update(pref_eval)
    xbest2 = optimizer2.xbest
X2 = np.array(optimizer2.X[:-1])
xopt2 = xbest2
fbest_seq2 = list(map(fun, X2[optimizer2.ibest_seq]))
fbest2 = min(fbest_seq2)

~~~
Below we show the best values `fbest_seq1` found by PWASp. Note that function evaluations here are shown solely for demonstration purposes, which are unknown to PWASp during the solution process.

<p align = "center">
<img src="https://github.com/mjzhu-p/PWAS/blob/main/figures/PWASp_XG-MNIST.png" alt="drawing" width=60%/>
</p>


### Include constraints

We note below how to include constraints if present. 

_Note_:  current package only supports **linear** equality/inequality constraints.

- `Aeq`: np array, dimension: (# of linear eq. const by n_encoded), where n_encoded is the length of the optimization variable 
**AFTER** being **encoded**, the coefficient matrix for the linear equality constraints
- `beq`: np array, dimension: (n_encode by 1), the RHS of the linear eq. constraints
- `Aineq`: np array, dimension: (# of linear ineq. const by n_encoded), the coefficient matrix for the linear inequality constraints
**AFTER** being **encoded** the coefficient matrix for the linear equality constraints
- `bineq`: np array, dimension: (n_encode by 1) the RHS of the linear ineq. constraints
- **Make sure to update the `Aeq` and `Aineq` with the one-hot encoded categorical variables, if present.**

~~~python
# if there is equality constraints
isLin_eqConstrained = True  #(Aeq x  = beq)

# specify the constraint matrix and right-hand-side vector
if isLin_eqConstrained:  # an example
    Aeq = np.array([1.6295, 1])
    beq = np.array([3.0786])


# if there is inequality constraints
isLin_ineqConstrained = True  # (Aineq x <= bineq)
# specify the constraint matrix and right-hand-side vector
if isLin_ineqConstrained:  # an example
    Aineq = np.array([[1.6295, 1],
                   [0.5, 3.875],
                   [-4.3023, -4],
                   [-2, 1],
                   [0.5, -1]])
    bineq = np.array([[3.0786],
                   [3.324],
                   [-1.4909],
                   [0.5],
                   [0.5]])

~~~

<a name="contributors"><a>
## Contributors

This package was coded by Mengjia Zhu with supervision from Prof. Alberto Bemporad.


This software is distributed without any warranty. Please cite the paper below if you use this software.

<a name="bibliography"><a>
## Citing PWAS/PWASp

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

 
