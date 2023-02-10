"""
1) Compare benchmark performance with results in [2] for unconstrained mixed-variable problems
2) Benchmark testing on constrained mixed-variable problems

The results are reported in [1]

[1] M. Zhu and A. Bemporad, “Global and Preference-based Optimization
    with Mixed Variables using Piecewise Affine Surrogates,”
     arXiv preprint arXiv:2302.04686, 2023.
[2] Ru, Binxin, Ahsan Alvi, Vu Nguyen, Michael A. Osborne, and Stephen Roberts.
"Bayesian optimisation over multiple continuous and categorical inputs."
In International Conference on Machine Learning, pp. 8276-8285. PMLR, 2020.

Authors: M. Zhu, A. Bemporad
"""

from src.pwas.main_pwas import PWAS
from src.pwas.main_pwasp import PWASp
from src.pwas.pref_fun1 import PWASp_fun1
from src.pwas.pref_fun import PWASp_fun

from numpy import array, zeros, ones
import numpy as np
import math
import sys

runPWAS = 1   #0 = run PWASp, 1 = run PWAS
runPWASp = 1-runPWAS

# synthetic benchmark from Ru's paper, reference: https://github.com/rubinxin/CoCaBO_code
# benchmark = 'Func-2C'
# benchmark = 'Func-3C'
# benchmark = 'Ackley-cC'

# real-world problems from Ru's paper, reference: https://github.com/rubinxin/CoCaBO_code
# benchmark = 'XG-MNIST'
# benchmark = 'NAS-CIFAR10'

# synthetic constrained benchmark
benchmark = 'Horst6_hs044_modified'  # 3 continuous variables, 4 integer variable, 2 categorical variable
# benchmark = 'roscam_modified'  # 2 continuous variables, 1 integer variable, 2 categorical variable

# default setting for the benchmarks
isLin_eqConstrained = False
isLin_ineqConstrained = False
Aeq = array([])
beq = array([])
Aineq = array([])
bineq = array([])

if benchmark == 'Func-2C':
    nc = 2
    nint = 0
    nd = 2
    X_d = [3, 3]
    lb = array([-1, -1, 0, 0])
    ub = array([1, 1, 2, 2])


    def fun(x):
        xc = x[:nc]
        xd = np.around(x[nc:])

        # xc = xc * 2

        assert len(xd) == 2
        ht1 = xd[0]
        ht2 = xd[1]

        if ht1 == 0:  # rosenbrock
            f = myrosenbrock(xc)
        elif ht1 == 1:  # six hump
            f = mysixhumpcamp(xc)
        elif ht1 == 2:  # beale
            f = mybeale(xc)

        if ht2 == 0:  # rosenbrock
            f = f + myrosenbrock(xc)
        elif ht2 == 1:  # six hump
            f = f + mysixhumpcamp(xc)
        else:
            f = f + mybeale(xc)

        y = f + 1e-6 * np.random.rand(f.shape[0], f.shape[1])
        return y[0][0]


    fopt0 = -0.20632
    xopt0 = array([[0.0898, -0.0898, 1, 1], [-0.7126, 0.7126, 1, 1]])

    isLin_eqConstrained = False
    isLin_ineqConstrained = False

    maxevals = 100
    n_initil = 20

elif benchmark == 'Func-3C':
    nc = 2
    nint = 0
    nd = 3
    X_d = [3, 3, 3]
    lb = array([-1, -1, 0, 0, 0])
    ub = array([1, 1, 2, 2, 2])


    def fun(x):
        xc = x[:nc]
        xd = np.around(x[nc:])

        xc = np.atleast_2d(xc)
        assert len(xd) == 3
        ht1 = xd[0]
        ht2 = xd[1]
        ht3 = xd[2]

        # xc = xc * 2
        if ht1 == 0:  # rosenbrock
            f = myrosenbrock(xc)
        elif ht1 == 1:  # six hump
            f = mysixhumpcamp(xc)
        elif ht1 == 2:  # beale
            f = mybeale(xc)

        if ht2 == 0:  # rosenbrock
            f = f + myrosenbrock(xc)
        elif ht2 == 1:  # six hump
            f = f + mysixhumpcamp(xc)
        else:
            f = f + mybeale(xc)

        if ht3 == 0:  # rosenbrock
            f = f + 5 * mysixhumpcamp(xc)
        elif ht3 == 1:  # six hump
            f = f + 2 * myrosenbrock(xc)
        else:
            f = f + ht3 * mybeale(xc)

        y = f + 1e-6 * np.random.rand(f.shape[0], f.shape[1])

        return y[0][0]


    fopt0 = -0.72214
    xopt0 = array([[0.0898, -0.0898, 1, 1, 0], [-0.7126, 0.7126, 1, 1, 0]])

    isLin_eqConstrained = False
    isLin_ineqConstrained = False

    maxevals = 100
    n_initil = 20

elif benchmark == 'Ackley-cC':
    nc = 1
    nint = 0
    # nd = 2  # Ackley-2C
    # nd = 3  # Ackley-3C
    # nd = 4  # Ackley-4C
    nd = 5  # Ackley-5C
    X_d = [17] * nd
    lb_cont = -ones((1))
    ub_cont = ones((1))
    lb_binary = zeros((nd))
    ub_binary = 16 * ones((nd))
    lb = np.hstack((lb_cont, lb_binary))
    ub = np.hstack((ub_cont, ub_binary))


    def fun(x):
        xc = x[:nc]
        xd = np.around(x[nc:])
        n = nc + nd

        a = 20
        b = 0.2
        c = 2 * np.pi
        s1 = 0
        s2 = 0

        x_ak = zeros((nc + nd, 1))
        x_ak[:nc, 0] = xc

        for i in range(nd):
            x_ak[nc + i, 0] = -1 + 0.125 * xd[i]

        for i in range(n):
            s1 += x_ak[i, 0] ** 2
            s2 += math.cos(c * x_ak[i, 0])

        f = -a * math.exp(-b * (1 / n * s1) ** (1 / 2)) - math.exp(1 / n * s2) + a + math.exp(1)

        return f


    isLin_eqConstrained = False
    isLin_ineqConstrained = False

    xopt0 = array([0, 8, 8, 8, 8, 8])
    fopt0 = 0

    maxevals = 100

    n_initil = 20

elif benchmark == 'XG-MNIST':
    nc = 4
    nint = 1
    nd = 3
    X_d = [2, 2, 2]

    lb_cont_int = array([1e-6, 1e-6, 0.001, 1e-6, 1])
    ub_cont_int = array([1, 10, 0.99999, 5, 10])
    lb_binary = zeros((nd))
    ub_binary = array([1, 1, 1])
    lb = np.hstack((lb_cont_int, lb_binary))
    ub = np.hstack((ub_cont_int, ub_binary))

    import xgboost
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn import metrics

    # example code: https://github.com/imrekovacs/XGBoost/blob/master/XGBoost%20MNIST%20digits%20classification.ipynb
    mnist = load_digits()
    X, y = mnist.data, mnist.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, stratify=y,
                                                        random_state=1)  # random_state used for reproducibility
    dtrain = xgboost.DMatrix(X_train, label=y_train)
    dtest = xgboost.DMatrix(X_test, label=y_test)


    def fun(x):
        xc = x[:nc]
        xint = x[nc:nc + nint]
        xd = x[nc + nint:]

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
            'max_depth': round(xint[0]),
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

    maxevals = 100
    n_initil = 20

elif benchmark == 'NAS-CIFAR10':
    nc = 21
    nint = 1
    nd = 5
    X_d = 5 * [3]
    lb_cont = 1e-6 * np.ones((nc))
    ub_cont = np.ones((nc))
    lb_int = 4 * np.ones((nint))
    ub_int = 9 * np.ones((nint))
    lb_binary = zeros((nd))
    ub_binary = array([2, 2, 2, 2, 2])
    lb = np.hstack((lb_cont, lb_int, lb_binary))
    ub = np.hstack((ub_cont, ub_int, ub_binary))

    from nas_cifar10 import NASCifar10C

    # Useful constants
    INPUT = 'input'
    OUTPUT = 'output'
    CONV3X3 = 'conv3x3-bn-relu'
    CONV1X1 = 'conv1x1-bn-relu'
    MAXPOOL3X3 = 'maxpool3x3'
    NUM_VERTICES = 7

    data_path = "data_nasbench"
    # data_path = "c:\\users\mengjia\Desktop\IMT\z-Research\\a_on_going_project\PWA-pref-based opt\code\pwas\\finalize_pls\pwas_Jan_2023"
    if data_path == "data_nasbench":
        errstr = "please indicate the location of file 'nasbench_only108.tfrecord',which is availalbe for download at https://storage.googleapis.com/nasbench/nasbench_only108.tfrecord"
        print(errstr)
        sys.exit(1)
    b_NAS = NASCifar10C(data_dir=data_path, multi_fidelity=False)
    cs = b_NAS.get_configuration_space()


    def fun(x):
        space = {}
        i = 0
        for h in cs.get_hyperparameters():
            if i < nc:
                space[h.name] = x[i]
                i += 1
            elif i == nc:
                space[h.name] = round(x[i])
                i += 1
            else:
                if x[i] == 0.:
                    space[h.name] = CONV1X1
                elif x[i] == 1.:
                    space[h.name] = CONV3X3
                else:
                    space[h.name] = MAXPOOL3X3
                i += 1

        f, c = b_NAS.objective_function(space)

        return f - 1  # since in NASCifar10C, the obj. fun. is 1-validation accuracy


    isLin_eqConstrained = False
    isLin_ineqConstrained = False

    maxevals = 100
    n_initil = 20

elif benchmark == "Horst6_hs044_modified":
    nc = 3
    nint = 4
    nd = 2
    X_d = [3,2]
    lb = array([0, 0, 0, 0,0, 0, 0,0,0])
    ub = array([6, 6, 3,3,10,3,10,2,1])
    # ub = array([6, 6, 3, 1, 3, 2, 2, 2, 1])


    def Horst6(x):
        Q = array([[0.992934, - 0.640117, 0.337286],
                   [-0.640117, - 0.814622, 0.960807],
                   [0.337286, 0.960807, 0.500874]])
        p = array([-0.992372, - 0.046466, 0.891766])
        f_Horst6 = np.transpose(x).dot(Q).dot(x) + p.dot(x)
        return f_Horst6


    def hs044(x):
        f_hs044 = x[0] - x[1] - x[2] - x[0]*x[2] + x[0]*x[3] + x[1]*x[2] - x[1]*x[3]
        return f_hs044


    def fun(x):
        xc = x[:nc]
        xint = np.around(x[nc:nc+nint])
        xd = np.around(x[nc + nint:])
        if xd[0] == 0:
            f = Horst6(xc) + hs044(xint)
        elif xd[0] == 1:
            f = 0.5*Horst6(xc) + hs044(xint)
        else:
            f = Horst6(xc) + 2*hs044(xint)

        if xd[1] == 0:
            f = abs(f)
        else:
            f = f
        return f


    isLin_eqConstrained = False
    isLin_ineqConstrained = True
    if isLin_eqConstrained:
        Aeq = array([])
        beq = array([])

    if isLin_ineqConstrained:
        Aineq = array([[0.488509, 0.063565, 0.945686, 0, 0, 0, 0,0, 0, 0, 0,0 ],
                       [-0.578592, -0.324014, -0.501754, 0, 0, 0, 0,0, 0, 0, 0,0],
                       [-0.719203, 0.099562, 0.445225, 0, 0, 0, 0,0, 0, 0, 0,0 ],
                       [-0.346896, 0.637939, -0.257623, 0, 0, 0, 0,0, 0, 0, 0,0 ],
                       [-0.202821, 0.647361, 0.920135, 0, 0, 0, 0,0, 0, 0, 0,0 ],
                       [-0.983091, -0.886420, -0.802444, 0, 0, 0, 0,0, 0, 0, 0,0 ],
                       [-0.305441, -0.180123, -0.515399, 0, 0, 0, 0,0, 0, 0, 0,0 ],
                       [0, 0, 0,1, 2, 0, 0,0, 0, 0, 0,0 ],
                       [0, 0, 0,4, 1, 0, 0,0, 0, 0, 0,0 ],
                       [0, 0, 0,3, 4, 0, 0,0, 0, 0, 0,0 ],
                       [0, 0, 0,0, 0, 2, 1,0, 0, 0, 0,0 ],
                       [0, 0, 0,0, 0, 1, 2,0, 0, 0, 0,0 ],
                       [0, 0, 0,0, 0, 1, 1,0, 0, 0, 0,0 ]
                       ])
        bineq = array([[2.865062],
                       [-1.491608],
                       [0.519588],
                       [1.584087],
                       [2.198036],
                       [-1.301853],
                       [-0.73829],
                       [8],
                       [12],
                       [12],
                       [8],
                       [8],
                       [5]
                       ])

    xopt_const = array([5.2106555627868909, 5.0279, 0,0, 3, 0, 4,2,1])  # constrained optimizers
    fopt_const =  -62.57932483728173 #constrained optimum

    maxevals = 100

elif benchmark == "roscam_modified":
    nc = 2
    nint = 1
    nd = 2
    X_d = [2,2]
    lb = array([-2.0, -2.0, 1, 0,0])
    ub = array([2.0, 2.0, 10, 1,1])


    def ros(x):
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        f_ros = 100 * (x2 - x1 ** 2) ** 2 + (x1 - 1) ** 2 + (x3-3)**2
        return f_ros


    def camp(x):
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        term1 = (4 - 2.1 * x1 ** 2 + (x1 ** 4) / 3) * x1 ** 2
        term2 = x1 * x2
        term3 = (-4 + 4 * x2 ** 2) * x2 ** 2
        f_camp = term1 + term2 + term3 + (x3-5)**2
        return f_camp


    def fun(x):
        xc = x[:nc + nint]
        xd = np.around(x[nc + nint:])
        if xd[0] == 0:
            f = ros(xc)
        else:
            f = camp(xc)

        if xd[1] == 0:
            f = f + ros(xc)
        else:
            f = f + camp(xc)
        return f


    isLin_eqConstrained = False
    isLin_ineqConstrained = True
    if isLin_eqConstrained:
        Aeq = array([])
        beq = array([])

    if isLin_ineqConstrained:
        Aineq = array([[1.6295, 1, 0,0,0,0,0],
                       [0.5,3.875,0, 0,0,0,0],
                       [-4.3023, -4,0, 0,0,0,0],
                       [-2,1,0, 0,0,0,0],
                       [0.5,-1,0, 0,0,0,0]])
        bineq = array([[3.0786],
                       [3.324],
                       [-1.4909],
                       [0.5],
                       [0.5]])

    xopt_const = array([0.0781, 0.6562,5,1,1])  # constrained optimizers
    fopt_const =  -1.81 #constrained optimum


    maxevals = 100

# ==========================================================================================================================================================
# the following rosenbrock, six-hump camel, and beal functions are used for comparison
# Taken from: https://github.com/rubinxin/CoCaBO_code/blob/master/testFunctions/syntheticFunctions.py
# =============================================================================
# Rosenbrock Function (f_min = 0)
# https://www.sfu.ca/~ssurjano/rosen.html
# =============================================================================
def myrosenbrock(X):
    X = np.asarray(X)
    X = X.reshape((-1, 2))
    if len(X.shape) == 1:  # one observation
        x1 = X[0]
        x2 = X[1]
    else:  # multiple observations
        x1 = X[:, 0]
        x2 = X[:, 1]
    fx = 100 * (x2 - x1 ** 2) ** 2 + (x1 - 1) ** 2
    return fx.reshape(-1, 1) / 300

# =============================================================================
#  Six-hump Camel Function (f_min = - 1.0316 )
#  https://www.sfu.ca/~ssurjano/camel6.html
# =============================================================================
def mysixhumpcamp(X):
    X = np.asarray(X)
    X = np.reshape(X, (-1, 2))
    if len(X.shape) == 1:
        x1 = X[0]
        x2 = X[1]
    else:
        x1 = X[:, 0]
        x2 = X[:, 1]
    term1 = (4 - 2.1 * x1 ** 2 + (x1 ** 4) / 3) * x1 ** 2
    term2 = x1 * x2
    term3 = (-4 + 4 * x2 ** 2) * x2 ** 2
    fval = term1 + term2 + term3
    return fval.reshape(-1, 1) / 10

# =============================================================================
# Beale function (f_min = 0)
# https://www.sfu.ca/~ssurjano/beale.html
# =============================================================================
def mybeale(X):
    X = np.asarray(X) / 2
    X = X.reshape((-1, 2))
    if len(X.shape) == 1:
        x1 = X[0] * 2
        x2 = X[1] * 2
    else:
        x1 = X[:, 0] * 2
        x2 = X[:, 1] * 2
    fval = (1.5 - x1 + x1 * x2) ** 2 + (2.25 - x1 + x1 * x2 ** 2) ** 2 + (
            2.625 - x1 + x1 * x2 ** 3) ** 2
    return fval.reshape(-1, 1) / 50


K_init = 20

try:
    nsamp = n_initil
except:
    nsamp = int(max(K_init, maxevals//4))

acq_stage = 'multi-stage'
feasible_sampling = True


key = 0
np.random.seed(key)  # rng default for reproducibility
####################################################################################
print("Solve the problem by feeding the simulator/synthetic decision-maker directly into the PWAS/PWASp solver")
# Solve global optimization problem
if runPWASp:
    delta_E = 1
    optimizer1 = PWASp(fun, lb, ub, delta_E, nc, nint, nd, X_d, nsamp, maxevals, feasible_sampling= feasible_sampling,
                     isLin_eqConstrained=isLin_eqConstrained, Aeq=Aeq, beq=beq,
                     isLin_ineqConstrained=isLin_ineqConstrained, Aineq=Aineq, bineq=bineq,
                     K=K_init, categorical=False,
                     acq_stage=acq_stage)

    xopt1 = optimizer1.solve()
    X1 = np.array(optimizer1.X)
    fbest_seq1 = list(map(fun, X1[optimizer1.ibest_seq]))
    fbest1 = min(fbest_seq1)

elif runPWAS:
    delta_E = 0.05
    optimizer1 = PWAS(fun, lb, ub, delta_E, nc, nint, nd, X_d, nsamp, maxevals,
                     feasible_sampling= feasible_sampling,
                     isLin_eqConstrained=isLin_eqConstrained, Aeq=Aeq, beq=beq,
                     isLin_ineqConstrained=isLin_ineqConstrained, Aineq=Aineq, bineq=bineq,
                     K=K_init, categorical=False,
                     acq_stage=acq_stage)

    xopt1, fopt1 = optimizer1.solve()
    X1 = np.array(optimizer1.X)
    fbest_seq1 = optimizer1.fbest_seq
####################################################################################


np.random.seed(key)  # rng default for reproducibility
####################################################################################
print("Solve the problem incrementally (i.e., provide the function evaluation at each iteration)")
# solve same problem, but incrementally
if runPWASp:
    delta_E = 1
    optimizer2 = PWASp(fun, lb, ub, delta_E, nc, nint, nd, X_d, nsamp, maxevals, feasible_sampling= feasible_sampling,
                     isLin_eqConstrained=isLin_eqConstrained, Aeq=Aeq, beq=beq,
                     isLin_ineqConstrained=isLin_ineqConstrained, Aineq=Aineq, bineq=bineq,
                     K=K_init, categorical=False,
                     acq_stage=acq_stage)

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


elif runPWAS:
    delta_E = 0.05
    optimizer2 = PWAS(fun, lb, ub, delta_E, nc, nint, nd, X_d, nsamp, maxevals,
                     feasible_sampling= feasible_sampling,
                     isLin_eqConstrained=isLin_eqConstrained, Aeq=Aeq, beq=beq,
                     isLin_ineqConstrained=isLin_ineqConstrained, Aineq=Aineq, bineq=bineq,
                     K=K_init, categorical=False,
                     acq_stage=acq_stage)

    x2 = optimizer2.initialize()
    for k in range(maxevals):
        f = fun(x2)
        x2 = optimizer2.update(f)
    X2 = np.array(optimizer2.X[:-1])  # it is because in prob.update, it will calculate the next point to query (the last x2 is calculated at max_evals +1)
    xopt2 = optimizer2.xbest
    fopt2 = optimizer2.fbest
##########################################








