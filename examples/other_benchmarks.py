"""
NLP, MIP, INLP, MIP Benchmarks tested with PWAS/PWASp [1]

NLP problems taking from DIRECTGOLib - DIRECT Global Optimization test problems Library [2]
    https://github.com/blockchain-group/DIRECTGOLib/tree/v1.1

MIP problems taken from # http://miplib.zib.de/index.html and https://www.minlplib.org/


[1] M. Zhu and A. Bemporad, “Global and Preference-based Optimization with Mixed Variables using Piecewise Affine Surrogates,”
     arXiv preprint arXiv:2302.04686, 2023.
[2] Stripinis, Linas, and Remigijus Paulavičius. "DIRECTGO: A new DIRECT-type MATLAB toolbox for derivative-free global optimization."
    ACM Transactions on Mathematical Software 48.4 (2022): 1-46.

Authors: M. Zhu, A. Bemporad
"""

from pwasopt.main_pwas import PWAS
from pwasopt.main_pwasp import PWASp

from numpy import array, zeros, ones
import numpy as np
from math import log,sqrt

from pysmps import smps_loader as smps

# plotting libraries
from numpy import arange, meshgrid
import matplotlib.pyplot as plt

runPWAS = 1   #0 = run PWASp, 1 = run PWAS
runPWASp = 1-runPWAS

# 2-D benchmarks with box constraints (NLP)
# benchmark = "PWA_example"
# benchmark="camelsixhumps"
# benchmark="ackley"

# 2-D benchmarks with box and linear constraints (NLP)
# benchmark='camelsixhumps-linearconstr' #camelsixhumps with 5 known linear constraints, optimal is at intersection
# benchmark='camelsixhumps-linearconstr_2' #camelsixhumps with 5 known linear constraints
# benchmark = "Horst3" # 3 linear inequality constraints, optimal is at intersection
# benchmark = "s232" # 3 linear inequality constraints, optimal is at intersection


# High dimension benchmarks with box and linear constraints (NLP)
# benchmark = "s250" # 3-D, 2 linear inequality constraints, optimal is at intersection
# benchmark = "s251" # 3-D, 1 linear inequality constraints
# benchmark = "Horst4" # 3-D, 4 linear inequality constraints, optimal is at intersection
# benchmark = "Horst5"  # 3-D, 4 linear inequality constraints, optimal is at intersection
# benchmark = "Horst6"  # 3-D, 7 linear inequality constraints, optimal is at intersection
# benchmark = "hs044"  # 4-D, 6 linear inequality constraints, optimal is at intersection (20 vertices)
# benchmark = "hs076"  # 4-D, 3 linear inequality constraints
# benchmark = "Bunnag3" # 5-D, 3 linear inequality constraints, optimal is at intersection
benchmark = "Bunnag6"  # 10-D, 11 linear inequality constraints, excessive number of vertices (1036)
# benchmark = "Bunnag7" # 10-D, 5 linear inequality constraints, optimal is at intersection, excessive number of vertices (594)
# benchmark = "Genocop9"  # 3-D, 5 linear inequality constraints, optimal is at intersection
# benchmark = "Genocop11" # 6-D, 5 linear inequality constraints, optimal is at intersection (92 vertices)

# Mixed integer problems # https://www.minlplib.org/
# benchmark = 'ex1222'  # two continuous variables and one binary variable
# benchmark = 'ex_int'  # two continuous variables, one integer variable and one binary variable

# MIP from MIP library # http://miplib.zib.de/index.html
# benchmark = "gen-ip054"  # 30 integer variables
# benchmark = "gen-ip054_2"  # treat integer variable as categorical variable

# INLP # https://www.minlplib.org/
# benchmark = 'nvs04'  # two integer variables

# Other MIP problems
# benchmark = "Knapsack"  # 6 integer variables
# benchmark = "Knapsack_2" # treat integer variable as categorical variable



# default setting for the benchmarks
isLin_eqConstrained = False
isLin_ineqConstrained = False
Aeq = array([])
beq = array([])
Aineq = array([])
bineq = array([])

# 2-D benchmarks with box constraints
if benchmark == "PWA_example":
    # PWA function
    nc = 2
    nint = 0
    nd = 0
    X_d = []
    lb = array([-1.0, -1.0])
    ub = array([1.0, 1.0])
    fun = lambda x: np.amax([0.8031 * x[0] + 0.0219 * x[1] - 0.3227 * 1,
                             0.0942 * x[0] - 0.5617 * x[1] - 0.1622,
                             0.9462 * x[0] - 0.7299 * x[1] - 0.7141,
                             -0.4799 * x[0] + 0.1084 * x[1] - 0.1210,
                             0.5770 * x[0] + 0.1574 * x[1] - 0.178,
                             0.2458 * x[0] - 0.5823 * x[1] - 0.1997], axis=0)
    maxevals = 100
    fopt0 = -0.148801483645783
    xopt0 = array([0.05461, -0.01469])

    # from pyswarm import pso  # https://pythonhosted.org/pyswarm/
    #
    # xopt0, fopt0 = pso(fun, lb, ub, swarmsize=200,
    #                    minfunc=1e-12, maxiter=10000)

elif benchmark == "camelsixhumps":
    # Camel six-humps function
    nc = 2
    nint = 0
    nd = 0
    X_d = []
    lb = array([-2.0, -1.0])
    ub = array([2.0, 1.0])
    fun = lambda x: ((4.0 - 2.1 * x[0] ** 2 + x[0] ** 4 / 3.0) * x[0] ** 2 + x[0] * x[1] + (4.0 * x[1] ** 2 - 4.0) * x[
        1] ** 2)
    xopt0 = array([[0.0898, -0.0898], [-0.7126, 0.7126]])  # unconstrained optimizers, one per column
    fopt0 = -1.0316  # unconstrained optimum
    maxevals = 100

elif benchmark == "ackley":
    nc = 2
    nint = 0
    nd = 0
    X_d = []
    lb = -5 * ones((nc, 1)).flatten("c")
    ub = -lb
    from math import exp, cos, sqrt, pi

    fun = lambda x: array([-20.0 * exp(-.2 * sqrt(0.5 * (x[0] ** 2 + x[1] ** 2))) - exp(
        0.5 * (cos(2.0 * pi * x[0]) + cos(2.0 * pi * x[1]))) + exp(1.0) + 20.0])
    maxevals = 100

    # compute optimum/optimizer by PSO
    # from pyswarm import pso # https://pythonhosted.org/pyswarm/
    # xopt0, fopt0 = pso(fun, lb, ub, swarmsize=200,
    #                    minfunc=1e-12, maxiter=10000)
    #
    xopt0 = array([0, 0])
    fopt0 = 0

# 2-D benchmarks with box and linear constraints
elif benchmark == "camelsixhumps-linearconstr":
    # Camel six-humps function
    nc = 2
    nint = 0
    nd = 0
    X_d = []
    lb = array([-2.0, -1.0])
    ub = array([2.0, 1.0])
    fun = lambda x: ((4.0 - 2.1 * x[0] ** 2 + x[0] ** 4 / 3.0) * x[0] ** 2 + x[0] * x[1] + (4.0 * x[1] ** 2 - 4.0) * x[
        1] ** 2)

    isLin_eqConstrained = False
    isLin_ineqConstrained = True
    if isLin_eqConstrained:
        Aeq = array([])
        beq = array([])

    if isLin_ineqConstrained:
        Aineq = array([[1.6295, 1],
                       [-1, 4.4553],
                       [-4.3023, -1],
                       [-5.6905, -12.1374],
                       [17.6198, 1]])
        bineq = array([[3.0786],
                       [2.7417],
                       [-1.4909],
                       [1],
                       [32.5198]])

    xopt0 = array([[0.0898, -0.0898], [-0.7126, 0.7126]])  # unconstrained optimizers, one per column
    fopt0 = -1.0316  # unconstrained optimum
    xopt_const = array([0.19341, 0.65879])  # constrained optimizers
    fopt_const = -0.708453  # constrained optimum

    maxevals = 100

elif benchmark == "camelsixhumps-linearconstr_2":
    # Camel six-humps function
    nc = 2
    nint = 0
    nd = 0
    X_d = []
    lb = array([-2.0, -1.0])
    ub = array([2.0, 1.0])
    fun = lambda x: ((4.0 - 2.1 * x[0] ** 2 + x[0] ** 4 / 3.0) * x[0] ** 2 + x[0] * x[1] + (4.0 * x[1] ** 2 - 4.0) * x[
        1] ** 2)

    isLin_eqConstrained = False
    isLin_ineqConstrained = True
    if isLin_eqConstrained:
        Aeq = array([])
        beq = array([])

    if isLin_ineqConstrained:
        Aineq = array([[1.6295, 1],
                       [0.5, 3.875],
                       [-4.3023, -4],
                       [-2, 1],
                       [0.5, -1]])
        bineq = array([[3.0786],
                       [3.324],
                       [-1.4909],
                       [0.5],
                       [0.5]])

    xopt0 = array([[0.0898, -0.0898], [-0.7126, 0.7126]])  # unconstrained optimizers, one per column
    fopt0 = -1.0316  # unconstrained optimum
    xopt_const = array([0.0781, 0.6562])  # constrained optimizers
    fopt_const = -0.9050  # constrained optimum

    maxevals = 100

elif benchmark == "Horst3":
    nc = 2
    nint = 0
    nd = 0
    X_d = []
    lb = array([0.0, 0.0])
    ub = array([1.0, 1.5])
    fun = lambda x: (-x[0] ** 2 + (4 / 3) * x[0] + ((log(1 + x[1])) / (log(exp(1)))) - (4 / 9))

    isLin_eqConstrained = False
    isLin_ineqConstrained = True
    if isLin_eqConstrained:
        Aeq = array([])
        beq = array([])

    if isLin_ineqConstrained:
        Aineq = array([[1, 1 / 10],
                       [1, 1],
                       [-2, 1]
                       ])
        bineq = array([[1],
                       [3 / 2],
                       [1]
                       ])

    # # compute optimum/optimizer by PSO
    # from pyswarm import pso # https://pythonhosted.org/pyswarm/
    # xopt0, fopt0 = pso(fun, lb, ub, swarmsize=200,
    #                    minfunc=1e-12, maxiter=10000)  # unconstrained optimizers, unconstrained optimum

    xopt0 = array([0, 0])  # unconstrained optimizers, one per column
    fopt0 = -0.4444444  # unconstrained optimum
    xopt_const = array([0, 0])  # constrained optimizers
    fopt_const = -0.4444444  # constrained optimum

    maxevals = 50

elif benchmark == "s232":
    nc = 2
    nint = 0
    nd = 0
    X_d = []
    lb = array([0.0, 0.0])
    ub = array([100.0, 100.0])
    fun = lambda x: (-1 * (9 - (x[0] - 3) ** 2) * (x[1] ** 3 / (27 * sqrt(3))))

    isLin_eqConstrained = False
    isLin_ineqConstrained = True
    if isLin_eqConstrained:
        Aeq = array([])
        beq = array([])

    if isLin_ineqConstrained:
        Aineq = array([[-1 / sqrt(3), 1],
                       [-1, -sqrt(3)],
                       [1, sqrt(3)]
                       ])
        bineq = array([[0],
                       [0],
                       [6]
                       ])

    # # compute optimum/optimizer by PSO
    # from pyswarm import pso # https://pythonhosted.org/pyswarm/
    # xopt0, fopt0 = pso(fun, lb, ub, swarmsize=200,
    #                    minfunc=1e-12, maxiter=10000)  # unconstrained optimizers, unconstrained optimum

    xopt0 = array([3, 100])  # unconstrained optimizers, one per column
    fopt0 = -192450.08972987527  # unconstrained optimum
    xopt_const = array([3, 1.7320508075688772])  # constrained optimizers
    fopt_const = -1  # constrained optimum

    maxevals = 50

# High dimension benchmarks with box and linear constraints
elif benchmark == "s250":
    nc = 3
    nint = 0
    nd = 0
    X_d = []
    lb = np.zeros(nc)
    ub = array([20, 11, 40])
    fun = lambda x: (-1 * x[0] * x[1] * x[2])
    xopt_const = array([20, 11, 15])  # constrained optimizers, one per column
    fopt_const = -3300  # constrained optimum

    isLin_eqConstrained = False
    isLin_ineqConstrained = True
    # if isLin_eqConstrained:
    #     # Aeq = array([[1,1,0,0,0],
    #     #              [0,1,1,0,0]]]
    #     # beq = array([[-2],
    #     #              [1]])
    #     Aeq = array([[1,1,0,0,0]])
    #     beq = array([[-2]])

    if isLin_ineqConstrained:
        Aineq = array([[-1, -2, -2],
                       [1, 2, 2]])
        bineq = array([[0],
                       [72]])

    maxevals = 50

elif benchmark == "s251":
    nc = 3
    nint = 0
    nd = 0
    X_d = []
    lb = np.zeros(nc)
    ub = array([42, 42, 42])
    fun = lambda x: (-1 * x[0] * x[1] * x[2])
    xopt_const = array([24, 12, 12])  # constrained optimizers, one per column
    fopt_const = -3456  # constrained optimum

    isLin_eqConstrained = False
    isLin_ineqConstrained = True
    # if isLin_eqConstrained:
    #     # Aeq = array([[1,1,0,0,0],
    #     #              [0,1,1,0,0]]]
    #     # beq = array([[-2],
    #     #              [1]])
    #     Aeq = array([[1,1,0,0,0]])
    #     beq = array([[-2]])

    if isLin_ineqConstrained:
        Aineq = array([[1, 2, 2]])
        bineq = array([[72]])

    maxevals = 50

elif benchmark == "Horst4":
    nc = 3
    nint = 0
    nd = 0
    X_d = []
    lb = np.zeros(nc)
    ub = array([2, 3, 2.8])
    fun = lambda x: (-(abs(x[0] + (1 / 2) * x[1] + (2 / 3) * x[2])) ** (3 / 2))
    xopt_const = array([2, 0, 2])  # constrained optimizers, one per column
    fopt_const = -6.085806194501845  # constrained optimum

    isLin_eqConstrained = False
    isLin_ineqConstrained = True
    # if isLin_eqConstrained:
    #     # Aeq = array([[1,1,0,0,0],
    #     #              [0,1,1,0,0]]]
    #     # beq = array([[-2],
    #     #              [1]])
    #     Aeq = array([[1,1,0,0,0]])
    #     beq = array([[-2]])

    if isLin_ineqConstrained:
        Aineq = array([[-1, 0, 0],
                       [0, -1, -2],
                       [1, 1 / 2, 0],
                       [1, 1, 2]])
        bineq = array([[-1 / 2],
                       [-1],
                       [2],
                       [6]])

    maxevals = 50

elif benchmark == "Horst5":
    nc = 3
    nint = 0
    nd = 0
    X_d = []
    lb = np.zeros(nc)
    ub = array([1.2, 1.2, 1.7])
    fun = lambda x: (-(abs(x[0] + (1 / 2) * x[1] + (2 / 3) * x[2])) ** (3 / 2)) - x[0] ** 2
    xopt_const = array([1.2, 0, 0.8])  # constrained optimizers, one per column
    fopt_const = -3.7220393738285287  # constrained optimum

    isLin_eqConstrained = False
    isLin_ineqConstrained = True
    # if isLin_eqConstrained:
    #     # Aeq = array([[1,1,0,0,0],
    #     #              [0,1,1,0,0]]]
    #     # beq = array([[-2],
    #     #              [1]])
    #     Aeq = array([[1,1,0,0,0]])
    #     beq = array([[-2]])

    if isLin_ineqConstrained:
        Aineq = array([[0, 0, 1],
                       [-2, -2, -1],
                       [1, 1, -1 / 4],
                       [1, 1, 1]])
        bineq = array([[3],
                       [1],
                       [1],
                       [2]])

    maxevals = 50

elif benchmark == "Horst6":
    nc = 3
    nint = 0
    nd = 0
    X_d = []
    lb = np.zeros(nc)
    ub = array([6, 5.0279, 2.6])

    Q = array([[0.992934, - 0.640117, 0.337286],
               [-0.640117, - 0.814622, 0.960807],
               [0.337286, 0.960807, 0.500874]])
    p = array([-0.992372, - 0.046466, 0.891766])
    fun = lambda x: np.transpose(x).dot(Q).dot(x) + p.dot(x)
    xopt_const = array([5.2106555627868909, 5.0279, 0])  # constrained optimizers, one per column
    fopt_const = -32.5793248372817317  # constrained optimum

    isLin_eqConstrained = False
    isLin_ineqConstrained = True
    # if isLin_eqConstrained:
    #     # Aeq = array([[1,1,0,0,0],
    #     #              [0,1,1,0,0]]]
    #     # beq = array([[-2],
    #     #              [1]])
    #     Aeq = array([[1,1,0,0,0]])
    #     beq = array([[-2]])

    if isLin_ineqConstrained:
        Aineq = array([[0.488509, 0.063565, 0.945686],
                       [-0.578592, -0.324014, -0.501754],
                       [-0.719203, 0.099562, 0.445225],
                       [-0.346896, 0.637939, -0.257623],
                       [-0.202821, 0.647361, 0.920135],
                       [-0.983091, -0.886420, -0.802444],
                       [-0.305441, -0.180123, -0.515399]])
        bineq = array([[2.865062],
                       [-1.491608],
                       [0.519588],
                       [1.584087],
                       [2.198036],
                       [-1.301853],
                       [-0.73829]])

    maxevals = 50

elif benchmark == "hs044":
    nc = 4
    nint = 0
    nd = 0
    X_d = []
    lb = zeros(nc)
    ub = 42 * ones(nc)
    fun = lambda x: x[0] - x[1] - x[2] - x[0] * x[2] + x[0] * x[3] + x[1] * x[2] - x[1] * x[3]
    xopt_const = array([0, 3, 0, 4])  # constrained optimizers, one per column
    fopt_const = -15  # constrained optimum

    isLin_eqConstrained = False
    isLin_ineqConstrained = True
    # if isLin_eqConstrained:
    #     # Aeq = array([[1,1,0,0,0],
    #     #              [0,1,1,0,0]]]
    #     # beq = array([[-2],
    #     #              [1]])
    #     Aeq = array([[1,1,0,0,0]])
    #     beq = array([[-2]])

    if isLin_ineqConstrained:
        Aineq = array([[1, 2, 0, 0],
                       [4, 1, 0, 0],
                       [3, 4, 0, 0],
                       [0, 0, 2, 1],
                       [0, 0, 1, 2],
                       [0, 0, 1, 1]])
        bineq = array([[8],
                       [12],
                       [12],
                       [8],
                       [8],
                       [5]])

    maxevals = 50

elif benchmark == "hs076":
    nc = 4
    nint = 0
    nd = 0
    X_d = []
    lb = zeros(nc)
    ub = np.array([1, 3, 1, 1])
    fun = lambda x: (x[0] ** 2) + 0.5 * (x[1] ** 2) + (x[2] ** 2) + 0.5 * x[3] ** 2 - x[0] * x[2] + x[2] * x[3] - x[
        0] - 3 * x[1] + x[2] - x[3]
    xopt_const = array([3 / 11, 23 / 11, 0, 6 / 11])  # constrained optimizers, one per column
    fopt_const = -4.6818181818181818  # constrained optimum

    isLin_eqConstrained = False
    isLin_ineqConstrained = True
    # if isLin_eqConstrained:
    #     # Aeq = array([[1,1,0,0,0],
    #     #              [0,1,1,0,0]]]
    #     # beq = array([[-2],
    #     #              [1]])
    #     Aeq = array([[1,1,0,0,0]])
    #     beq = array([[-2]])

    if isLin_ineqConstrained:
        Aineq = array([[1, 2, 1, 1],
                       [3, 1, 2, -1],
                       [0, -1, -4, 0]])
        bineq = array([[5],
                       [4],
                       [-1.5]])

    maxevals = 50

elif benchmark == "Bunnag3":
    nc = 5
    nint = 0
    nd = 0
    X_d = []
    # since in the obj. fun. x is raised to the power of 0.6, to prevent numerical issues, set lb = 1.0e-6
    lb = array([1.0e-6, 1.0e-6, 1.0e-6, 1.0e-6, 1.0e-6])
    ub = array([3.0, 2.0, 4.0, 4.0, 2.0])
    fun = lambda x: (x[0] ** 0.6 + x[1] ** 0.6 + x[2] ** 0.6 - 4 * x[2] - 2 * x[3] + 5 * x[4])
    xopt_const = array([0, 0, 4, 1.33333, 0])  # constrained optimizers, one per column
    fopt_const = -16.369269956672596  # constrained optimum

    isLin_eqConstrained = False
    isLin_ineqConstrained = True
    # if isLin_eqConstrained:
    #     # Aeq = array([[1,1,0,0,0],
    #     #              [0,1,1,0,0]]]
    #     # beq = array([[-2],
    #     #              [1]])
    #     Aeq = array([[1,1,0,0,0]])
    #     beq = array([[-2]])

    if isLin_ineqConstrained:
        Aineq = array([[1, 0, 0, 2, 0],
                       [3, 0, 0, 3, 1],
                       [0, 2, 0, 4, 2]])
        bineq = array([[4],
                       [4],
                       [6]])

    maxevals = 50

elif benchmark == "Bunnag6":
    nc = 10
    nint = 0
    nd = 0
    X_d = []
    lb = zeros(nc)
    ub = ones(nc)


    def fun(x):
        d = [-20, -80, -20, -50, -60, -90, 0]
        sum1 = 0
        sum2 = 0
        sum3 = 0

        for i in range(7):
            sum1 += d[i] * x[i]
            sum3 += x[i] ** 2
        for i in range(7, 10):
            sum2 += x[i]

        f = sum1 + 10 * sum2 - 5 * sum3

        return f


    xopt_const = array(
        [1, 0.90754716, 0, 1, 0.71509433, 1, 0, 0.91698113, 1, 1])  # constrained optimizers, one per column
    fopt_const = -268.0146300421110368  # constrained optimum

    isLin_eqConstrained = False
    isLin_ineqConstrained = True
    # if isLin_eqConstrained:
    #     # Aeq = array([[1,1,0,0,0],
    #     #              [0,1,1,0,0]]]
    #     # beq = array([[-2],
    #     #              [1]])
    #     Aeq = array([[1,1,0,0,0]])
    #     beq = array([[-2]])

    if isLin_ineqConstrained:
        Aineq = array([
            [-2, -6, -1, 0, -3, -3, -2, -6, -2, -2],
            [6, -5, 8, -3, 0, 1, 3, 8, 9, -3],
            [-5, 6, 5, 3, 8, -8, 9, 2, 0, -9],
            [9, 5, 0, -9, 1, -8, 3, -9, -9, -3],
            [-8, 7, -4, -5, -9, 1, -7, -1, 3, -2],
            [-7, -5, -2, 0, -6, -6, -7, -6, 7, 7],
            [1, -3, -3, -4, -1, 0, -4, 0, 6, 0],
            [1, -2, 6, 9, 0, -7, 9, -9, -6, 4],
            [-4, 6, 7, 2, 2, 0, 6, 6, -7, 4],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
        ])
        bineq = array([
            [-4],
            [22],
            [-6],
            [-23],
            [-12],
            [-3],
            [1],
            [12],
            [15],
            [9],
            [-1]
        ])

    maxevals = 50

elif benchmark == "Bunnag7":
    nc = 10
    nint = 0
    nd = 0
    X_d = []
    lb = zeros(nc)
    ub = ones(nc)

    d = [48, 42, 48, 45, 44, 41, 47, 42, 45, 46]
    fun = lambda x: (sum(d * x) - 50 * (sum((x ** 2))))

    xopt_const = array([1, 0, 0, 1, 1, 1, 0, 1, 1, 1])  # constrained optimizers, one per column
    fopt_const = -39  # constrained optimum

    isLin_eqConstrained = False
    isLin_ineqConstrained = True
    # if isLin_eqConstrained:
    #     # Aeq = array([[1,1,0,0,0],
    #     #              [0,1,1,0,0]]]
    #     # beq = array([[-2],
    #     #              [1]])
    #     Aeq = array([[1,1,0,0,0]])
    #     beq = array([[-2]])

    if isLin_ineqConstrained:
        Aineq = array([
            [-2, -6, -1, 0, -3, -3, -2, -6, -2, -2],
            [6, -5, 8, -3, 0, 1, 3, 8, 9, -3],
            [-5, 6, 5, 3, 8, -8, 9, 2, 0, -9],
            [9, 5, 0, -9, 1, -8, 3, -9, -9, -3],
            [-8, 7, -4, -5, -9, 1, -7, -1, 3, -2]
        ])
        bineq = array([
            [-4],
            [22],
            [-6],
            [-23],
            [-12]
        ])

    maxevals = 100

elif benchmark == "Genocop9":
    nc = 3
    nint = 0
    nd = 0
    X_d = []
    lb = zeros(nc)
    ub = 3 * ones(nc)
    fun = lambda x: (-((3 * x[0] + x[1] - 2 * x[2] + 0.8) / (2 * x[0] - x[1] + x[2]) + (4 * x[0] - 2 * x[1] + x[2]) / (
                7 * x[0] + 3 * x[1] - x[2])))
    xopt_const = array([1, 0, 0])  # constrained optimizers, one per column
    fopt_const = -2.47142857142857153  # constrained optimum

    isLin_eqConstrained = False
    isLin_ineqConstrained = True
    # if isLin_eqConstrained:
    #     # Aeq = array([[1,1,0,0,0],
    #     #              [0,1,1,0,0]]]
    #     # beq = array([[-2],
    #     #              [1]])
    #     Aeq = array([[1,1,0,0,0]])
    #     beq = array([[-2]])

    if isLin_ineqConstrained:
        Aineq = array([
            [1, 1, -1],
            [-1, 1, -1],
            [12, 5, 12],
            [12, 12, 7],
            [-6, 1, 1]
        ])
        bineq = array([
            [1],
            [-1],
            [34.8],
            [29.1],
            [-4.1]
        ])

    maxevals = 50

elif benchmark == "Genocop11":
    nc = 6
    nint = 0
    nd = 0
    X_d = []
    lb = zeros(nc)
    ub = array([10, 10, 10, 1, 1, 2])
    fun = lambda x: (6.5 * x[0] - 0.5 * x[0] ** 2 - x[1] - 2 * x[2] - 3 * x[3] - 2 * x[4] - x[5])
    xopt_const = array([0, 6, 0, 1, 1, 0])  # constrained optimizers, one per column
    fopt_const = -11  # constrained optimum

    isLin_eqConstrained = False
    isLin_ineqConstrained = True
    # if isLin_eqConstrained:
    #     # Aeq = array([[1,1,0,0,0],
    #     #              [0,1,1,0,0]]]
    #     # beq = array([[-2],
    #     #              [1]])
    #     Aeq = array([[1,1,0,0,0]])
    #     beq = array([[-2]])

    if isLin_ineqConstrained:
        Aineq = array([
            [1, 2, 8, 1, 3, 5],
            [-8, -4, -2, 2, 4, -1],
            [2, 0.5, 0.2, -3, -1, -4],
            [0.2, 2, 0.1, -4, 2, 2],
            [-0.1, -0.5, 2, 5, -5, 3]
        ])
        bineq = array([
            [16],
            [-1],
            [24],
            [12],
            [2]
        ])

    maxevals = 50



# Mixed integer problems
elif benchmark == 'ex1222':
    nc = 2
    nint = 0
    nd = 1
    X_d = [2]
    lb = array([0.2, -2.22554, 0])
    ub = array([1, -1, 1])
    fun = lambda x: 5 * (-0.5 + x[0]) ** 2 - x[1] + 0.7 * x[2] + 0.8
    maxevals = 50

    fopt0 = 1.8
    xopt0 = array([0.5, -1, 0])

elif benchmark == 'ex_int':
    nc = 2
    nint = 1
    nd = 1
    X_d = [2]
    lb = array([0.2, -4.5, -3])
    ub = array([1, -2, 4])
    fun = lambda x: 5 * (-0.5 + x[0]) ** 2 - x[2] + 0.7 * x[3] + 0.8

    isLin_eqConstrained = True
    isLin_ineqConstrained = False
    if isLin_eqConstrained:
        # Aeq = array([[1,1,0,0,0],
        #              [0,1,1,0,0]])
        # beq = array([[-2],
        #              [1]])
        Aeq = array([[1, 1, 0, 0, 0]])
        beq = array([[-2]])

    if isLin_ineqConstrained:
        Aineq = array([[1, 1, 0, 0, 0],
                       [0, 1, 1, 0, 0]])
        bineq = array([[-2],
                       [1]])

    maxevals = 50

    fopt_const = -3.2
    xopt0 = array([0.5, -2.5, 4, 0])

# INLP
elif benchmark == 'nvs04':
    nc = 0
    nint = 2
    nd = 0
    X_d = []
    lb = array([0, 0])
    ub = array([200, 200])
    fun = lambda x: 100 * (-(0.6 + x[0]) ** 2 + 0.5 + x[1]) ** 2 + (0.4 - x[0]) ** 2

    isLin_eqConstrained = False
    isLin_ineqConstrained = False
    if isLin_eqConstrained:
        # Aeq = array([[1,1,0,0,0],
        #              [0,1,1,0,0]])
        # beq = array([[-2],
        #              [1]])
        Aeq = array([])
        beq = array([])

    if isLin_ineqConstrained:
        Aineq = array([])
        bineq = array([])

    maxevals = 100

    fopt0 = 0.720000000000006
    xopt0 = array([1, 2])


##
elif benchmark == "gen-ip054":
    nc = 0
    nint = 30
    nd = 0
    X_d = []

    name, objective_name, row_names, col_names, col_types, types, c, A, rhs_names, rhs, bnd_names, bnd = smps.load_mps(
        'gen-ip054.mps')

    lb = bnd['bnd']['LO']
    # ub = bnd['bnd']['UP']
    ub = 5 * np.ones((nint))
    fun = lambda x: c.dot(x)

    isLin_eqConstrained = False
    isLin_ineqConstrained = True
    if isLin_eqConstrained:
        Aeq = array([])
        beq = array([])

    if isLin_ineqConstrained:
        Aineq = A
        bineq = rhs['rhs'].reshape((-1, 1))

    maxevals = 200

    fopt_const = 6840.966
    xopt_const = array([4, 1, 0, 3, 2, 1, 0, 0, 0, 1, 0, 2, 1, 1, 1, 0, 0, 1, 0, 0, 2, 2, 0, 0, 5, 0, 1, 2, 0, 1])

elif benchmark == "gen-ip054_2":
    nc = 0
    nint = 0
    nd = 30
    X_d = 30 * [6]

    name, objective_name, row_names, col_names, col_types, types, c, A, rhs_names, rhs, bnd_names, bnd = smps.load_mps(
        'gen-ip054.mps')

    # lb = bnd['bnd']['LO']
    # # ub = bnd['bnd']['UP']
    # ub = 5*np.ones((nint))
    lb = np.zeros((nd))
    ub = 5 * np.ones((nd))
    fun = lambda x: c.dot(x)

    isLin_eqConstrained = False
    isLin_ineqConstrained = True
    if isLin_eqConstrained:
        Aeq = array([])
        beq = array([])

    if isLin_ineqConstrained:
        cc = A.copy()
        AAA = np.ones((27, 30 * 6))
        for i in range(27):
            for j in range(30):
                AAA[i, 6 * j:6 * (j + 1)] = cc[i, j] * np.arange(6)
        Aineq = AAA
        bineq = rhs['rhs'].reshape((-1, 1)).copy()

    maxevals = 200

    fopt_const = 6840.966
    xopt_const = array([4, 1, 0, 3, 2, 1, 0, 0, 0, 1, 0, 2, 1, 1, 1, 0, 0, 1, 0, 0, 2, 2, 0, 0, 5, 0, 1, 2, 0, 1])

##
elif benchmark == "Knapsack":
    nc = 0
    nint = 6
    nd = 0
    X_d = []

    lb = np.zeros((nint))
    ub = np.ones((nint))
    c = np.array([10, 13, 18, 32, 7, 15])
    fun = lambda x: -c.dot(x)

    isLin_eqConstrained = False
    isLin_ineqConstrained = True
    if isLin_eqConstrained:
        Aeq = array([])
        beq = array([])

    if isLin_ineqConstrained:
        w = array([11, 15, 20, 35, 10, 33]).reshape((1, -1))
        Aineq = w
        bineq = array([47]).reshape((-1, 1))

    maxevals = 30

    fopt_const = -42
    xopt_const = array([1, 0, 0, 1, 0, 0])

elif benchmark == "Knapsack_2":
    nc = 0
    nint = 0
    nd = 6
    X_d = nd * [2]

    lb = np.zeros((nd))
    ub = np.ones((nd))
    c = np.array([10, 13, 18, 32, 7, 15])
    fun = lambda x: -c.dot(x)

    isLin_eqConstrained = False
    isLin_ineqConstrained = True
    if isLin_eqConstrained:
        Aeq = array([])
        beq = array([])

    if isLin_ineqConstrained:
        w = array([11, 15, 20, 35, 10, 33]).reshape((1, -1))
        Aineq = np.zeros((1, 6 * 2))
        for i in range(6):
            Aineq[0, 2 * i:2 * (i + 1)] = w[0, i] * arange(2)
        bineq = array([47]).reshape((-1, 1))

    maxevals = 30

    fopt_const = -42
    xopt_const = array([1, 0, 0, 1, 0, 0])


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
    X = np.array(optimizer1.X)
    fbest_seq1 = list(map(fun, X[optimizer1.ibest_seq]))
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
    X = np.array(optimizer1.X)
    fbest_seq1 = optimizer1.fbest_seq
####################################################################################

nvars = nc+nint+nd

# plt.plot(arange(0, maxevals), minf, color=[0.8500, 0.3250, 0.0980])
plt.plot(arange(0, maxevals), fbest_seq1, color=(.6, 0, 0), linewidth=1.0)
plt.scatter(arange(0, maxevals), fbest_seq1, color=(.6, 0, 0), marker='o', linewidth=1.0)

if runPWASp:
    plt.xlabel("preference queries")
    thelegend = ["PWASp"]
elif runPWAS:
    plt.xlabel("number of fun. eval.")
    thelegend = ["PWAS"]
plt.title("Best value of latent function")

plt.grid()

if not (isLin_ineqConstrained or isLin_eqConstrained):
    plt.plot(arange(0, maxevals), fopt0 * ones(maxevals), linestyle='--',
             color=(0, 0, .6), linewidth=2.0)
else:
    plt.plot(arange(0, maxevals), fopt_const * ones(maxevals), linestyle='--',
             color=(0, 0, .6), linewidth=2.0)

thelegend.append("optimum")
plt.legend(thelegend)
plt.show()


if nvars == 2:
    fig, ax = plt.subplots(figsize=(14, 7))
    if not benchmark == "s232":
        [x, y] = meshgrid(arange(lb[0], ub[0], .01), arange(lb[1], ub[1], .01))
    elif benchmark == "s232" and feasible_sampling:
        [x, y] = meshgrid(arange(lb[0], 8, 0.1), arange(lb[1], ub[1], 0.1))
    elif benchmark == "s232" and (not feasible_sampling):
        [x, y] = meshgrid(arange(lb[0], ub[0], 1), arange(lb[1], ub[1], 1))
    z = zeros(x.shape)
    for i in range(0, x.shape[0]):
        for j in range(0, x.shape[1]):
            z[i, j] = fun(array([x[i, j], y[i, j]]))

    plt.contour(x, y, z, 100, alpha=.4)
    plt.plot(X[0:nsamp, 0], X[0:nsamp, 1], "*", color=[237 / 256, 177 / 256, 32 / 256], markersize=11)
    plt.plot(X[nsamp:maxevals + 1, 0], X[nsamp:maxevals + 1, 1], "*", color=[0.3, 0, 1], markersize=11)
    plt.plot(xopt0[0,], xopt0[1,], "o", color=[0, 0.4470, 0.7410], markersize=15)
    if (isLin_eqConstrained or isLin_ineqConstrained):
        plt.plot(xopt_const[0,], xopt_const[1,], "s", color=[0, 0.9, 0.1], markersize=15)
    plt.plot(xopt1[0], xopt1[1], "*", color=[0.8500, 0.3250, 0.0980], markersize=15)

    import matplotlib.patches as mpatches
    from matplotlib.collections import PatchCollection

    patches = []

    if (isLin_eqConstrained or isLin_ineqConstrained):
        if (benchmark =="camelsixhumps-linearconstr"):  # plot for benchmark camelsixhumps-linearconstr
            V = array([[0.4104, -0.2748], [0.1934, 0.6588], [1.3286, 0.9136],
                   [1.8412, 0.0783], [1.9009, -0.9736]])

        elif (benchmark =="camelsixhumps-linearconstr_2"):  # plot for benchmark camelsixhumps-linearconstr
            V = array([[1.48,0.667], [0.168,0.836], [ -0.041,0.417],
                   [ 0.554, -0.223], [1.68,0.34]])

        elif (benchmark == "Horst3"):  # plot for benchmark Horst3
            V = array([[0.9444, 0.556], [0.1667, 1.333], [0, 1],
                       [0, 0], [1, 0]])

        elif (benchmark == "s232"):  # plot for benchmark s232
            V = array([[3, 1.732], [0, 0], [6, 0]])

        polygon = mpatches.Polygon(V, True)
        patches.append(polygon)

    if (isLin_eqConstrained or isLin_ineqConstrained):
        collection = PatchCollection(patches, edgecolor=[0, 0, 0], facecolor=[.5, .5, .5], alpha=0.6)
        ax.add_collection(collection)

    plt.grid()
    plt.show()

if benchmark == "s232":
    fig, ax = plt.subplots(figsize=(14, 7))
    [x_, y_] = meshgrid(arange(lb[0], 7, 0.01), arange(lb[1], 2, 0.01))
    z_ = zeros(x_.shape)
    for i in range(0, x_.shape[0]):
        for j in range(0, x_.shape[1]):
            z_[i, j] = fun(array([x_[i, j], y_[i, j]]))

    plt.contour(x_, y_, z_, 100, alpha=.4)
    if feasible_sampling:
        plt.plot(X[0:nsamp, 0], X[0:nsamp, 1], "*", color=[237 / 256, 177 / 256, 32 / 256], markersize=11)
    plt.plot(X[nsamp:maxevals + 1, 0], X[nsamp:maxevals + 1, 1], "*", color=[0.3, 0, 1], markersize=11)
    if (isLin_eqConstrained or isLin_ineqConstrained):
        plt.plot(xopt_const[0,], xopt_const[1,], "s", color=[0, 0.9, 0.1], markersize=15)
    plt.plot(xopt1[0], xopt1[1], "*", color=[0.8500, 0.3250, 0.0980], markersize=15)

    import matplotlib.patches as mpatches
    from matplotlib.collections import PatchCollection

    patches = []

    V = array([[3, 1.732], [0, 0], [6, 0]])
    polygon = mpatches.Polygon(V, True)
    patches.append(polygon)

    collection = PatchCollection(patches, edgecolor=[0, 0, 0], facecolor=[.5, .5, .5], alpha=0.6)
    ax.add_collection(collection)

    plt.grid()
    plt.show()








