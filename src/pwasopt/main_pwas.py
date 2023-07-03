"""
PWAS -  Optimization with mixed variables using Piecewise Affine surrogate

[1] M. Zhu and A. Bemporad, “Global and Preference-based Optimization
    with Mixed Variables using Piecewise Affine Surrogates,”
     arXiv preprint arXiv:2302.04686, 2023.

reference code:
                - PARC package by A. Bemporad, 2021, http://cse.lab.imtlucca.it/~bemporad/parc
                - GLIS package by A.Bemporad & M. Zhu, 2023, https://github.com/bemporad/GLIS

(C) 2021-2023 Mengjia Zhu, Alberto Bemporad
"""

from pyparc.parc import PARC # https://github.com/bemporad/PyPARC, https://pypi.org/project/pyparc/
from pwasopt.prob_setup import *
from pwasopt.sample import *
from pwasopt.acquisition import *
from pwasopt.categorical_encoder import *
from pwasopt.integ_encoder import *

import time


class PWAS:
    """
    Main class for PWAS
    Note:
            - The optimization variable are ordered in this way: [continuous vars, integer vars, categorical vars]
            -  When fitting the surrogate, the integer variables (ordinal) are treated as
                - continuous variables if the number of possible integer values are large (see 'int_encoded' in prob_setup.py)
                    - but when query the next point, integer variables are considered as integer variables to enhance constraint satisfication
                - categorical variables, i.e., one-hot encoded if the number of possible integer values are small
                    - the relevant constraints are updated accordingly (see prob_setup.py for more details)
            - Each categorical variable need to be first ordinal encoded as 0 to X_d[i] when specify the problem, then
                - Specify the lower and upper bounds of the categorical variables as follows:
                    - Lower bounds: 0
                    - Upper bounds: X_d[i] -1  (number of possible classes for that caategorical variable -1)
                    - This is to ease the random sample generation in sample.py (generate then encode)
            - If the problem is linearly equality/inequality constrained, when provide the coefficient matrix Aeq/Aineq,
                do not forget to include the columns for EACH options of the categorical variables, if categorical variable exists
            - PARC package (parc.py) is used to fit the PWA surrogate
            - since numpy array cannot have integer and float variable types together, the user might need to explicitly declare X[nc:nci].dtype(int)
                if strict integer variable is needed.
            - When solve the MILP, it will first attempt to solve via GUROBI, if GUROBI is not available, GLPK will be use
                - User may choose to switch to other solvers by replacing the ones in the acquisition.py
    """

    def __init__(self, f0, lb, ub, delta_E, nc, nint, nd, X_d, nsamp, maxevals, feasible_sampling=True,
                 isLin_eqConstrained=None, Aeq=[], beq=[], isLin_ineqConstrained=None, Aineq=[], bineq=[],
                 K= None, scale_vars = True, shrink_range = True, alpha=1.0e-5, sigma=1, separation=None, maxiter=100, cost_tol=1e-4,
                 min_number=None, fit_on_partition=True, softmax_solver='lbfgs', softmax_maxiter=None, beta=None,
                 initialization=None, verbose=None, categorical=False, timelimit=None, epsDeltaF=None,
                 acq_stage = 'multi-stage', sepvalue=None, synthetic_dm = False):

        """ PWAS - Optimization with mixed variables using Piecewise Affine surrogate
        (C) Jan, 2023, M.Zhu

        Parameters:
        -----------
        f0: function/simulator
             function or simulator used to evaluate the decision/optimization variable
        lb: np array
            lower bounds on the continuous and integer variables with the following order: [continuous, integers]
        ub: np array
            upper bounds on the continuous and integer variables with the following order: [continuous, integers]
        delta_E: float
            the exploration parameter in the acquisition function, which trades-off between the exploitation of the surrogate and the exploration of the exploration function
        nc: int
            number of continuous variables in the optimization variable
        nint: int
            number of integer variables in the optimization variable
        nd: int
            number of categorical variables in the optimization variable
        X_d: list with int elements, dimension: (1 by nd)
            each element in X_d represents the number of options for each categorical variables
            e.g., if nd = 2, nd_1 has 2 possible categories and nd_2 has 4 possible categories, X_d = [2, 4]
        nsamp: int
            number of initial samples
        maxevals: int
            number of maximum function evaluations
        feasible_sampling: bool
            if True, initial samples obtained satisfies the known constraints (if there is any)
            if False, constraints are ignored during initial sampling stage
        isLin_eqConstrained: bool
            if True, the problem has Linear equality constraints
            if False, the problem does not have Linear equality constraints
        (Aeq x  = beq)
        Aeq: np array, dimension: (# of linear eq. const by n_encoded), where n_encoded is the length of the optimization variable AFTER being encoded
            the coefficient matrix for the linear equality constraints
        beq: np array, dimension: (n_encode by 1)
            the RHS of the linear eq. constraints
        isLin_ineqConstrained: bool
            if True, the problem has Linear inequality constraints
            if False, the problem does not have Linear inequality constraints
        (Aineq x <= bineq)
        Aineq:np array, dimension: (# of linear ineq. const by n_encoded)
            the coefficient matrix for the linear inequality constraints
        bineq: np array, dimension: (n_encode by 1)
            the RHS of the linear ineq. constraints
        K: int
            number of linear affine regressor/classifiers in PWA predictor
        scale_vars: bool
            if True, scale the continuous and integer variables to the range of [-1,1]
            if Flase, scaling is not performed
        shrink_range: bool
            if True, shrink the continuous and integer variables further according to the linear equality constraints
            if Flase, shrink range is not performed

        PARC related parameters, see the definition in PARC package:
            alpha, sigma, separation, maxiter, cost_tol, min_number, fit_on_partition, softmax_solver, softmax_maxiter, beta,
                initialization, verbose, categorical

        timelimit: float
            maximum time allowed for the solver in PULP to solve the MILP/LP/MIP
        epsDeltaF: float
            the tolerance for the difference between function evaluations
        acq_stage: str
            whether to solve the acquisition step in one-stage or multiple stages
            either 'one-stage' or 'multi-stage'

        parameters not used: sepvalue, synthetic_dm (parameters for PWASp)

        """

        # obtain the problem setup
        isPref = 0
        self.prob = problem_defn(isPref,f0, lb, ub, delta_E, nc, nint, nd, X_d, nsamp, maxevals, feasible_sampling,
                                 isLin_eqConstrained, Aeq, beq, isLin_ineqConstrained, Aineq, bineq,
                                 K, scale_vars, shrink_range, alpha, sigma, separation, maxiter, cost_tol, min_number,
                                 fit_on_partition, softmax_solver, softmax_maxiter, beta, initialization,
                                 verbose, categorical, timelimit, epsDeltaF, acq_stage, sepvalue, synthetic_dm)

        # obtain the encoder if categorical variables are involved
        if nd > 0:
            EC_cat = cat_encoder(self.prob)
            self.encoder_cat = EC_cat.cat_encoder()

        # obtain the encoder if integer variables are encoded
        if self.prob.int_encoded:
            EC_int = integ_encoder(self.prob)
            self.encoder_int = EC_int.integ_encoder()

        self.AL = active_learn(self.prob)

        self.isInitialized = False
        self.X = list()  # decision variable in the original format (X in the paper)
        self.Xs = list()  # the scaled and/or encoded decision vector (\bar X in the paper)
        self.F = list()  # function/simulation/experiment evaluations
        self.Fmin = np.inf
        self.Fmax = -np.inf
        self.iter = 0
        self.xnext = None
        self.xsnext = None
        self.fbest = np.inf
        self.ibest = None
        self.xbest = None
        self.xsbest = None
        self.fbest_seq = list()  # keep track of the current best evaluation
        self.ibest_seq = list()  # keep track of the index of the current best decision vector
        self.isfeas_seq = list() # keep track of the feasibility of the tested decision vectors
        self.time_fun_eval = list() # keep track of the CPU time used to evaluation one function/simulation/experiment
        self.time_opt_acquisition = list()  # keep track of the CPU time used to optimize the acq. fun. at each iteration
        self.time_fit_surrogate = list() # keep track of the CPU time used to fit the surrogate fun. at each iteration

    def isKnownFeasible(self, xs):
        """
        Check the feasibility of sample xs w.r.t known constraints
        """
        Aeq = self.prob.Aeq
        beq = self.prob.beq
        Aineq = self.prob.Aineq
        bineq = self.prob.bineq

        isfeas = True
        if self.isLin_eqConstrained:
            isfeas = isfeas and all(Aeq @ xs <= beq + 1.0e-8)
            isfeas = isfeas and all(-Aeq @ xs <= -beq + 1.0e-8)
        if self.isLin_ineqConstrained:
            isfeas = isfeas and all(Aineq @ xs <= bineq+ 1.0e-8)
        return isfeas


    def initialize(self):
        """
        Initialize the problem
            - obtain the initial samples to query

        return:
            self.xnext: 1D array
                the initial sample to query
        """
        s = init_sampl(self.prob)
        Xs, X = s.initial_sampling()
        self.X = X
        self.Xs = Xs
        self.xnext = self.X[0]
        self.xsnext = self.Xs[0]
        self.isInitialized = True
        if self.prob.nint > 0:
            self.xnext[self.prob.nc:self.prob.nci] = np.round(
                self.xnext[self.prob.nc:self.prob.nci])
        return self.xnext

    def update(self, f_val):
        """
        - Update the relevant variables w.r.t the newly queried sample
        - And then solve the optimization problem on the updated acquisition function to obtain the next point to query

        - Note:
            - initial samples are always feasible wrt known constraints if self.prob.feasible_sampling = True
            - actively generated samples are always feasible wrt known constraints (constraints are enforced in the MILP)

        Input:
            f_val: float
                evaluation of the objective function at x (the last queried point)
        Return:
            self.xnext: 1D-array
                The next point to query
        """

        nc = self.prob.nc
        nint = self.prob.nint
        nint_encoded = self.prob.nint_encoded
        nci = self.prob.nci
        nci_encoded = self.prob.nci_encoded
        nd = self.prob.nd
        if nd > 0:
            EC_cat = cat_encoder(self.prob)
        int_encoded = self.prob.int_encoded
        if int_encoded:
            EC_int = integ_encoder(self.prob)
        nvars_encoded = self.prob.nvars_encoded
        sum_X_d = self.prob.sum_X_d

        acq_stage = self.prob.acq_stage

        self.F.append(f_val)
        f0 = f_val

        # TODO: for future, introduce objective transformations like in GLIS package

        F = np.array(self.F)

        x = self.xnext # this was either computed at the previous call after n_initial_random iterations or iterated from the initial samples
        xs = self.xsnext

        if self.iter < self.prob.nsamp:
            isfeas = True
            if not self.prob.feasible_sampling:
                isfeas = self.isKnownFeasible(xs)
            self.time_opt_acquisition.append(0.)
            self.time_fit_surrogate.append(0.)
        else:
            isfeas = True  # actively generated samples are always feasible wrt known constraints

        if isfeas and f_val < self.fbest:
            self.fbest = f_val
            self.ibest = self.iter
            self.xbest = x.copy()
            self.xsbest = xs.copy()

        self.ibest_seq.append(self.ibest)
        self.fbest_seq.append(self.fbest)
        self.isfeas_seq.append(isfeas)
        ind_feas = [i for i, x in enumerate(self.isfeas_seq) if x == True]

        if isfeas:
            self.Fmax = max(self.Fmax, f0)
            self.Fmin = min(self.Fmin, f0)

        if self.prob.verbose > 0:
            self.results_display(self.iter + 1, x, f_val, self.fbest, self.ibest + 1)

        if self.iter >= self.prob.nsamp - 1:
            Xs_all = self.Xs.copy()
            F_all = np.array(self.F)
            X_curr = Xs_all[ind_feas, ].copy()  # feasible Xs tested
            F_curr = F_all[ind_feas, ].copy()  # feasible F evaluated

            # use PARC package to fit the PWA surrogate
            predictor = PARC(K=self.prob.K, alpha=self.prob.alpha, sigma=self.prob.sigma,
                             separation=self.prob.separation,
                             maxiter=self.prob.maxiter, cost_tol=self.prob.cost_tol, min_number=self.prob.min_number,
                             fit_on_partition=self.prob.fit_on_partition,
                             softmax_solver=self.prob.softmax_solver, softmax_maxiter=self.prob.softmax_maxiter,
                             beta=self.prob.beta, verbose=0)

            t0 = time.time()
            predictor.fit(X_curr, F_curr, self.prob.categorical)
            self.time_fit_surrogate.append(time.time() - t0)

            Kf = predictor.K  # final number of partitions
            a = predictor.coef_
            b = predictor.intercept_
            omega = predictor.omega
            gamma = predictor.gamma

            dF = self.Fmax - self.Fmin
            if dF == -np.inf:  # no feasible samples found so far
                dF_ = np.array([1.])
            else:
                dF_ = max(dF, self.prob.epsDeltaF)

            N = self.iter + 1
            t0 = time.time()
            skip_z = False
            if acq_stage == 'multi-stage':
                if nd >0:
                    z1 = self.AL.discrete_explore(X_curr[:, nci_encoded:], self.xsbest[:nci_encoded].reshape(nci_encoded, 1), a, b,
                                             N, omega, gamma, Kf, dF_)
                    if np.isnan(z1).any():
                        print(
                            'The optimal solution is not reached within the timeLimit, solution is sampled using \'acq_surrogate\' function in acquisition.py.')
                        if int_encoded:
                            z = self.AL.acq_surrogate_intEncoded(a, b, omega, gamma, Kf, dF_)
                        else:
                            z = self.AL.acq_surrogate(a, b, omega, gamma, Kf, dF_)
                        skip_z = True
                else:
                    z1 = np.array([])
                if not skip_z:
                    z = z1

                if nint>0:
                    if not skip_z:
                        if int_encoded:
                            z2 = self.AL.integ_explore_intEncoded(X_curr[:, nc:nci_encoded], self.xsbest[:nc].reshape(nc, 1), z1.reshape(sum_X_d, 1), a,
                                                  b, N, omega, gamma, Kf, dF_)
                        else:
                            z2 = self.AL.integ_explore(X_curr[:, nc:nci_encoded], self.xsbest[:nc].reshape(nc, 1),z1.reshape(sum_X_d, 1), a, b, N, omega, gamma, Kf, dF_)
                            if np.isnan(z2).any():
                                print(
                                    'The optimal solution is not reached within the timeLimit, solution is sampled using \'acq_surrogate\' function in acquisition.py.')
                                if int_encoded:
                                    z = self.AL.acq_surrogate_intEncoded(a, b, omega, gamma, Kf, dF_)
                                else:
                                    z = self.AL.acq_surrogate(a, b, omega, gamma, Kf, dF_)
                                skip_z = True
                else:
                    z2 = np.array([])
                if not skip_z:
                    z = np.hstack((z2, z))

                if nc>0:
                    if not skip_z:
                        z3 = self.AL.cont_explore(X_curr[:, :nc], z2.reshape(nint_encoded, 1), z1.reshape(sum_X_d, 1), a, b,
                                             N, omega, gamma, Kf, dF_)
                        if np.isnan(z3).any():
                            print(
                                'The optimal solution is not reached within the timeLimit, solution is sampled using \'acq_surrogate\' function in acquisition.py.')
                            if int_encoded:
                                z = self.AL.acq_surrogate_intEncoded(a, b, omega, gamma, Kf, dF_)
                            else:
                                z = self.AL.acq_surrogate(a, b, omega, gamma, Kf, dF_)
                            skip_z = True
                else:
                    z3 = np.array([])
                if not skip_z:
                    z = np.hstack((z3, z))

            elif acq_stage == 'one-stage':
                # Solve the acquisition step in one stage
                if int_encoded:
                    z = self.AL.acq_explore_intEncoded(X_curr, a, b, N, omega, gamma, Kf, dF_)
                else:
                    z = self.AL.acq_explore(X_curr, a, b, N, omega, gamma, Kf, dF_)
                if np.isnan(z).any():
                    print(
                        'The optimal solution is not reached within the timeLimit, solution is sampled using \'acq_surrogate\' function in acquisition.py.')
                    z = self.AL.acq_surrogate(a, b, omega, gamma, Kf, dF_)

            else:
                errstr_acq_stage = "acq_stage can only be 'one-stage' or 'multi-stage', please check the string assigned"
                print(errstr_acq_stage)
                sys.exit(1)
            self.time_opt_acquisition.append(time.time() - t0)

            z_decoded = z.copy()
            if int_encoded and nd > 0:
                z_decoded_int = EC_int.decode(z.reshape(1, nvars_encoded), self.encoder_int)
                z_decoded = EC_cat.decode(z_decoded_int, self.encoder_cat)
            elif int_encoded:
                z_decoded = EC_int.decode(z.reshape(1, nvars_encoded), self.encoder_int)
            elif nd > 0:
                z_decoded = EC_cat.decode(z.reshape(1, nvars_encoded), self.encoder_cat)

            self.xsnext = z.T.reshape(nvars_encoded)
            self.Xs = np.vstack((self.Xs,self.xsnext))
            self.xnext = z_decoded.T.reshape(self.prob.nvars) * self.prob.dd_nvars+ self.prob.d0_nvars
            if nint >0:
                self.xnext[nc:nci] = np.round(self.xnext[nc:nci])

            self.X.append(self.xnext)

        else:
            self.xnext = self.X[self.iter + 1]
            self.xsnext = self.Xs[self.iter + 1]

        self.iter += 1
        return self.xnext

    def solve(self):
        """
        If the simulator/fun have already be integrated with the PWAS solver,
            - use solve() to solve the problem directly

        Return:
            self.xbest: 1D-array
                the best x sampled
            self.fbest: float
                function evaluation at xbest
        """
        t_all = time.time()
        x = self.initialize()  # x is unscaled and/or decoded
        fun = self.prob.f

        for k in range(self.prob.maxevals):
            # evaluate fun/performance
            t0 = time.time()
            f_val = fun(x)
            self.time_fun_eval.append(time.time() - t0)
            x = self.update(f_val)

        self.X = self.X[:-1] # it is because in prob.update, it will calculate the next point to query (the last x2 is calculated but not assessed at max_evals +1)
        self.time_total = time.time() - t_all

        return self.xbest, self.fbest


    def results_display(self, N, z, fz, fbest, ibest):
        """
        Display intermediate results
        """
        z = z.reshape(self.prob.nci + self.prob.nd)
        print("N = %4d, cost = %7.4f, best = %7.4f, N_best =%4d " % (N, fz, fbest,ibest))

        string = ""
        for j in range(self.prob.nc):
            aux = z[j]
            if self.prob.scale_vars:
                aux = aux * self.prob.dd[j] + self.prob.d0[j]
            string = string + " x" + str(j + 1) + " = " + ('%7.4f' % aux) + "   "
        for j in range(self.prob.nint):
            aux = z[self.prob.nc+j]
            if self.prob.scale_vars and (not self.prob.int_encoded):
                aux = aux * self.prob.dd_int[j] + self.prob.d0_int[j]
            string = string + " x" + str(j + self.prob.nc + 1) + " = " + ('%5d' % round(aux)) + "   "
        for j in range(self.prob.nd):
            aux = z[self.prob.nci+j]
            string = string + " x" + str(j + self.prob.nci + 1) + " = " + ('%5d' % round(aux))

        print(string)
        return











