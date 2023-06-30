"""
Set up the problem

(C) 2021-2023 Mengjia Zhu, Alberto Bemporad
"""

import numpy as np
from scipy.optimize import linprog as linprog
import sys

from src.pwasopt.pref_fun1 import PWASp_fun1
from src.pwasopt.pref_fun import PWASp_fun


class problem_defn:
    """
    Initial set up of the problem (problem description/definition)

    """

    def __init__(self, isPref, fun, lb, ub, delta_E, nc, nint, nd, X_d, nsamp, maxevals, feasible_sampling,
                 isLin_eqConstrained, Aeq, beq, isLin_ineqConstrained, Aineq, bineq, K, scale_vars, shrink_range,
                 alpha, sigma, separation, maxiter, cost_tol, min_number,
                 fit_on_partition, softmax_solver, softmax_maxiter, beta, initialization,
                 verbose, categorical,timelimit, epsDeltaF, acq_stage, sepvalue, synthetic_dm):

        self.synthetic_dm = synthetic_dm
        self.isPref = isPref

        self.delta_E = delta_E
        self.nc = nc
        self.nint = nint
        self.nci = nc + nint
        self.nd = nd
        self.X_d = X_d
        self.sum_X_d = sum(X_d)
        self.nvars = self.nci + nd
        self.nvars_encoded = self.nci + self.sum_X_d

        if np.isinf(lb).any() or np.isinf(ub).any():
            errstr_inf = "Please specify non-infinity upper and lower bounds"
            print(errstr_inf)
            sys.exit(1)

        if lb.shape[0] != self.nvars or ub.shape[0] != self.nvars:
            errstr_bd_shape = "Please specify the lower and upper bounds for all types of variables. Check the Notes in 'main_pwas.py' or 'main_pwasp for how to specify the bounds for the categorical variables"
            print(errstr_bd_shape)
            sys.exit(1)

        lb_encoded = np.zeros(self.nvars_encoded)
        ub_encoded = np.ones(self.nvars_encoded)
        lb_encoded[:self.nci] = lb[:self.nci]
        ub_encoded[:self.nci] = ub[:self.nci]
        self.lb = lb_encoded
        self.ub = ub_encoded
        self.lb_nvars = lb.copy()
        self.ub_nvars = ub.copy()
        self.lb_original = lb.copy()
        self.ub_original = ub.copy()

        self.nsamp = nsamp
        self.maxevals = maxevals

        if self.maxevals < self.nsamp:
            errstr = "Max number of function evaluations is too low. You specified"
            errstr = errstr + " maxevals = " + str(maxevals) + " and nsamp = " + str(nsamp)
            print(errstr)
            sys.exit(1)

        self.feasible_sampling = feasible_sampling
        self.isLin_eqConstrained = isLin_eqConstrained
        self.Aeq = Aeq
        self.beq = beq
        self.isLin_ineqConstrained = isLin_ineqConstrained
        self.Aineq = Aineq
        self.bineq = bineq

        if K is None:
            errstr_k = "Please specify the number of initial partitions"
            print(errstr_k)
            sys.exit(1)
        else:
            self.K = K

        if nc ==0:
            scale_vars = False

        self.scale_vars = scale_vars

        dd = np.ones((self.nvars_encoded,)) # To be used for constraints scaling
        d0 = np.zeros((self.nvars_encoded,))
        dd[:nc] = (ub[:nc] - lb[:nc]) / 2
        d0[:nc] = (ub[:nc] + lb[:nc]) / 2
        # Note: in dd and d0, for integer variables, no scaling is applied (when apply constraints scaling, integer variables are not scaled)
        # instead, in the following, dd_int and d0_int are used, when int variables need to be scaled
        self.dd = dd
        self.d0 = d0

        self.dd_int = (ub[nc:self.nci] - lb[nc:self.nci]) / 2
        self.d0_int = (ub[nc:self.nci] + lb[nc:self.nci]) / 2

        self.alpha = alpha
        self.sigma = sigma

        if separation is None:
            self.separation = 'Softmax'
        else:
            self.separation = separation

        self.maxiter = maxiter
        self.cost_tol = cost_tol

        if min_number is None:
            self.min_number = 1
        else:
            self.min_number = min_number

        self.fit_on_partition = fit_on_partition
        self.softmax_solver = softmax_solver

        if softmax_maxiter is None:
            self.softmax_maxiter = 10000
        else:
            self.softmax_maxiter = softmax_maxiter

        if beta is None:
            self.beta = 0.02
        else:
            self.beta = beta

        if initialization is None:
            self.initialization = ("kmeans", 10)
        else:
            self.initialization = initialization

        if verbose is None:
            self.verbose = 1
        else:
            self.verbose = verbose

        if timelimit is None:
            self.timelimit = self.nvars_encoded * 3  # seconds
            # self.timelimit = 5
        else:
            self.timelimit = timelimit

        if epsDeltaF is None:
            self.epsDeltaF = 1e-4
        else:
            self.epsDeltaF = epsDeltaF

        self.categorical = categorical
        self.acq_stage = acq_stage

        if sepvalue is None:
            self.sepvalue = float(1/maxevals)
        else:
            self.sepvalue = float(sepvalue)

        if self.scale_vars:
            lb_encoded[:nc] = -np.ones(nc)
            ub_encoded[:nc] = np.ones(nc)
            self.lb = lb_encoded
            self.ub = ub_encoded
            self.lb_nvars[:nc] = -np.ones(nc)
            self.ub_nvars[:nc] = np.ones(nc)

            if isLin_eqConstrained:
                if Aeq.shape[1] != self.nvars_encoded:
                    errstr_eq = "The size of the linear equality constraint matrix is not consistent with the encoded optimization variables (please include an INDIVIDUAL constraint for each option for categorical/discrete/binary varialbes)"
                    print(errstr_eq)
                    sys.exit(1)
                self.Aeq = Aeq.dot(np.diag(dd.flatten('C')))
                self.beq = beq-Aeq.dot(d0.reshape(self.nvars_encoded,1))

            if isLin_ineqConstrained:
                if Aineq.shape[1] != self.nvars_encoded:
                    errstr_eq = "The size of the linear inequality constraint matrix is not consistent with the encoded optimization variables (please include an INDIVIDUAL constraint for each option for categorical/discrete/binary varialbes)"
                    print(errstr_eq)
                    sys.exit(1)
                self.Aineq = Aineq.dot(np.diag(dd.flatten('C')))
                self.bineq = bineq - Aineq.dot(d0.reshape(self.nvars_encoded,1))

        self.lb_unshrink = self.lb.copy()
        self.ub_unshrink = self.ub.copy()
        if (shrink_range) and (isLin_ineqConstrained):
            lb_shrink = self.lb.copy()
            ub_shrink = self.ub.copy()
            bounds_linprog = np.zeros((self.nvars_encoded,2))
            bounds_linprog[:,0] = self.lb.copy()
            bounds_linprog[:, 1] = self.ub.copy()
            flin = np.zeros((self.nvars_encoded,1))
            for i in range(self.nvars_encoded):
                flin[i] = 1
                res = linprog(flin,self.Aineq,self.bineq,bounds = bounds_linprog)
                aux = max(self.lb[i],res.fun)
                lb_shrink[i] = aux

                flin[i] = -1
                res = linprog(flin,self.Aineq,self.bineq,bounds =bounds_linprog)
                aux = min(self.ub[i],-res.fun)
                ub_shrink[i] = aux
                flin[i] = 0

            self.lb = lb_shrink
            self.ub = ub_shrink

            self.lb_nvars[:self.nci] = lb_shrink[:self.nci]
            self.ub_nvars[:self.nci] = ub_shrink[:self.nci]

        if nint >0:
            int_interval = np.round(self.ub_original[self.nc:self.nci] - self.lb_original[self.nc:self.nci] + 1)
            self.int_prod = np.prod(int_interval)
            int_sum = round(np.sum(int_interval))
            self.int_interval = int_interval
        else:
            self.int_prod = 0
            self.int_interval = []

        if nint >0 and self.int_prod < maxevals:
            self.int_encoded = True
            # if number of possible combinations of integer variables exceed the number of max evaluations, one-hot encode integer variables
            if isLin_eqConstrained:
                Aeq_int_encode = np.zeros((Aeq.shape[0],nc+int_sum+self.sum_X_d))
                Aeq_int_encode[:,:nc] = self.Aeq[:,:nc]
                Aeq_int_encode[:,nc+int_sum:] = self.Aeq[:,self.nci:]
                for i in range(nint):
                    Aeq_int_encode[:, nc + round(np.sum(int_interval[:i])):nc + round(np.sum(int_interval[:i + 1]))] = \
                        self.Aeq[:,nc+i].dot(np.arange(self.lb[nc+i],self.ub[nc+i]+1))

                self.Aeq = Aeq_int_encode

            if isLin_ineqConstrained:
                Aineq_int_encode = np.zeros((Aineq.shape[0], nc + int_sum + self.sum_X_d))
                Aineq_int_encode[:, :nc] = self.Aineq[:, :nc]
                Aineq_int_encode[:, nc + int_sum:] = self.Aineq[:, self.nci:]
                for i in range(nint):
                    Aineq_int_encode[:, nc + round(np.sum(int_interval[:i])):nc + round(np.sum(int_interval[:i + 1]))] = \
                        self.Aineq[:,nc+i].reshape((-1,1)).dot(np.arange(self.lb_original[nc + i], self.ub_original[nc + i] + 1).reshape((1,-1)))

                self.Aineq = Aineq_int_encode

            self.nint_encoded = int_sum
            self.nci_encoded = self.nc + self.nint_encoded
            self.nvars_encoded = self.nci_encoded + self.sum_X_d

            lb_int_encoded = np.zeros(self.nvars_encoded)
            ub_int_encoded = np.ones(self.nvars_encoded)
            lb_int_encoded[:self.nc] = self.lb[:self.nc]
            ub_int_encoded[:self.nc] = self.ub[:self.nc]
            self.lb = lb_int_encoded
            self.ub = ub_int_encoded
        else:
            self.int_encoded = False
            self.nint_encoded = nint
            self.nci_encoded = self.nci

        dd_nvars = np.ones((self.nvars,)) # Here, integer variables are also scaled, to be used for obj. fun. scaling
        d0_nvars = np.zeros((self.nvars,))
        if self.int_encoded:
            dd_nvars[:self.nc] = (ub[:self.nc] - lb[:self.nc]) / 2
            d0_nvars[:self.nc] = (ub[:self.nc] + lb[:self.nc]) / 2
        else:
            dd_nvars[:self.nci] = (ub[:self.nci] - lb[:self.nci]) / 2
            d0_nvars[:self.nci] = (ub[:self.nci] + lb[:self.nci]) / 2
        self.dd_nvars = dd_nvars
        self.d0_nvars = d0_nvars

        dd_nvars_encoded_ = np.ones((self.nvars_encoded,)) # Here, integer variables are also scaled
        d0_nvars_encoded_ = np.zeros((self.nvars_encoded,))
        if self.int_encoded:
            dd_nvars_encoded_[:self.nc] = dd_nvars[:self.nc]
            d0_nvars_encoded_[:self.nc] = d0_nvars[:self.nc]
        else:
            dd_nvars_encoded_[:self.nci] = dd_nvars[:self.nci]
            d0_nvars_encoded_[:self.nci] = d0_nvars[:self.nci]


        if isPref:
            if synthetic_dm:
                comparetol = 1e-4
                if isLin_ineqConstrained or isLin_eqConstrained:
                    pref_fun = PWASp_fun1(fun, comparetol, self.Aeq, self.beq, self.Aineq, self.bineq)  # preference function object
                else:
                    pref_fun = PWASp_fun(fun, comparetol)
                pref = lambda x, y, x_encoded, y_encoded: pref_fun.eval(x, y, x_encoded, y_encoded)
                self.pref = pref
                self.pref_fun = pref_fun
            else:
                self.pref = fun
            self.f = lambda x: 0
        else:
            self.f = fun
            self.pref = lambda x: 0

        # if self.scale_vars:
        #     if isPref:
        #         self.pref = lambda x, y, x_encoded, y_encoded: pref(x * self.dd_nvars + self.d0_nvars,
        #                                                            y * self.dd_nvars + self.d0_nvars,
        #                                                            x_encoded * dd_nvars_encoded_ + d0_nvars_encoded_,
        #                                                            y_encoded * dd_nvars_encoded_ + d0_nvars_encoded_)
        #     else:
        #         self.f = lambda x: fun(x * self.dd_nvars + self.d0_nvars)






