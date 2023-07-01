"""
Fit the surrogate and obtain the relevant coefficients
    - Coefficient for PWL separation: omega, gamma
    - Coefficient for PWA in each partition: a, b

(C) 2021-2023 Mengjia Zhu, Alberto Bemporad
"""
import numpy as np

from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from scipy.special import logsumexp
import pulp as plp

import sys


class fit_surrogate:

    def __init__(self, prob):
        """
        Obtain and constructs all necessary attributes for self
        """
        self.nvars_encoded = prob.nvars_encoded
        self.timelimit = prob.timelimit
        self.sepvalue = prob.sepvalue
        self.K = prob.K
        self.initialization = prob.initialization
        self.separation = prob.separation
        self.K = prob.K
        self.nx = prob.nvars_encoded
        self.sigma = prob.sigma
        self.cost_tol = prob.cost_tol
        self.maxiter = prob.maxiter
        self.beta = prob.beta
        self.softmax_solver = prob.softmax_solver
        self.softmax_maxiter = prob.softmax_maxiter
        self.min_number = prob.min_number
        self.fit_on_partition = prob.fit_on_partition


    def get_parameters(self, X, I, Ieq, N, K, z_pwl_N):
        """
        Given the training samples, the corresponding preference relations,
         and the cluster assignments, determine the PWA parameters (a,b)

        Inputs:
                X: np array, the training samples
                I: np array, the preference relationship, I[i,1:2]=[h k] if F(h)<F(k)-comparetol
                Ieq: np array, the preference relationship, Ieq[i,1:2]=[h k] if |F(h)-F(k)|<=comparetol
                N: int, number of training samples
                K: int, number of partitions
                z_pwl_N:  np array with int elements, the assignments of each sampling point to the cluster

        Outputs:
                a_opt: np array, optimum coefficient a of PWA
                b_opt: np array, optimum coefficient b of PWA
                y_pred_opt: np array, the predicted function evaluations for the training samples X with the optimum coefficients
        """

        nvars_encoded = self.nvars_encoded
        timelimit = self.timelimit
        sepvalue = self.sepvalue

        m = I.shape[0]
        meq = Ieq.shape[0]
        # m_total = m + 2 * meq  # total number of comparisons
        m_total = m + meq

        prob_param = plp.LpProblem('PWASp_param', plp.LpMinimize)  # problem to find the parameters

        y_pred = plp.LpVariable.dicts("y_pred", range(N), cat=plp.LpContinuous)
        a = plp.LpVariable.dicts('a', (range(K), range(nvars_encoded)), cat=plp.LpContinuous)
        b = plp.LpVariable.dicts('b', range(K), cat=plp.LpContinuous)
        eps = plp.LpVariable.dicts("eps", range(m_total), lowBound=0, cat=plp.LpContinuous)
        reg_para = plp.LpVariable.dicts("reg_para", range(1), lowBound=0, cat=plp.LpContinuous)

        # Objective function
        cost = plp.lpSum(eps) + 1.0e-6*plp.lpSum(reg_para)
        prob_param += cost

        # Exclusive-or constraint
        for n in range(N):
            # prob_param += (plp.lpSum(z_pwl_N[n][i] for i in range(K)) == 1.0)

            for j in range(K):
                if z_pwl_N[n] == j:
                    ax = plp.lpSum(a[j][h] * X[n][h] for h in range(nvars_encoded))
                    prob_param += (y_pred[n] == ax + b[j])


        # Objective function: Minimize sum|eps|
        for k in range(0, m):
            i = I[k][0]
            j = I[k][1]
            prob_param += y_pred[i] + sepvalue <= y_pred[j] + eps[k]

        for k in range(0, meq):
            i = Ieq[k][0]
            j = Ieq[k][1]
            prob_param += y_pred[i] - y_pred[j] <= sepvalue + eps[m + k]
            prob_param += y_pred[j] - y_pred[i] <= sepvalue + eps[m + k]

        # infinite norm regularization of coefficients
        for j in range(K):
            prob_param += reg_para >= -b[j]
            prob_param += reg_para >= b[j]
            for i in range(nvars_encoded):
                prob_param += reg_para >= -a[j][i]
                prob_param += reg_para >= a[j][i]

        try:
            prob_param.solve(plp.GUROBI(timeLimit=timelimit, msg=0))
        except:
            prob_param.solve(plp.GLPK(timeLimit=timelimit, msg=0))

        status = prob_param.status

        if status == plp.LpStatusOptimal:  # See plp.constants.LpStatus. See more in constants.py in pulp/ folder
            a_opt = np.zeros((K, nvars_encoded, 1))
            b_opt = np.zeros((K, 1))
            y_pred_opt = np.zeros((N, 1))

            for k in range(K):
                b_opt[k] = b[k].varValue
                for var in range(nvars_encoded):
                    a_opt[k, var, 0] = a[k][var].varValue
            for n in range(N):
                y_pred_opt[n, 0] = y_pred[n].varValue

        return a_opt, b_opt, y_pred_opt


    def get_init_delta(self,X, N):
        """
        Get initial clustering of the training samples
            Reference code: 'parc_init()' function in PARC package by A. Bemporad, 2021
            use 'kmeans' or 'random' methods

        Inputs:
                X: np array, the training samples
                N: int, number of training samples

        Outputs:
                delta: np array with int elements, the initial assignments of each training samples to the clusters
        """

        K = self.K


        if K == 1:
            return np.zeros(N, dtype=int)

        raiseerror = False
        if not isinstance(self.initialization, tuple):
            raiseerror = True
        if (not raiseerror):
            if len(self.initialization) != 2:
                raiseerror = True
        if raiseerror:
            raise Exception('initialization option must be a tuple (string, value).')

        init_type = self.initialization[0]

        if init_type == "kmeans":
            # Run K-means++ to initialize delta
            kmeans = KMeans(n_clusters=K, init='k-means++',
                            n_init=self.initialization[1]).fit(X)
            delta = kmeans.labels_

        elif init_type == 'random':
            delta = np.random.random_integers(0, K - 1, N)  # random assignment
        else:
            raise Exception('Unknown initialization option "%s".' % init_type)

        return delta


    def get_pwl_param(self,X, delta, N):
        """
        Get optimum number of partitions, the cluster assignments, as well as the optimum coefficients for the PWL separation function
            reference code: PARC package by A. Bemporad, 2021

        Inputs:
                X: np array, the training samples
                delta: np array with int elements, the initial assignments of each training samples to the clusters
                N: int, number of training samples

        Outputs:
                Kf: int, optimum number of partitions
                delta: np array with int elements, the optimum assignments of each training samples to the clusters
                omega: np array, optimum coefficient a of the PWL separation function
                delta: np array, optimum coefficient a of the PWL separation function

        """

        separation = self.separation
        K = self.K
        nx = self.nvars_encoded
        sigma = self.sigma / N
        cost_tol = float(self.cost_tol)
        maxiter = int(self.maxiter)

        isSoftmax = (separation == 'Softmax')

        if not isSoftmax:
            errstr_softmax = 'Only \'Softmax\' separation method is implemented in this version. Please sepcify \'separation\' == \'Softmax\' '
            print(errstr_softmax)
            sys.exit(1)

        PWLsoftmax = LogisticRegression(multi_class='multinomial', C=2.0 / self.beta,
                                            solver=self.softmax_solver,
                                            max_iter=self.softmax_maxiter, tol=1e-6,
                                            warm_start=True)
        omega = np.zeros((K, nx))
        gamma = np.zeros(K)

        Nk = np.zeros(K, dtype=int)  # number of points in cluster
        killed = np.zeros(K, dtype=bool)  # clusters that have disappeared

        go = True
        iters = 0
        cost_old = np.inf
        cost_sequence = []

        while go:
            iters += 1
            ##########
            # Solve K ridge regression problems for the numeric targets
            ##########
            for j in range(K):
                if not killed[j]:
                    # Check if some clusters are empty
                    ii = (delta == j)
                    Nk[j] = np.sum(ii)
                    if Nk[j] == 0:
                        killed[j] = True

            ##########
            # find PWL separation function by softmax regression
            ##########
            if K > 1:
                omega1, gamma1 = self.fit_PWL(PWLsoftmax, X, delta)  # dim(gamma1) = unique(delta)
                h = 0
                for i in range(nx):
                    omega[~killed, i] = omega1[:, h]
                    h += 1
                gamma[~killed] = gamma1

            ##########
            # Compute cost and assign labels
            ##########
            cost = 0.0
            for k in range(N):
                cost_k = np.zeros(K)
                x_k = X[k, :].ravel()
                lin_terms = np.zeros(K)
                for j in range(K):
                    if not killed[j]:
                        aux = np.sum(omega[j, ] * x_k) + gamma[j]
                        cost_k[j] -= sigma * aux
                        lin_terms[j] = aux
                    else:
                        cost_k[j] = np.inf  # this prevents reassignement to killed cluster

                cost_k += sigma * logsumexp(lin_terms)

                # reassign labels
                delta[k] = np.argmin(cost_k)

                # compute current cost
                cost += cost_k[delta[k]]

            cost_sequence.append(cost)

            if (cost_old - cost <= cost_tol) or (iters == maxiter) or (K == 1):
                go = False

            cost_old = cost

        # update Nk and compute final number of clusters, possibly eliminating
        # empty or very small clusters, then clean-up solution
        for j in range(K):
            Nk[j] = np.sum(delta == j)
        killed = (Nk < self.min_number)

        isoutside = np.zeros(N, dtype=bool)
        anyoutside = False
        for i in range(K):
            if killed[i]:
                anyoutside = True
                isoutside[delta == i] = True

        NC = K - np.sum(killed)  # final number of clusters
        delta[isoutside] = -1  # mark points in small clusters with delta = -1
        Nk = np.zeros(NC, dtype=int)

        # recompute PWL partition based on final assignment
        for j in range(NC):
            ii = (delta == j)  # outliers are automatically excluded as j=0,...,NC-1
            Nk[j] = np.sum(ii)

        if NC > 1:
            if K > 1:
                omega = np.zeros((NC, nx))
                omega[:, ], gamma = self.fit_PWL(PWLsoftmax, X[~isoutside].T[:].T,
                                                 delta[~isoutside])  # dim(gamma1) = unique(delta[~isoutside])
        else:
            omega = np.zeros((1, nx))
            gamma = np.zeros(1)

        if not self.fit_on_partition:
            # re-label existing delta in (0,NC)
            elems = list(set(delta))
            if anyoutside:
                elems.remove(-1)
            for i in range(NC):
                delta[delta == elems[i]] = i
        else:
            # reassign points based on the polyhedron they belong to
            delta = np.argmax(X[:, ] @ omega[:, ].T + gamma, axis=1)
            delta[isoutside] = -1

            # some clusters may have disappeared after reassignment
            h = 0
            keep = np.ones(NC, dtype=bool)
            for j in range(NC):
                ii = (delta == j)
                aux = np.sum(ii)
                if aux >= self.min_number:
                    Nk[j] = aux
                    delta[ii] = h  # relabel points
                    h += 1
                else:
                    delta[ii] = -1  # also mark these as outliers, as they form an excessively small cluster
                    keep[j] = False


            if NC > h:
                omega = omega[keep, :]
                gamma = gamma[keep]
                Nk = Nk[keep]
            NC = h

            Kf = int(NC)

        return Kf, delta, omega, gamma


    def fit_PWL(self, softmax_reg, X, delta):
        """
        Fit a PWL separation function to the clusters obtained
            'fit_PWL' function taken from PARC package by A. Bemporad, 2021

        Inputs:
                softmax_reg: initial setup for the softmax regressor
                X: np array, the training samples
                delta: np array, the cluster assignments for each training sample

        Outputs:
                omega1: np array, coefficients of PWL separation function
                gamma1: np array, coefficients of PWL separation function

        """
        try:
            softmax_reg.fit(X, delta)
        except:
            if not len(np.unique(delta)) == 1:
                softmax_reg.warm_start = False  # new problem has different number of classes, disable warm start
                softmax_reg.fit(X, delta)
                softmax_reg.warm_start = True
        if len(np.unique(delta)) > 2:
            omega1 = softmax_reg.coef_
            gamma1 = softmax_reg.intercept_
        elif len(np.unique(delta)) == 1:
            omega1 = np.zeros((1, X.shape[1]))
            gamma1 = np.zeros(1)
        else:
            omega1 = np.zeros((2, X.shape[1]))
            gamma1 = np.zeros(2)
            omega1[0, :] = -softmax_reg.coef_
            gamma1[0] = -softmax_reg.intercept_
            omega1[1, :] = softmax_reg.coef_
            gamma1[1] = softmax_reg.intercept_

        return omega1, gamma1