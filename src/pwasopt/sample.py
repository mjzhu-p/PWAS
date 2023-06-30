"""
Generate initial samples for the surrogate fitting

(C) 2021-2023 Mengjia Zhu, Alberto Bemporad
"""

import math as mt
from pyDOE import lhs #https://pythonhosted.org/pyDOE/
import pulp as plp
import cdd  #https://pypi.org/project/pycddlib/
import random

from pwasopt.categorical_encoder import *
from pwasopt.integ_encoder import *

class init_sampl:

    def __init__(self, prob):
        """
        Obtain and constructs all necessary attributes for self
        """
        self.isPref = prob.isPref
        self.nsamp = prob.nsamp
        self.maxevals = prob.maxevals
        self.feasible_sampling = prob.feasible_sampling
        self.isLin_eqConstrained = prob.isLin_eqConstrained
        self.Aeq = prob.Aeq
        self.beq = prob.beq
        self.isLin_ineqConstrained = prob.isLin_ineqConstrained
        self.Aineq = prob.Aineq
        self.bineq = prob.bineq

        self.nc = prob.nc
        self.nint = prob.nint
        self.nci = prob.nci
        self.nd = prob.nd
        self.nvars = prob.nvars
        self.nvars_encoded = prob.nvars_encoded
        self.X_d = prob.X_d
        self.sum_X_d = prob.sum_X_d

        self.scale_vars = prob.scale_vars
        self.dd_int = prob.dd_int
        self.d0_int = prob.d0_int

        self.dd_nvars = prob.dd_nvars
        self.d0_nvars = prob.d0_nvars

        self.f = prob.f
        self.lb = prob.lb
        self.ub = prob.ub
        self.lb_nvars = prob.lb_nvars
        self.ub_nvars = prob.ub_nvars
        self.lb_unshrink = prob.lb_unshrink
        self.ub_unshrink = prob.ub_unshrink

        EC_cat = cat_encoder(prob)
        self.encode_cat = EC_cat.encode
        self.decode_cat = EC_cat.decode
        if self.nd > 0:
            self.encoder_cat = EC_cat.cat_encoder()

        self.int_encoded = prob.int_encoded
        self.nint_encoded = prob.nint_encoded
        self.nci_encoded = prob.nci_encoded
        self.int_interval = prob.int_interval
        EC_int = integ_encoder(prob)
        self.encode_int = EC_int.encode
        self.decode_int = EC_int.decode
        if self.int_encoded:
            self.encoder_int = EC_int.integ_encoder()

        self.timelimit = prob.timelimit


    def initial_sampling(self):
        """
        Initial sampling stage

        Outputs:
                Xs: the initial samples obtained, encoded if nd >0
                F: the function evaluation of the samples
                X_samp_decoded: the decoded version of the initial samples
        """

        nsamp = self.nsamp
        maxevals = self.maxevals
        feasible_sampling = self.feasible_sampling
        isLin_eqConstrained = self.isLin_eqConstrained
        Aeq = self.Aeq
        beq = self.beq
        isLin_ineqConstrained = self.isLin_ineqConstrained
        Aineq = self.Aineq
        bineq = self.bineq

        nc = self.nc
        nint = self.nint
        nci = self.nci
        # nci_encoded = self.nci_encoded
        nd = self.nd
        nvars = self.nvars
        nvars_encoded = self.nvars_encoded
        # sum_X_d = self.sum_X_d

        scale_vars = self.scale_vars
        dd_int = self.dd_int
        d0_int = self.d0_int

        f = self.f
        Xs = np.zeros((nsamp, nvars_encoded))
        F = np.zeros((nsamp, 1))

        use_solver = False
        if (not feasible_sampling) or ((not isLin_eqConstrained) and (not isLin_ineqConstrained)):
            # use LHS as initial sampling method
            Xs[0:nsamp, :nvars] = lhs(nvars, nsamp, "m")
            Xs[0:nsamp, :nvars] = Xs[0:nsamp, :nvars] * \
                                 (np.ones((nsamp, 1)) * (self.ub_nvars[:nvars] - self.lb_nvars[:nvars])) + np.ones(
                (nsamp, 1)) * self.lb_nvars[:nvars]

            # for integer variable, round it to the closest integer, then scale it (if scale_vars = True)
            if nint > 0:
                Xs[0:nsamp, nc:nci] = np.round(Xs[0:nsamp, nc:nci])
                if (not self.int_encoded) and (scale_vars):
                    Xs[0:nsamp, nc:nci] = (Xs[0:nsamp, nc:nci] - (np.ones((nsamp, 1)) * d0_int)) * (
                            np.ones((nsamp, 1)) * 1 / dd_int)

            # for binary/categorical variables, round it to the closest integer
            # for binary/categorical variables, generate samples first, then encode (one-hot encoding)
            if nd > 0:
                Xs[0:nsamp, nci:nvars] = np.round(Xs[0:nsamp, nci:nvars])
        else:
            if nint == 0 and nd ==0:
                if isLin_eqConstrained and isLin_ineqConstrained:
                    A_mat = np.vstack((-Aeq, -Aineq, np.identity(nvars_encoded), -np.identity(nvars_encoded)))
                    b_mat = np.vstack((beq, bineq, -self.lb_unshrink.reshape((nvars_encoded, 1)), self.ub_unshrink.reshape((nvars_encoded, 1))))
                elif isLin_eqConstrained:
                    A_mat = np.vstack((-Aeq, np.identity(nvars_encoded), -np.identity(nvars_encoded)))
                    b_mat = np.vstack((beq, -self.lb_unshrink.reshape((nvars_encoded, 1)), self.ub_unshrink.reshape((nvars_encoded, 1))))
                elif isLin_ineqConstrained:
                    A_mat = np.vstack((-Aineq, np.identity(nvars_encoded), -np.identity(nvars_encoded)))
                    b_mat = np.vstack((bineq, -self.lb_unshrink.reshape((nvars_encoded, 1)), self.ub_unshrink.reshape((nvars_encoded, 1))))
                else:
                    A_mat = np.vstack(( np.identity(nvars_encoded), -np.identity(nvars_encoded)))
                    b_mat = np.vstack((-self.lb_unshrink.reshape((nvars_encoded, 1)), self.ub_unshrink.reshape((nvars_encoded, 1))))

                mat_comb = np.hstack((b_mat, A_mat))
                mat = cdd.Matrix(mat_comb)
                mat.rep_type = cdd.RepType.INEQUALITY
                if isLin_eqConstrained:
                    mat.lin_set =[x for x in range(Aeq.shape[0])]
                poly = cdd.Polyhedron(mat)
                gen = poly.get_generators()
                numVert = len(gen)

                list_vert = np.asarray(gen[:])[:, 1:]
                if numVert > nsamp:
                    np.random.shuffle(list_vert)
                    Xs[0:nsamp, :nvars] =list_vert[:nsamp,:]
                else:
                    Xs[0:numVert, :nvars] = list_vert.copy()

                    if not isLin_eqConstrained:
                        coeff_vert = np.random.dirichlet(np.ones(numVert),size = nsamp-numVert)
                        Xs[numVert:nsamp,:nvars] = coeff_vert.dot(list_vert)
                    else:
                        XX_encoded = Xs[:numVert,:].copy()
                        fes_init = 1
                        use_solver = True
                        for ind in range(nsamp-numVert):
                            if self.int_encoded:
                                X_next = self.feasible_sampling_eq_ineq_constrained_intEncoded(XX_encoded, fes_init)
                            else:
                                X_next = self.feasible_sampling_eq_ineq_constrained(XX_encoded, fes_init)
                            Xs[ind+numVert, :] = X_next.copy()

                            XX_encoded = Xs[0:ind + numVert + 1, :]
                            fes_init = 1

            else:
                nn = nsamp
                nk = 0
                n_iter = 0
                n_iter_2 = 0
                while (nk < nsamp):
                    if nn*nvars < 2000 or nk >0:
                        XX = lhs(nvars, nn, "m")
                        XX = XX * (np.ones((nn, 1)) * (self.ub_nvars[:nvars] - self.lb_nvars[:nvars])) + np.ones(
                            (nn, 1)) * self.lb_nvars[:nvars]

                        # for integer variable, round it to the closest integer, then scale it (if scale_vars = True)
                        if nint > 0:
                            XX[:, nc:nci] = np.round(XX[:, nc:nci])
                            if (not self.int_encoded) and (scale_vars):
                                XX[:, nc:nci] = (XX[:, nc:nci] - (np.ones((nn, 1)) * d0_int)) * (np.ones((nn, 1)) * 1 / dd_int)

                        # for binary/categorical variables, rounded it to closest integer
                        # for binary/categorical variables, generate samples first, then encode (one-hot encoding)
                        if nd > 0:
                            XX[:, nci:] = np.round(XX[:, nci:])

                        XX_decoded = XX[:,:nvars].copy()
                        if self.int_encoded and nd > 0:
                            XX_encoded_int = self.encode_int(XX_decoded, self.encoder_int)
                            XX_encoded = self.encode_cat(XX_encoded_int, self.encoder_cat)
                        elif self.int_encoded:
                            XX_encoded = self.encode_int(XX_decoded, self.encoder_int)
                        elif nd > 0:
                            XX_encoded = self.encode_cat(XX_decoded, self.encoder_cat)
                        else:
                            XX_encoded = XX.copy()

                        ii = np.ones((nn, 1)).flatten("C")
                        XX_encoded_fes_check = XX_encoded.copy()
                        if (scale_vars) and (nint > 0) and (not self.int_encoded):
                            XX_encoded_fes_check[:, nc:nci] = XX_encoded_fes_check[:, nc:nci]* (np.ones((nn, 1)) * dd_int) + np.ones((nn, 1)) * d0_int

                        for i in range(nn):
                            if isLin_eqConstrained:
                                ii[i] = all(Aeq.dot(XX_encoded_fes_check[i,].T) <= beq.flatten("c") + 1.0e-8)
                                ii[i] = ii[i] and all(-Aeq.dot(XX_encoded_fes_check[i,].T) <= -beq.flatten("c") + 1.0e-8)
                            if isLin_ineqConstrained:
                                ii[i] = ii[i] and all(Aineq.dot(XX_encoded_fes_check[i,].T) <= bineq.flatten("c") + 1.0e-8)

                        nk = sum(ii)
                    if (n_iter < 4) and (n_iter_2) < 2 and (nk == 0):
                        nn = 20 * nn
                        n_iter_2 += 1
                        if ((isLin_eqConstrained) and (nn > 200)) or ((isLin_ineqConstrained) and (nn > 400)) or (20 * nn >500):
                            n_iter_2 += 1
                        n_iter += 1
                    elif (nk < nsamp) and (nk >0):
                        nn = mt.ceil(min(20, 1.1 * nsamp / nk) * nn)
                        n_iter += 1
                    elif (nk < nsamp) :
                        # use a optimization solver to find initial samples
                        # (when it's hard to find enough # of initial samples that satisfies constraints)
                        use_solver = True

                        if nk >0:
                            ii = np.asarray(np.where(ii))
                            n_fes = ii.size
                            XX = XX[ii[0][0:n_fes],].reshape(n_fes, nvars)
                            fes_init = 1
                        else:
                            n_fes = 0
                            ind_rand = random.randrange(XX.shape[0])
                            XX = XX[ind_rand, :].reshape(1, nvars)
                            fes_init = ii[ind_rand]

                        if self.int_encoded and nd > 0:
                            XX_encoded_int = self.encode_int(XX, self.encoder_int)
                            XX_encoded = self.encode_cat(XX_encoded_int, self.encoder_cat)
                        elif self.int_encoded:
                            XX_encoded = self.encode_int(XX, self.encoder_int)
                        elif nd > 0:
                            XX_encoded = self.encode_cat(XX, self.encoder_cat)
                        else:
                            XX_encoded = XX

                        Xs[0:n_fes,:] = XX_encoded
                        for ind in range(nsamp-n_fes):
                            if self.int_encoded:
                                X_next = self.feasible_sampling_eq_ineq_constrained_intEncoded(XX_encoded, fes_init)
                            else:
                                X_next = self.feasible_sampling_eq_ineq_constrained(XX_encoded, fes_init)
                            Xs[ind+n_fes, :] = X_next.copy()

                            XX_encoded = Xs[0:ind + n_fes + 1, :]
                            fes_init = 1
                        nk = nsamp

        if not use_solver:
            if nint == 0 and nd == 0:
                X_sampl_decoded = Xs[0:nsamp, :]
            elif feasible_sampling and (isLin_ineqConstrained or isLin_eqConstrained):
                ii = np.where(ii)
                Xs[0:nsamp, :nvars] = XX[ii[0][0:nsamp],]
                X_sampl_decoded = Xs[0:nsamp, :nvars].copy()
                if self.int_encoded or nd > 0:
                    X_sampl_encoded = XX_encoded[ii[0][0:nsamp],]
                    Xs[0:nsamp, :] = X_sampl_encoded
            else:
                X_sampl_decoded = Xs[0:nsamp, :nvars].copy()
                if self.int_encoded and nd > 0:
                    X_sampl_encoded_int = self.encode_int(X_sampl_decoded, self.encoder_int)
                    X_sampl_encoded = self.encode_cat(X_sampl_encoded_int, self.encoder_cat)
                elif self.int_encoded:
                    X_sampl_encoded = self.encode_int(X_sampl_decoded, self.encoder_int)
                elif nd > 0:
                    X_sampl_encoded = self.encode_cat(X_sampl_decoded, self.encoder_cat)
                if self.int_encoded or nd > 0:
                    Xs[0:nsamp, :] = X_sampl_encoded
        else:
            X_sampl_decoded = Xs[0:nsamp, :].copy()
            if self.int_encoded and nd > 0:
                X_sampl_decoded_int = self.decode_int(X_sampl_decoded, self.encoder_int)
                X_sampl_decoded = self.decode_cat(X_sampl_decoded_int, self.encoder_cat)
            elif self.int_encoded:
                X_sampl_decoded = self.decode_int(X_sampl_decoded, self.encoder_int)
            elif nd > 0:
                X_sampl_decoded = self.decode_cat(X_sampl_decoded, self.encoder_cat)

        if scale_vars:
            X = list(X_sampl_decoded * (np.ones((nsamp, 1)) * self.dd_nvars) + np.ones((nsamp, 1)) * self.d0_nvars)
        else:
            X = list(X_sampl_decoded)

        # if not self.isPref:
        #     for i in range(nsamp):
        #         F[i] = f(X_sampl_decoded[i, :].T)
        # else:
        #     F = []

        # return Xs, F, X_sampl_decoded
        return Xs, X


    def feasible_sampling_eq_ineq_constrained(self, Xs, fes_init):
        """
        Use MILP to obtain feasible initial samples when integer variables, if exist, are not one-hot encoded
            - Used when the constraints are hard to fulfill by solely randomly sampling
            - Optimization is used to maximize the distance between initial samples

        Inputs:
                Xs: np array, initial samples already obtained
                fes_init: bool,
                    if True, Xs are all feasible
                    if False, Xs are not feasible
                    (necessary for the initial guess of Xs because it was randomly generated)
        Outputs:
                xopt: np array, a feasible initial sample generated

        """

        lb = self.lb
        ub = self.ub
        nc = self.nc
        nint = self.nint
        nci = self.nci
        nd = self.nd
        X_d = self.X_d
        sum_X_d = self.sum_X_d

        scale_vars = self.scale_vars
        dd_int = self.dd_int
        d0_int = self.d0_int

        X_curr_c = Xs[:, :nci]
        X_curr_d = Xs[:, nci:]

        N = np.shape(Xs)[0]

        isLin_eqConstrained = self.isLin_eqConstrained
        Aeq = self.Aeq
        beq = self.beq
        isLin_ineqConstrained = self.isLin_ineqConstrained
        Aineq = self.Aineq
        bineq = self.bineq

        timelimit = self.timelimit

        prob = plp.LpProblem('feas_sampling', plp.LpMinimize)
        xc = plp.LpVariable.dicts("xc", (range(nci), range(1)), cat=plp.LpContinuous)
        xd = plp.LpVariable.dicts("xd", (range(sum_X_d), range(1)), cat=plp.LpBinary)
        xint = plp.LpVariable.dicts("xint", (range(nint), range(1)), cat=plp.LpInteger)
        if (N > 1 or fes_init):
            if nc >0:
                delta_p = plp.LpVariable.dicts("delta_p", (range(N), range(nci)), cat=plp.LpBinary)
                delta_m = plp.LpVariable.dicts("delta_m", (range(N), range(nci)), cat=plp.LpBinary)
            if nint >0:
                delta_p_int = plp.LpVariable.dicts("delta_p_int", (range(N), range(nint)), cat=plp.LpBinary)
                delta_m_int = plp.LpVariable.dicts("delta_m_int", (range(N), range(nint)), cat=plp.LpBinary)
        beta = plp.LpVariable.dicts("beta", range(1), lowBound=0, cat=plp.LpContinuous)
        beta_int = plp.LpVariable.dicts("beta_int",  range(1), lowBound=0, cat=plp.LpContinuous)
        ham_dist_scaled = plp.LpVariable.dicts("ham_dist_scaled", range(1), lowBound=0, cat=plp.LpContinuous)

        # Objective function
        if (N > 1 or fes_init):
            cost = - plp.lpSum(ham_dist_scaled) - plp.lpSum(beta) - plp.lpSum(beta_int) # max the distance between points
        else:
            cost = - plp.lpSum(xc[h][0]-X_curr_c[0][h] for h in range(nci)) - plp.lpSum(xd[h][0]-X_curr_d[0][h] for h in range(sum_X_d))
        prob += cost

        # up and lower bound of xc
        for h in range(nc):
            prob += xc[h][0] <= ub[h]
            prob += xc[h][0] >= lb[h]

        # up and lower bound of xint
        for h in range(nint):
            prob += xint[h][0] <= ub[nc + h]
            prob += xint[h][0] >= lb[nc + h]

        # relationship between xint and xint_scaled
        for h in range(nint):
            if scale_vars:
                prob += xc[nc+h][0]*dd_int[h] + d0_int[h] == xint[h][0]
            else:
                prob += xc[nc + h][0] == xint[h][0]

        # constraints for one-hot encoded xd
        for i in range(nd):
            prob += (plp.lpSum(xd[h][0] for h in range(sum(X_d[:i]), sum(X_d[:i + 1]))) == 1)

        # constraints satisfaction
        if isLin_eqConstrained:
            for i in range(np.size(beq)):
                prob += (plp.lpSum(Aeq[i, h] * xc[h][0] for h in range(nc))
                         + plp.lpSum(Aeq[i, nc + h] * xint[h][0] for h in range(nint))
                         + plp.lpSum(Aeq[i, nci + h] * xd[h][0] for h in range(sum_X_d))
                         == beq[i][0])

        if isLin_ineqConstrained:
            for i in range(np.size(bineq)):
                prob += (plp.lpSum(Aineq[i, h] * xc[h][0] for h in range(nc))
                         + plp.lpSum(Aineq[i, nc + h] * xint[h][0] for h in range(nint))
                         + plp.lpSum(Aineq[i, nci+h] * xd[h][0] for h in range(sum_X_d))
                         <= bineq[i][0])

        # acquisition function related
        if nd > 0 and (N > 1 or fes_init):
            # discrete variables
            sum_ham_dist = 0
            for i in range(N):
                ind_0 = list(np.where(X_curr_d[i, :] == 0)[0])
                ind_1 = list(np.where(X_curr_d[i, :] == 1)[0])
                sum_ham_dist += plp.lpSum(xd[h][0] for h in ind_0) + plp.lpSum(1 - xd[h][0] for h in ind_1)

            prob += ham_dist_scaled == 1 / (nd * N) * sum_ham_dist
        else:
            prob += ham_dist_scaled[0] == 0

        # continuous variables
        # big-M for exploration function
        M_x = 1.0e5 * np.ones((1, nci))
        # if N >1:
        #     max_X_curr = np.zeros((1, nci))
        #     min_X_curr = np.zeros((1, nci))
        #     for h in range(nci):
        #         max_X_curr[0][h] = np.amax(X_curr_c[:, h])
        #         min_X_curr[0][h] = np.amin(X_curr_c[:, h])
        #     for i in range(nci):
        #         if max_X_curr[0,i] > 0:
        #             M_x[0,i] = 2 * max_X_curr[0,i] - min_X_curr[0,i]
        #         else:
        #             M_x[0,i] = max_X_curr[0,i] - 2 * min_X_curr[0,i]

        if nint >0:
            sum_delta_int_N = 0
        if (N > 1 or fes_init):
            for i in range(N):
                if nc >0:
                    for j in range(nc):
                        prob += delta_p[i][j] == 1 - delta_m[i][j]
                        prob += xc[j][0] - X_curr_c[i, j] >= beta[0] - M_x[0, j] * (1 - delta_p[i][j])
                        prob += -xc[j][0] + X_curr_c[i, j] >= beta[0] - M_x[0, j] * (1 - delta_m[i][j])
                    # prob += (plp.lpSum(delta_p[i][j] + delta_m[i][j] for j in range(nci)) >= 1)
                if nint>0:
                    for j in range(nint):
                        prob += delta_p_int[i][j] == 1 - delta_m_int[i][j]
                        prob += (xc[nc + j][0] - X_curr_c[i, nc + j]) * dd_int[j] >= beta_int[0]- M_x[0, nc + j] * (1 - delta_p_int[i][j])
                        prob += (xc[nc + j][0] - X_curr_c[i, nc + j]) * dd_int[j] <= -beta_int[0] + M_x[0, nc + j] * (1 - delta_m_int[i][j])
        else:
            prob += beta[0] == 0
            prob += beta_int[0] == 0

        if nc <1:
            prob += beta[0] == 0
        if nint <1:
            prob += beta_int[0] == 0



        try:
            prob.solve(plp.GUROBI(timeLimit=timelimit, msg=0))
        except:
            prob.solve(plp.GLPK(timeLimit=timelimit, msg=0))
        # prob.solve()
        status = prob.status

        if status == plp.LpStatusOptimal:  # See plp.constants.LpStatus. See more in constants.py in pulp/ folder
            xopt = np.zeros((nci + sum_X_d,))
            for i in range(nci):
                xopt[i,] = xc[i][0].varValue
            for i in range(nci, nci + sum_X_d):
                xopt[i,] = xd[i - nci][0].varValue
        # elif prob.solverModel.status == 9:
        #     print("Acquisition is not solved to optimal, the current best solution is used")
        #     xopt = np.array(prob.solverModel.Xs[:nci+sum_X_d])

            # # # to assess the results:
            # delta_p_opt = np.zeros((N,nci))
            # delta_m_opt = np.zeros((N, nci))
            # for i in range(N):
            #     for j in range(nci):
            #         delta_p_opt[i,j] = delta_p[i][j].varValue
            #         delta_m_opt[i, j] = delta_m[i][j].varValue
            # ham_dist_scaled = ham_dist_scaled[0].varValue
            # beta_opt = beta[0].varValue
        else:
            xopt = None

        return xopt


    def feasible_sampling_eq_ineq_constrained_intEncoded(self, Xs, fes_init):
        """
        Use MILP to obtain feasible initial samples when integer variables, if exist, are one-hot encoded
            - Used when the constraints are hard to fulfill by solely randomly sampling
            - Optimization is used to maximize the distance between initial samples

        Inputs:
                Xs: np array, initial samples already obtained
                fes_init: bool,
                    if True, Xs are all feasible
                    if False, Xs are not feasible
                    (necessary for the initial guess of Xs because it was randomly generated)
        Outputs:
                xopt: np array, a feasible initial sample generated

        """

        lb = self.lb
        ub = self.ub
        nc = self.nc
        nint = self.nint
        # nci = self.nci
        nci_encoded = self.nci_encoded
        nint_encoded = self.nint_encoded
        int_interval = self.int_interval
        nd = self.nd
        X_d = self.X_d
        sum_X_d = self.sum_X_d

        X_curr_c = Xs[:, :nc]
        X_curr_int = Xs[:, nc:nci_encoded]
        X_curr_d = Xs[:, nci_encoded:]

        N = np.shape(Xs)[0]

        isLin_eqConstrained = self.isLin_eqConstrained
        Aeq = self.Aeq
        beq = self.beq
        isLin_ineqConstrained = self.isLin_ineqConstrained
        Aineq = self.Aineq
        bineq = self.bineq

        timelimit = self.timelimit

        prob = plp.LpProblem('feas_sampling', plp.LpMinimize)
        xc = plp.LpVariable.dicts("xc", (range(nc), range(1)), cat=plp.LpContinuous)
        xint = plp.LpVariable.dicts("xint", (range(nint_encoded), range(1)), cat=plp.LpBinary)
        xd = plp.LpVariable.dicts("xd", (range(sum_X_d), range(1)), cat=plp.LpBinary)
        if (N > 1 or fes_init):
            if nc >0:
                delta_p = plp.LpVariable.dicts("delta_p", (range(N), range(nc)), cat=plp.LpBinary)
                delta_m = plp.LpVariable.dicts("delta_m", (range(N), range(nc)), cat=plp.LpBinary)
        beta = plp.LpVariable.dicts("beta", range(1), lowBound=0, cat=plp.LpContinuous)
        int_ham_dist_scaled = plp.LpVariable.dicts("int_ham_dist_scaled", range(1), lowBound=0, cat=plp.LpContinuous)
        cat_ham_dist_scaled = plp.LpVariable.dicts("cat_ham_dist_scaled", range(1), lowBound=0, cat=plp.LpContinuous)

        # Objective function
        if (N > 1 or fes_init):
            cost = - plp.lpSum(cat_ham_dist_scaled) - plp.lpSum(beta) - plp.lpSum(int_ham_dist_scaled) # max the distance between points
        else:
            cost = - plp.lpSum(xc[h][0]-X_curr_c[0][h] for h in range(nc)) \
                   - plp.lpSum(xint[h][0]-X_curr_int[0][h] for h in range(nint_encoded)) \
                   - plp.lpSum(xd[h][0] - X_curr_d[0][h] for h in range(sum_X_d))
        prob += cost

        # up and lower bound of xc
        for h in range(nc):
            prob += xc[h][0] <= ub[h]
            prob += xc[h][0] >= lb[h]

        # constraints for one-hot encoded xd
        for i in range(nd):
            prob += (plp.lpSum(xd[h][0] for h in range(sum(X_d[:i]), sum(X_d[:i + 1]))) == 1)

        # constraints for one-hot encoded xint
        for i in range(nint):
            prob += (plp.lpSum(xint[h][0] for h in range(round(sum(int_interval[:i])), round(sum(int_interval[:i + 1])))) == 1)

        # constraints satisfaction
        if isLin_eqConstrained:
            for i in range(np.size(beq)):
                prob += (plp.lpSum(Aeq[i, h] * xc[h][0] for h in range(nc))
                         + plp.lpSum(Aeq[i, nc + h] * xint[h][0] for h in range(nint_encoded))
                         + plp.lpSum(Aeq[i, nci_encoded + h] * xd[h][0] for h in range(sum_X_d))
                         == beq[i][0])

        if isLin_ineqConstrained:
            for i in range(np.size(bineq)):
                prob += (plp.lpSum(Aineq[i, h] * xc[h][0] for h in range(nc))
                         + plp.lpSum(Aineq[i, nc + h] * xint[h][0] for h in range(nint_encoded))
                         + plp.lpSum(Aineq[i, nci_encoded+h] * xd[h][0] for h in range(sum_X_d))
                         <= bineq[i][0])

        # acquisition function related
        if nd > 0 and (N > 1 or fes_init):
            # discrete variables
            cat_sum_ham_dist = 0
            for i in range(N):
                ind_0 = list(np.where(X_curr_d[i, :] == 0)[0])
                ind_1 = list(np.where(X_curr_d[i, :] == 1)[0])
                cat_sum_ham_dist += plp.lpSum(xd[h][0] for h in ind_0) + plp.lpSum(1 - xd[h][0] for h in ind_1)

            prob += cat_ham_dist_scaled == 1 / (nd * N) * cat_sum_ham_dist
        else:
            prob += cat_ham_dist_scaled[0] == 0

        if nint > 0 and (N > 1 or fes_init):
            # encoded integer variables
            int_sum_ham_dist = 0
            for i in range(N):
                ind_0 = list(np.where(X_curr_int[i, :] == 0)[0])
                ind_1 = list(np.where(X_curr_int[i, :] == 1)[0])
                int_sum_ham_dist += plp.lpSum(xint[h][0] for h in ind_0) + plp.lpSum(1 - xint[h][0] for h in ind_1)

            prob += int_ham_dist_scaled == 1 / (nint * N) * int_sum_ham_dist
        else:
            prob += int_ham_dist_scaled[0] == 0

        # continuous variables
        # big-M for exploration function
        M_x = 1.0e5 * np.ones((1, nc))
        # if N >1:
        #     max_X_curr = np.zeros((1, nc))
        #     min_X_curr = np.zeros((1, nc))
        #     for h in range(nc):
        #         max_X_curr[0][h] = np.amax(X_curr_c[:, h])
        #         min_X_curr[0][h] = np.amin(X_curr_c[:, h])
        #     for i in range(nc):
        #         if max_X_curr[0,i] > 0:
        #             M_x[0,i] = 2 * max_X_curr[0,i] - min_X_curr[0,i]
        #         else:
        #             M_x[0,i] = max_X_curr[0,i] - 2 * min_X_curr[0,i]

        if (N > 1 or fes_init):
            for i in range(N):
                if nc >0:
                    for j in range(nc):
                        prob += delta_p[i][j] == 1 - delta_m[i][j]
                        prob += xc[j][0] - X_curr_c[i, j] >= beta[0] - M_x[0, j] * (1 - delta_p[i][j])
                        prob += -xc[j][0] + X_curr_c[i, j] >= beta[0] - M_x[0, j] * (1 - delta_m[i][j])
                    # prob += (plp.lpSum(delta_p[i][j] + delta_m[i][j] for j in range(nc)) >= 1)
        else:
            prob += beta[0] == 0

        if nc <1:
            prob += beta[0] == 0

        try:
            prob.solve(plp.GUROBI(timeLimit=timelimit, msg=0))
        except:
            prob.solve(plp.GLPK(timeLimit=timelimit, msg=0))
        # prob.solve()
        status = prob.status

        if status == plp.LpStatusOptimal:  # See plp.constants.LpStatus. See more in constants.py in pulp/ folder
            xopt = np.zeros((nci_encoded + sum_X_d,))
            for i in range(nc):
                xopt[i,] = xc[i][0].varValue
            for i in range(nc, nci_encoded):
                xopt[i,] = xint[i - nc][0].varValue
            for i in range(nci_encoded, nci_encoded + sum_X_d):
                xopt[i,] = xd[i - nci_encoded][0].varValue
        # elif prob.solverModel.status == 9:
        #     print("Acquisition is not solved to optimal, the current best solution is used")
        #     xopt = np.array(prob.solverModel.Xs[:nci_encoded+sum_X_d])

            # # # to assess the results:
            # delta_p_opt = np.zeros((N,nc))
            # delta_m_opt = np.zeros((N, nc))
            # for i in range(N):
            #     for j in range(nc):
            #         delta_p_opt[i,j] = delta_p[i][j].varValue
            #         delta_m_opt[i, j] = delta_m[i][j].varValue
            # cat_ham_dist_scaled = cat_ham_dist_scaled[0].varValue
            # beta_opt = beta[0].varValue
        else:
            xopt = None

        return xopt