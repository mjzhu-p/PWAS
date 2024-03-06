"""
Acquisition function used to find the next point to query in the active learning stage
    - If integer variables are
        - one-hot encoded,
            - frequency-based (hamming distance) are used for exploration
        - treated as continuous variable
            - max box method is used for exploration
    - The acquisition step may be splitted into multiple stages for different types of variables
        - may explore continuous, integer, and categorical variables in different steps
        - may explore them in one step

(C) 2021-2023 Mengjia Zhu, Alberto Bemporad
"""

import numpy as np
import pulp as plp


class active_learn:

    def __init__(self, prob):
        """
        Obtain and constructs all necessary attributes for self
        """
        self.lb = prob.lb
        self.ub = prob.ub
        self.nc = prob.nc
        self.nint = prob.nint
        self.nint_encoded = prob.nint_encoded
        self.nci = prob.nci
        self.nci_encoded = prob.nci_encoded
        self.int_encoded = prob.int_encoded
        self.int_interval = prob.int_interval
        self.nd = prob.nd
        self.X_d = prob.X_d
        self.sum_X_d = prob.sum_X_d
        self.nvars_encoded = prob.nvars_encoded
        self.maxevals = prob.maxevals

        self.scale_vars = prob.scale_vars
        self.dd_int = prob.dd_int
        self.d0_int = prob.d0_int

        self.delta_E = prob.delta_E
        self.timelimit = prob.timelimit

        self.isLin_eqConstrained = prob.isLin_eqConstrained
        self.Aeq = prob.Aeq
        self.beq = prob.beq
        self.isLin_ineqConstrained = prob.isLin_ineqConstrained
        self.Aineq = prob.Aineq
        self.bineq = prob.bineq

        self.integer_cut = prob.integer_cut

    def discrete_explore(self, X, zbest_nci, a, b, N, omega, gamma, Kf, dF):
        """
        Solve the acquisition step to obtain optimal categorical variables (encoded) based on current samples and surrogate model
            - treat the continuous variables, if exist, as constants
            - treat the integer variables, if exist, as constants
            - explore categorical variables

        Inputs:
                X: np array, training samples
                zbest_nci: np array, continuous/integer variables of the current best x
                a: np array, coefficients of the PWA
                b: np array, coefficients of the PWA
                N: int, number of training samples
                omega: np array, coefficients of the PWL
                gamma: np array, coefficients of the PWL
                Kf: number of partitions
                dF: float, scale parameter, defined as: max(max(y_pred)-min(y_pred), self.prob.epsDeltaF)

        Outputs:
                xd_opt: np array, the optimum categorical variables

        """
        nc = self.nc
        nint = self.nint
        # nci = self.nci
        nci_encoded = self.nci_encoded
        # int_encoded = self.int_encoded
        nd = self.nd
        X_d = self.X_d
        sum_X_d = self.sum_X_d

        scale_vars = self.scale_vars
        dd_int = self.dd_int
        d0_int = self.d0_int

        zbest_c_unscaled = zbest_nci.copy()
        if (not self.int_encoded) and (scale_vars) and (nint >0):
            zbest_c_unscaled[nc:, 0] = zbest_nci[nc:, 0] * (np.ones((1, 1)) * dd_int) + np.ones((1, 1)) * d0_int

        delta_E = self.delta_E
        timelimit = self.timelimit

        isLin_eqConstrained = self.isLin_eqConstrained
        Aeq = self.Aeq
        beq = self.beq
        isLin_ineqConstrained = self.isLin_ineqConstrained
        Aineq = self.Aineq
        bineq = self.bineq

        prob = plp.LpProblem('discrete_explore', plp.LpMinimize)
        xd = plp.LpVariable.dicts("xd", (range(sum_X_d), range(1)), cat=plp.LpBinary)
        y = plp.LpVariable.dicts("y", range(1), cat=plp.LpContinuous)
        z_pwl_x = plp.LpVariable.dicts("z_pwl_x", range(Kf), cat=plp.LpBinary)
        v_pwl_x = plp.LpVariable.dicts("v_pwl_x", range(Kf), cat=plp.LpContinuous)
        cat_ham_dist_scaled = plp.LpVariable.dicts("cat_ham_dist_scaled", range(1), lowBound=0, cat=plp.LpContinuous)

        # Objective function
        cost = - delta_E * plp.lpSum(cat_ham_dist_scaled) + plp.lpSum(y) / dF
        prob += cost

        # constraints for one-hot encoded xd
        for i in range(nd):
            prob += (plp.lpSum(xd[h][0] for h in range(sum(X_d[:i]), sum(X_d[:i + 1]))) == 1)

        # constraints satisfaction
        if isLin_eqConstrained:
            for i in range(np.size(beq)):
                prob += (plp.lpSum(Aeq[i, h] * zbest_c_unscaled[h][0] for h in range(nci_encoded))
                         + (plp.lpSum(Aeq[i, nci_encoded + h] * xd[h][0] for h in range(sum_X_d)))
                         == beq[i][0])

        if isLin_ineqConstrained:
            for i in range(np.size(bineq)):
                prob += (plp.lpSum(Aineq[i, h] * zbest_c_unscaled[h][0] for h in range(nci_encoded))
                         + (plp.lpSum(Aineq[i, nci_encoded + h] * xd[h][0] for h in range(sum_X_d)))
                         <= bineq[i][0])

        # Compute big-M for partition
        M = 1.0e5 * np.ones((Kf, Kf))
        # for j in range(Kf):
        #     for i in range(Kf):
        #         if not i == j:
        #             M[j, i] = sum((max(omega[i, h] - omega[j, h], 0) * ub[h]
        #                            - max(-omega[i, h] + omega[j, h], 0) * lb[h])
        #                           for h in range(nvar)) - gamma[j] + gamma[i]

        # Exclusive-or constraint
        prob += (plp.lpSum(z_pwl_x[i] for i in range(Kf)) == 1)

        # big-M constraint for PWL partition
        for j in range(Kf):
            for i in range(Kf):
                if not i == j:
                    prob += (plp.lpSum((omega[i, h] - omega[j, h]) * zbest_nci[h][0] for h in range(nci_encoded))
                             + plp.lpSum((omega[i, nci_encoded + h] - omega[j, nci_encoded + h]) * xd[h][0] for h in range(sum_X_d))
                             <= gamma[j] - gamma[i] + M[j, i] * (1 - z_pwl_x[j]))

        # Compute big-M for y
        Mcp = 1.0e5 * np.ones((Kf, 1))
        Mcm = -1.0e5 * np.ones((Kf, 1))

        # for j in range(Kf):
        #     Mcp[j, 0] = np.sum(a[j][:, 0] * (a[j][:, 0] >= 0) * ub
        #                        - (-a[j][:, 0] * (a[j][:, 0] <= 0) * lb)) + b[j, 0]
        #     Mcm[j, 0] = np.sum(a[j][:, 0] * (a[j][:, 0] >= 0) * lb
        #                        - (-a[j][:, 0] * (a[j][:, 0] <= 0) * ub)) + b[j, 0]

        # big-M constraint for y
        for j in range(Kf):
            ax = plp.lpSum(a[j][h, 0] * zbest_nci[h][0] for h in range(nci_encoded)) + plp.lpSum(
                a[j][nci_encoded + h, 0] * xd[h][0] for h in range(sum_X_d))
            prob += (v_pwl_x[j] <= ax + b[j] - Mcm[j, 0] * (1 - z_pwl_x[j]))
            prob += (v_pwl_x[j] >= ax + b[j] - Mcp[j, 0] * (1 - z_pwl_x[j]))
            prob += (v_pwl_x[j] <= Mcp[j, 0] * z_pwl_x[j])
            prob += (v_pwl_x[j] >= Mcm[j, 0] * z_pwl_x[j])

        # Set y
        prob += (y == plp.lpSum(v_pwl_x[j] for j in range(Kf)))

        # acquisition function related
        sum_ham_dist = 0
        for i in range(N):
            ind_0 = list(np.where(X[i, :] == 0)[0])
            ind_1 = list(np.where(X[i, :] == 1)[0])
            sum_ham_dist += plp.lpSum(xd[h][0] for h in ind_0) + plp.lpSum(1 - xd[h][0] for h in ind_1)

        prob += cat_ham_dist_scaled == 1 / (nd * N) * sum_ham_dist

        try:
            prob.solve(plp.GUROBI(timeLimit=timelimit, msg=0))
        except:
            prob.solve(plp.GLPK(timeLimit=timelimit, msg=0))
        status = prob.status

        if status == plp.LpStatusOptimal:  # See plp.constants.LpStatus. See more in constants.py in pulp/ folder
            xd_opt = np.zeros((sum_X_d,))
            for i in range(sum_X_d):
                xd_opt[i,] = xd[i][0].varValue
        else:
            xd_opt = np.nan

            # # to assess the results:
            # z_pwl_x_opt = np.zeros((Kf))
            # for i in range(Kf):
            #     z_pwl_x_opt[i,] = z_pwl_x[i].varValue
            # cat_ham_dist_scaled = cat_ham_dist_scaled[0].varValue
            # y_opt = y[0].varValue
            # cost_opt = - delta_E * cat_ham_dist_scaled + y_opt / dF
            # # cost_opt = - delta_E * beta_opt + y_opt

        # status_msg = plp.LpStatus[status]

        return xd_opt


    def integ_explore(self, X, zbest_nc, zbest_d, a, b, N, omega, gamma, Kf, dF):
        """
        When integer variables are not one-hot encoded (max box)
        Solve the acquisition step to obtain optimal integer variables based on current samples and surrogate model
            - treat the categorical variables, if exist, as constants
            - treat the continuous variables, if exist, as constants
            - explore integer variables

        Inputs:
                X: np array, training samples
                zbest_nc: np array, continuous variables of the current best x
                zbest_d: np array, categorical variables of the current best x
                a: np array, coefficients of the PWA
                b: np array, coefficients of the PWA
                N: int, number of training samples
                omega: np array, coefficients of the PWL
                gamma: np array, coefficients of the PWL
                Kf: number of partitions
                dF: float, scale parameter, defined as: max(max(y_pred)-min(y_pred), self.prob.epsDeltaF)

        Outputs:
                xc_opt: np array, the optimum continuous/integer variables

        """
        lb = self.lb
        ub = self.ub
        nc = self.nc
        nint = self.nint
        nci = self.nci
        sum_X_d = self.sum_X_d


        scale_vars = self.scale_vars
        dd_int = self.dd_int
        d0_int = self.d0_int

        delta_E = self.delta_E
        timelimit = self.timelimit

        isLin_eqConstrained = self.isLin_eqConstrained
        Aeq = self.Aeq
        beq = self.beq
        isLin_ineqConstrained = self.isLin_ineqConstrained
        Aineq = self.Aineq
        bineq = self.bineq

        prob = plp.LpProblem('integ_explore', plp.LpMinimize)
        xint_c = plp.LpVariable.dicts("xint_c", (range(nint), range(1)), cat=plp.LpContinuous)
        xint = plp.LpVariable.dicts("xint", (range(nint), range(1)), cat=plp.LpInteger)
        y = plp.LpVariable.dicts("y", range(1), cat=plp.LpContinuous)
        z_pwl_x = plp.LpVariable.dicts("z_pwl_x", range(Kf), cat=plp.LpBinary)
        v_pwl_x = plp.LpVariable.dicts("v_pwl_x", range(Kf), cat=plp.LpContinuous)
        if nint >0:
            if (N * nint < 120):
                delta_p_int = plp.LpVariable.dicts("delta_p_int", (range(N), range(nint)), cat=plp.LpBinary)
                delta_m_int = plp.LpVariable.dicts("delta_m_int", (range(N), range(nint)), cat=plp.LpBinary)
            else:
                delta_p_int = plp.LpVariable.dicts("delta_p_int", (range(20), range(nint)), cat=plp.LpBinary)
                delta_m_int = plp.LpVariable.dicts("delta_m_int", (range(20), range(nint)), cat=plp.LpBinary)
        beta_int = plp.LpVariable.dicts("beta_int", range(1), lowBound=0, cat=plp.LpContinuous)

        # Objective function
        cost = - delta_E * (plp.lpSum(beta_int)) + plp.lpSum(y) / dF
        prob += cost

        # up and lower bound of xint
        for h in range(nint):
            prob += xint[h][0] <= np.round(ub[nc + h])
            prob += xint[h][0] >= np.round(lb[nc + h])

        # relationship between xint and xint_scaled, i.e., xint_c
        for h in range(nint):
            if scale_vars:
                prob += xint_c[h][0] * dd_int[h] + d0_int[h] == xint[h][0]
            else:
                prob += xint_c[h][0] == xint[h][0]

        # constraints satisfaction
        if isLin_eqConstrained:
            for i in range(np.size(beq)):
                prob += (plp.lpSum(Aeq[i, h] * zbest_nc[h][0] for h in range(nc))
                         + plp.lpSum(Aeq[i, nc + h] * xint[h][0] for h in range(nint))
                         + (plp.lpSum(Aeq[i, nci + h] * zbest_d[h][0] for h in range(sum_X_d)))
                         == beq[i][0])

        if isLin_ineqConstrained:
            for i in range(np.size(bineq)):
                prob += (plp.lpSum(Aineq[i, h] * zbest_nc[h][0] for h in range(nc))
                         + plp.lpSum(Aineq[i, nc + h] * xint[h][0] for h in range(nint))
                         + (plp.lpSum(Aineq[i, nci + h] * zbest_d[h][0] for h in range(sum_X_d)))
                         <= bineq[i][0])

        # Compute big-M for partition
        M = 1.0e5 * np.ones((Kf, Kf))
        # for j in range(Kf):
        #     for i in range(Kf):
        #         if not i == j:
        #             M[j, i] = sum((max(omega[i, h] - omega[j, h], 0) * ub[h]
        #                            - max(-omega[i, h] + omega[j, h], 0) * lb[h])
        #                           for h in range(nvar)) - gamma[j] + gamma[i]

        # Exclusive-or constraint
        prob += (plp.lpSum(z_pwl_x[i] for i in range(Kf)) == 1)

        # big-M constraint for PWL partition
        for j in range(Kf):
            for i in range(Kf):
                if not i == j:
                    prob += (plp.lpSum((omega[i, h] - omega[j, h]) * zbest_nc[h][0] for h in range(nc))
                             + plp.lpSum((omega[i, nc + h] - omega[j, nc + h]) * xint_c[h][0] for h in range(nint))
                             + plp.lpSum((omega[i, nci + h] - omega[j, nci + h]) * zbest_d[h][0] for h in range(sum_X_d))
                             <= gamma[j] - gamma[i] + M[j, i] * (1 - z_pwl_x[j]))

        # Compute big-M for y
        Mcp = max(1.0e5,dF*10) * np.ones((Kf, 1))
        Mcm = -max(1.0e5,dF*10) * np.ones((Kf, 1))

        # for j in range(Kf):
        #     Mcp[j, 0] = np.sum(a[j][:, 0] * (a[j][:, 0] >= 0) * ub
        #                        - (-a[j][:, 0] * (a[j][:, 0] <= 0) * lb)) + b[j, 0]
        #     Mcm[j, 0] = np.sum(a[j][:, 0] * (a[j][:, 0] >= 0) * lb
        #                        - (-a[j][:, 0] * (a[j][:, 0] <= 0) * ub)) + b[j, 0]

        # big-M constraint for y
        for j in range(Kf):
            ax = (plp.lpSum(a[j][h, 0] * zbest_nc[h][0] for h in range(nc))
                  + plp.lpSum(a[j][nc + h, 0] * xint_c[h][0] for h in range(nint))
                  + plp.lpSum(a[j][nci + h, 0] * zbest_d[h][0] for h in range(sum_X_d)))
            prob += (v_pwl_x[j] <= ax + b[j] - Mcm[j, 0] * (1 - z_pwl_x[j]))
            prob += (v_pwl_x[j] >= ax + b[j] - Mcp[j, 0] * (1 - z_pwl_x[j]))
            prob += (v_pwl_x[j] <= Mcp[j, 0] * z_pwl_x[j])
            prob += (v_pwl_x[j] >= Mcm[j, 0] * z_pwl_x[j])

        # Set y
        prob += (y == plp.lpSum(v_pwl_x[j] for j in range(Kf)))

        # acquisition function related

        # big-M for exploration function
        M_x = 1.0e5 * np.ones((1, nint))
        # M_x = np.zeros((1, nci))
        # max_X_curr = np.zeros((1, nci))
        # min_X_curr = np.zeros((1, nci))
        # for h in range(nci):
        #     max_X_curr[0][h] = np.amax(X[:, h])
        #     min_X_curr[0][h] = np.amin(X[:, h])
        # for i in range(nci):
        #     if max_X_curr[0,i] > 0:
        #         M_x[0,i] = 2 * max_X_curr[0,i] - min_X_curr[0,i]
        #     else:
        #         M_x[0,i] = max_X_curr[0,i] - 2 * min_X_curr[0,i]

        if nint >0:
            if (N * nint < 120):
                for i in range(N):
                    for j in range(nint):
                        prob += delta_p_int[i][j] <= 1 - delta_m_int[i][j]
                        prob += (xint_c[j][0] - X[i, j])*dd_int[j] >= beta_int[0] - M_x[0, j] * (1 - delta_p_int[i][j])
                        prob += (-xint_c[j][0] + X[i, j])*dd_int[j] >= beta_int[0] - M_x[0, j] * (1 - delta_m_int[i][j])
                    prob += (plp.lpSum(delta_p_int[i][j] + delta_m_int[i][j] for j in range(nint)) >= 1)
            else:
                N_iter = N - 20
                for i in range(20):
                    for j in range(nint):
                        prob += delta_p_int[i][j] <= 1 - delta_m_int[i][j]
                        prob += (xint_c[j][0] - X[N_iter+i, j]) * dd_int[j] >= beta_int[0] - M_x[0, j] * (1 - delta_p_int[i][j])
                        prob += (-xint_c[j][0] + X[N_iter+i, j]) * dd_int[j] >= beta_int[0] - M_x[0, j] * (1 - delta_m_int[i][j])
                prob += (plp.lpSum(delta_p_int[i][j] + delta_m_int[i][j] for j in range(nint)) >= 1)
        else:
            prob += beta_int[0] == 0

        try:
            prob.solve(plp.GUROBI(timeLimit=timelimit, msg=0))
        except:
            prob.solve(plp.GLPK(timeLimit=timelimit, msg=0))
        status = prob.status

        if status == plp.LpStatusOptimal:  # See plp.constants.LpStatus. See more in constants.py in pulp/ folder
            xint_c_opt = np.zeros((nint,))
            for i in range(nint):
                xint_c_opt[i,] = xint_c[i][0].varValue
        else:
            xint_c_opt = np.nan

            # # to assess the results:
            # z_pwl_x_opt = np.zeros((Kf))
            # for i in range(Kf):
            #     z_pwl_x_opt[i,] = z_pwl_x[i].varValue
            # beta_opt = beta[0].varValue
            # y_opt = y[0].varValue
            # cost_opt = - delta_E * beta_opt + y_opt / dF
            # # cost_opt = - delta_E * beta_opt + y_opt

        # status_msg = plp.LpStatus[status]

        return xint_c_opt


    def integ_explore_intEncoded(self, X, zbest_nc, zbest_d, a, b, N, omega, gamma, Kf, dF):
        """
        When integer variables are one-hot encoded (hamming distance)
        Solve the acquisition step to obtain optimal integer variables based on current samples and surrogate model
            - treat the categorical variables, if exist, as constants
            - treat the continuous variables, if exist, as constants
            - explore integer variables

        Inputs:
                X: np array, training samples
                zbest_nc: np array, continuous variables of the current best x
                zbest_d: np array, categorical variables of the current best x
                a: np array, coefficients of the PWA
                b: np array, coefficients of the PWA
                N: int, number of training samples
                omega: np array, coefficients of the PWL
                gamma: np array, coefficients of the PWL
                Kf: number of partitions
                dF: float, scale parameter, defined as: max(max(y_pred)-min(y_pred), self.prob.epsDeltaF)

        Outputs:
                xc_opt: np array, the optimum continuous/integer variables

        """
        nc = self.nc
        nint = self.nint
        nci_encoded = self.nci_encoded
        nint_encoded = self.nint_encoded
        int_interval = self.int_interval
        sum_X_d = self.sum_X_d

        delta_E = self.delta_E
        timelimit = self.timelimit

        isLin_eqConstrained = self.isLin_eqConstrained
        Aeq = self.Aeq
        beq = self.beq
        isLin_ineqConstrained = self.isLin_ineqConstrained
        Aineq = self.Aineq
        bineq = self.bineq

        prob = plp.LpProblem('integ_explore_intEncoded', plp.LpMinimize)
        xint = plp.LpVariable.dicts("xint", (range(nint_encoded), range(1)), cat=plp.LpBinary)
        y = plp.LpVariable.dicts("y", range(1), cat=plp.LpContinuous)
        z_pwl_x = plp.LpVariable.dicts("z_pwl_x", range(Kf), cat=plp.LpBinary)
        v_pwl_x = plp.LpVariable.dicts("v_pwl_x", range(Kf), cat=plp.LpContinuous)
        int_ham_dist_scaled = plp.LpVariable.dicts("int_ham_dist_scaled", range(1), lowBound=0, cat=plp.LpContinuous)

        # Objective function
        cost = - delta_E * (plp.lpSum(int_ham_dist_scaled)) + plp.lpSum(y) / dF
        prob += cost

        # constraints for one-hot encoded xint
        for i in range(nint):
            prob += (plp.lpSum(
                xint[h][0] for h in range(int(round(sum(int_interval[:i]))), int(round(sum(int_interval[:i + 1]))))) == 1)

        # constraints satisfaction
        if isLin_eqConstrained:
            for i in range(np.size(beq)):
                prob += (plp.lpSum(Aeq[i, h] * zbest_nc[h][0] for h in range(nc))
                         + plp.lpSum(Aeq[i, nc + h] * xint[h][0] for h in range(nint_encoded))
                         + (plp.lpSum(Aeq[i, nci_encoded + h] * zbest_d[h][0] for h in range(sum_X_d)))
                         == beq[i][0])

        if isLin_ineqConstrained:
            for i in range(np.size(bineq)):
                prob += (plp.lpSum(Aineq[i, h] * zbest_nc[h][0] for h in range(nc))
                         + plp.lpSum(Aineq[i, nc + h] * xint[h][0] for h in range(nint_encoded))
                         + (plp.lpSum(Aineq[i, nci_encoded + h] * zbest_d[h][0] for h in range(sum_X_d)))
                         <= bineq[i][0])

        # Compute big-M for partition
        M = 1.0e5 * np.ones((Kf, Kf))
        # for j in range(Kf):
        #     for i in range(Kf):
        #         if not i == j:
        #             M[j, i] = sum((max(omega[i, h] - omega[j, h], 0) * ub[h]
        #                            - max(-omega[i, h] + omega[j, h], 0) * lb[h])
        #                           for h in range(nvar)) - gamma[j] + gamma[i]

        # Exclusive-or constraint
        prob += (plp.lpSum(z_pwl_x[i] for i in range(Kf)) == 1)

        # big-M constraint for PWL partition
        for j in range(Kf):
            for i in range(Kf):
                if not i == j:
                    prob += (plp.lpSum((omega[i, h] - omega[j, h]) * zbest_nc[h][0] for h in range(nc))
                             + plp.lpSum((omega[i, nc + h] - omega[j, nc + h]) * xint[h][0] for h in range(nint_encoded))
                             + plp.lpSum((omega[i, nci_encoded + h] - omega[j, nci_encoded + h]) * zbest_d[h][0] for h in range(sum_X_d))
                             <= gamma[j] - gamma[i] + M[j, i] * (1 - z_pwl_x[j]))

        # Compute big-M for y
        Mcp = max(1.0e5,dF*10) * np.ones((Kf, 1))
        Mcm = -max(1.0e5,dF*10) * np.ones((Kf, 1))

        # for j in range(Kf):
        #     Mcp[j, 0] = np.sum(a[j][:, 0] * (a[j][:, 0] >= 0) * ub
        #                        - (-a[j][:, 0] * (a[j][:, 0] <= 0) * lb)) + b[j, 0]
        #     Mcm[j, 0] = np.sum(a[j][:, 0] * (a[j][:, 0] >= 0) * lb
        #                        - (-a[j][:, 0] * (a[j][:, 0] <= 0) * ub)) + b[j, 0]

        # big-M constraint for y
        for j in range(Kf):
            ax = (plp.lpSum(a[j][h, 0] * zbest_nc[h][0] for h in range(nc))
                  + plp.lpSum(a[j][nc + h, 0] * xint[h][0] for h in range(nint_encoded))
                  + plp.lpSum(a[j][nci_encoded + h, 0] * zbest_d[h][0] for h in range(sum_X_d)))
            prob += (v_pwl_x[j] <= ax + b[j] - Mcm[j, 0] * (1 - z_pwl_x[j]))
            prob += (v_pwl_x[j] >= ax + b[j] - Mcp[j, 0] * (1 - z_pwl_x[j]))
            prob += (v_pwl_x[j] <= Mcp[j, 0] * z_pwl_x[j])
            prob += (v_pwl_x[j] >= Mcm[j, 0] * z_pwl_x[j])

        # Set y
        prob += (y == plp.lpSum(v_pwl_x[j] for j in range(Kf)))

        # acquisition function related
        int_sum_ham_dist = 0
        for i in range(N):
            ind_0 = list(np.where(X[i, :] == 0)[0])
            ind_1 = list(np.where(X[i, :] == 1)[0])
            int_sum_ham_dist += plp.lpSum(xint[h][0] for h in ind_0) + plp.lpSum(1 - xint[h][0] for h in ind_1)

        prob += int_ham_dist_scaled == 1 / (nint * N) * int_sum_ham_dist

        try:
            prob.solve(plp.GUROBI(timeLimit=timelimit, msg=0))
        except:
            prob.solve(plp.GLPK(timeLimit=timelimit, msg=0))
        status = prob.status

        if status == plp.LpStatusOptimal:  # See plp.constants.LpStatus. See more in constants.py in pulp/ folder
            xint_opt = np.zeros((nint_encoded,))
            for i in range(nint_encoded):
                xint_opt[i,] = xint[i][0].varValue
        else:
            xint_opt = np.nan

            # # to assess the results:
            # z_pwl_x_opt = np.zeros((Kf))
            # for i in range(Kf):
            #     z_pwl_x_opt[i,] = z_pwl_x[i].varValue
            # beta_opt = beta[0].varValue
            # y_opt = y[0].varValue
            # cost_opt = - delta_E * beta_opt + y_opt / dF
            # # cost_opt = - delta_E * beta_opt + y_opt

        # status_msg = plp.LpStatus[status]

        return xint_opt


    def cont_explore(self, X, zbest_int_c, zbest_d, a, b, N, omega, gamma, Kf, dF):
        """
        Solve the acquisition step to obtain optimal continuous variables based on current samples and surrogate model
            - treat the categorical variables, if exist, as constants
            - treat the integer variables, if exist, as constants
            - explore continuous variables

        Inputs:
                X: np array, training samples
                zbest_int_c: np array, integer variables of the current best x (scaled if not one-hot encoded and if scale_vars = True)
                zbest_d: np array, categorical variables of the current best x
                a: np array, coefficients of the PWA
                b: np array, coefficients of the PWA
                N: int, number of training samples
                omega: np array, coefficients of the PWL
                gamma: np array, coefficients of the PWL
                Kf: number of partitions
                dF: float, scale parameter, defined as: max(max(y_pred)-min(y_pred), self.prob.epsDeltaF)

        Outputs:
                xc_opt: np array, the optimum continuous/integer variables

        """
        lb = self.lb
        ub = self.ub
        nc = self.nc
        nci_encoded = self.nci_encoded
        nint_encoded = self.nint_encoded
        sum_X_d = self.sum_X_d

        scale_vars = self.scale_vars
        dd_int = self.dd_int
        d0_int = self.d0_int

        zbest_int_c_unscaled = zbest_int_c.copy()
        if (not self.int_encoded) and (scale_vars) and (nint_encoded >0):
            zbest_int_c_unscaled[:,0] = zbest_int_c[:,0] * (np.ones((1, 1)) * dd_int) + np.ones((1, 1)) * d0_int

        delta_E = self.delta_E
        timelimit = self.timelimit

        isLin_eqConstrained = self.isLin_eqConstrained
        Aeq = self.Aeq
        beq = self.beq
        isLin_ineqConstrained = self.isLin_ineqConstrained
        Aineq = self.Aineq
        bineq = self.bineq

        prob = plp.LpProblem('cont_explore', plp.LpMinimize)
        xc = plp.LpVariable.dicts("xc", (range(nc), range(1)), cat=plp.LpContinuous)
        y = plp.LpVariable.dicts("y", range(1), cat=plp.LpContinuous)
        z_pwl_x = plp.LpVariable.dicts("z_pwl_x", range(Kf), cat=plp.LpBinary)
        v_pwl_x = plp.LpVariable.dicts("v_pwl_x", range(Kf), cat=plp.LpContinuous)
        if nc >0:
            if (N * nc < 120):
                delta_p = plp.LpVariable.dicts("delta_p", (range(N), range(nc)), cat=plp.LpBinary)
                delta_m = plp.LpVariable.dicts("delta_m", (range(N), range(nc)), cat=plp.LpBinary)
            else:
                delta_p = plp.LpVariable.dicts("delta_p", (range(20), range(nc)), cat=plp.LpBinary)
                delta_m = plp.LpVariable.dicts("delta_m", (range(20), range(nc)), cat=plp.LpBinary)
        beta = plp.LpVariable.dicts("beta", range(1), lowBound=0, cat=plp.LpContinuous)

        # Objective function
        cost = - delta_E * (plp.lpSum(beta)) + plp.lpSum(y) / dF
        prob += cost

        # up and lower bound of xc
        for h in range(nc):
            prob += xc[h][0] <= ub[h]
            prob += xc[h][0] >= lb[h]

        # constraints satisfaction
        if isLin_eqConstrained:
            for i in range(np.size(beq)):
                prob += (plp.lpSum(Aeq[i, h] * xc[h][0] for h in range(nc))
                         + plp.lpSum(Aeq[i, nc + h] * zbest_int_c_unscaled[h][0] for h in range(nint_encoded))
                         + (plp.lpSum(Aeq[i, nci_encoded + h] * zbest_d[h][0] for h in range(sum_X_d)))
                         == beq[i][0])

        if isLin_ineqConstrained:
            for i in range(np.size(bineq)):
                prob += (plp.lpSum(Aineq[i, h] * xc[h][0] for h in range(nc))
                         + plp.lpSum(Aineq[i, nc + h] * zbest_int_c_unscaled[h][0] for h in range(nint_encoded))
                         + (plp.lpSum(Aineq[i, nci_encoded + h] * zbest_d[h][0] for h in range(sum_X_d)))
                         <= bineq[i][0])

        # Compute big-M for partition
        M = 1.0e5 * np.ones((Kf, Kf))
        # for j in range(Kf):
        #     for i in range(Kf):
        #         if not i == j:
        #             M[j, i] = sum((max(omega[i, h] - omega[j, h], 0) * ub[h]
        #                            - max(-omega[i, h] + omega[j, h], 0) * lb[h])
        #                           for h in range(nvar)) - gamma[j] + gamma[i]

        # Exclusive-or constraint
        prob += (plp.lpSum(z_pwl_x[i] for i in range(Kf)) == 1)

        # big-M constraint for PWL partition
        for j in range(Kf):
            for i in range(Kf):
                if not i == j:
                    prob += (plp.lpSum((omega[i, h] - omega[j, h]) * xc[h][0] for h in range(nc))
                             + plp.lpSum((omega[i, nc + h] - omega[j, nc + h]) * zbest_int_c[h][0] for h in range(nint_encoded))
                             + plp.lpSum((omega[i, nci_encoded + h] - omega[j, nci_encoded + h]) * zbest_d[h][0] for h in range(sum_X_d))
                             <= gamma[j] - gamma[i] + M[j, i] * (1 - z_pwl_x[j]))

        # Compute big-M for y
        Mcp = max(1.0e5,dF*10) * np.ones((Kf, 1))
        Mcm = -max(1.0e5,dF*10) * np.ones((Kf, 1))

        # for j in range(Kf):
        #     Mcp[j, 0] = np.sum(a[j][:, 0] * (a[j][:, 0] >= 0) * ub
        #                        - (-a[j][:, 0] * (a[j][:, 0] <= 0) * lb)) + b[j, 0]
        #     Mcm[j, 0] = np.sum(a[j][:, 0] * (a[j][:, 0] >= 0) * lb
        #                        - (-a[j][:, 0] * (a[j][:, 0] <= 0) * ub)) + b[j, 0]

        # big-M constraint for y
        for j in range(Kf):
            ax = (plp.lpSum(a[j][h, 0] * xc[h][0] for h in range(nc))
                  + plp.lpSum(a[j][nc + h, 0] * zbest_int_c[h][0] for h in range(nint_encoded))
                  + plp.lpSum(a[j][nci_encoded + h, 0] * zbest_d[h][0] for h in range(sum_X_d)))
            prob += (v_pwl_x[j] <= ax + b[j] - Mcm[j, 0] * (1 - z_pwl_x[j]))
            prob += (v_pwl_x[j] >= ax + b[j] - Mcp[j, 0] * (1 - z_pwl_x[j]))
            prob += (v_pwl_x[j] <= Mcp[j, 0] * z_pwl_x[j])
            prob += (v_pwl_x[j] >= Mcm[j, 0] * z_pwl_x[j])

        # Set y
        prob += (y == plp.lpSum(v_pwl_x[j] for j in range(Kf)))

        # acquisition function related

        # big-M for exploration function
        M_x = 1.0e5 * np.ones((1, nc))
        # M_x = np.zeros((1, nci))
        # max_X_curr = np.zeros((1, nci))
        # min_X_curr = np.zeros((1, nci))
        # for h in range(nci):
        #     max_X_curr[0][h] = np.amax(X[:, h])
        #     min_X_curr[0][h] = np.amin(X[:, h])
        # for i in range(nci):
        #     if max_X_curr[0,i] > 0:
        #         M_x[0,i] = 2 * max_X_curr[0,i] - min_X_curr[0,i]
        #     else:
        #         M_x[0,i] = max_X_curr[0,i] - 2 * min_X_curr[0,i]

        if nc > 0:
            if (N * nc < 120):
                for i in range(N):
                    for j in range(nc):
                        prob += delta_p[i][j] <= 1 - delta_m[i][j]
                        prob += xc[j][0] - X[i, j] >= beta[0] - M_x[0, j] * (1 - delta_p[i][j])
                        prob += -xc[j][0] + X[i, j] >= beta[0] - M_x[0, j] * (1 - delta_m[i][j])
                    prob += (plp.lpSum(delta_p[i][j] + delta_m[i][j] for j in range(nc)) >= 1)
            else:
                N_iter = N - 20
                for i in range(20):
                    for j in range(nc):
                        prob += delta_p[i][j] <= 1 - delta_m[i][j]
                        prob += xc[j][0] - X[N_iter + i, j] >= beta[0] - M_x[0, j] * (1 - delta_p[i][j])
                        prob += -xc[j][0] + X[N_iter + i, j] >= beta[0] - M_x[0, j] * (1 - delta_m[i][j])
                    prob += (plp.lpSum(delta_p[i][j] + delta_m[i][j] for j in range(nc)) >= 1)
        else:
            prob += beta[0] == 0

        try:
            prob.solve(plp.GUROBI(timeLimit=timelimit, msg=0))
        except:
            prob.solve(plp.GLPK(timeLimit=timelimit, msg=0))
        status = prob.status

        if status == plp.LpStatusOptimal:  # See plp.constants.LpStatus. See more in constants.py in pulp/ folder
            xc_opt = np.zeros((nc,))
            for i in range(nc):
                xc_opt[i,] = xc[i][0].varValue
        else:
            xc_opt = np.nan

            # # to assess the results:
            # z_pwl_x_opt = np.zeros((Kf))
            # for i in range(Kf):
            #     z_pwl_x_opt[i,] = z_pwl_x[i].varValue
            # beta_opt = beta[0].varValue
            # y_opt = y[0].varValue
            # cost_opt = - delta_E * beta_opt + y_opt / dF
            # # cost_opt = - delta_E * beta_opt + y_opt

        # status_msg = plp.LpStatus[status]

        return xc_opt


    def acq_explore(self, X, a, b, N, omega, gamma, Kf, dF):
        """
        When integer variables are NOT one-hot encoded
        Solve the acquisition step to obtain optimal point to query based on current samples and surrogate model
            - obtain the continuous, integer, and categorical variables, if exist, all in one-step
            - explore continuous, integer, and categorical variables

        Inputs:
                X: np array, training samples
                a: np array, coefficients of the PWA
                b: np array, coefficients of the PWA
                N: int, number of training samples
                omega: np array, coefficients of the PWL
                gamma: np array, coefficients of the PWL
                Kf: number of partitions
                dF: float, scale parameter, defined as: max(max(y_pred)-min(y_pred), self.prob.epsDeltaF)

        Outputs:
                xopt: np array, the optimum variables (next point to query)

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

        X_curr_c = X[:, :nci]
        X_curr_d = X[:, nci:]

        delta_E = self.delta_E
        timelimit = self.timelimit

        isLin_eqConstrained = self.isLin_eqConstrained
        Aeq = self.Aeq
        beq = self.beq
        isLin_ineqConstrained = self.isLin_ineqConstrained
        Aineq = self.Aineq
        bineq = self.bineq

        integer_cut = self.integer_cut

        prob = plp.LpProblem('acq_explore', plp.LpMinimize)
        xc = plp.LpVariable.dicts("xc", (range(nci), range(1)), cat=plp.LpContinuous)
        xd = plp.LpVariable.dicts("xd", (range(sum_X_d), range(1)), cat=plp.LpBinary)
        xint = plp.LpVariable.dicts("xint", (range(nint), range(1)), cat=plp.LpInteger)
        y = plp.LpVariable.dicts("y", range(1), cat=plp.LpContinuous)
        z_pwl_x = plp.LpVariable.dicts("z_pwl_x", range(Kf), cat=plp.LpBinary)
        v_pwl_x = plp.LpVariable.dicts("v_pwl_x", range(Kf), cat=plp.LpContinuous)
        if nc >0:
            if (N * nc < 120) and (N * nint < 120):
                delta_p = plp.LpVariable.dicts("delta_p", (range(N), range(nci)), cat=plp.LpBinary)
                delta_m = plp.LpVariable.dicts("delta_m", (range(N), range(nci)), cat=plp.LpBinary)
            else:
                delta_p = plp.LpVariable.dicts("delta_p", (range(20), range(nci)), cat=plp.LpBinary)
                delta_m = plp.LpVariable.dicts("delta_m", (range(20), range(nci)), cat=plp.LpBinary)
        beta = plp.LpVariable.dicts("beta", range(1), lowBound=0, cat=plp.LpContinuous)
        if nint >0:
            if (N * nc < 120) and (N * nint < 120):
                delta_p_int = plp.LpVariable.dicts("delta_p_int", (range(N), range(nint)), cat=plp.LpBinary)
                delta_m_int = plp.LpVariable.dicts("delta_m_int", (range(N), range(nint)), cat=plp.LpBinary)
            else:
                delta_p_int = plp.LpVariable.dicts("delta_p_int", (range(20), range(nint)), cat=plp.LpBinary)
                delta_m_int = plp.LpVariable.dicts("delta_m_int", (range(20), range(nint)), cat=plp.LpBinary)
        beta_int = plp.LpVariable.dicts("beta_int", range(1), lowBound=0, cat=plp.LpContinuous)
        cat_ham_dist_scaled = plp.LpVariable.dicts("cat_ham_dist_scaled", range(1), lowBound=0, cat=plp.LpContinuous)

        if integer_cut and (nc < 1 and nint > 0):
            already_sampled = [[plp.LpVariable(f"var{j}_same_as_previous_{i}", cat=plp.LpBinary) for j in range(nint+sum_X_d)] for i in range(N)]
            as_z1 = [[plp.LpVariable(f"as_z1{j}_same_as_previous_{i}", cat=plp.LpBinary) for j in range(nint+sum_X_d)] for i in range(N)]
            as_z2 = [[plp.LpVariable(f"as_z2{j}_same_as_previous_{i}", cat=plp.LpBinary) for j in range(nint + sum_X_d)] for i in range(N)]


        # Objective function
        cost = - delta_E * (plp.lpSum(cat_ham_dist_scaled) + plp.lpSum(beta) + plp.lpSum(beta_int)) + plp.lpSum(y) / dF
        prob += cost

        # up and lower bound of xc
        for h in range(nc):
            prob += xc[h][0] <= ub[h]
            prob += xc[h][0] >= lb[h]

        # up and lower bound of xint
        for h in range(nint):
            prob += xint[h][0] <= np.round(ub[nc + h])
            prob += xint[h][0] >= np.round(lb[nc + h])

        # relationship between xint and xint_scaled
        for h in range(nint):
            if scale_vars:
                prob += xc[nc + h][0] * dd_int[h] + d0_int[h] == xint[h][0]
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
                         + plp.lpSum(Aineq[i, nci + h] * xd[h][0] for h in range(sum_X_d))
                         <= bineq[i][0])

        # Compute big-M for partition
        M = 1.0e5 * np.ones((Kf, Kf))
        # for j in range(Kf):
        #     for i in range(Kf):
        #         if not i == j:
        #             M[j, i] = sum((max(omega[i, h] - omega[j, h], 0) * ub[h]
        #                            - max(-omega[i, h] + omega[j, h], 0) * lb[h])
        #                           for h in range(nvar)) - gamma[j] + gamma[i]

        # Exclusive-or constraint
        prob += (plp.lpSum(z_pwl_x[i] for i in range(Kf)) == 1)

        # big-M constraint for PWL partition
        for j in range(Kf):
            for i in range(Kf):
                if not i == j:
                    prob += (plp.lpSum((omega[i, h] - omega[j, h]) * xc[h][0] for h in range(nci))
                             + plp.lpSum((omega[i, nci + h] - omega[j, nci + h]) * xd[h][0] for h in range(sum_X_d))
                             <= gamma[j] - gamma[i] + M[j, i] * (1 - z_pwl_x[j]))

        # Compute big-M for y
        Mcp = max(1.0e5,dF*10) * np.ones((Kf, 1))
        Mcm = -max(1.0e5,dF*10) * np.ones((Kf, 1))

        # for j in range(Kf):
        #     Mcp[j, 0] = np.sum(a[j][:, 0] * (a[j][:, 0] >= 0) * ub
        #                        - (-a[j][:, 0] * (a[j][:, 0] <= 0) * lb)) + b[j, 0]
        #     Mcm[j, 0] = np.sum(a[j][:, 0] * (a[j][:, 0] >= 0) * lb
        #                        - (-a[j][:, 0] * (a[j][:, 0] <= 0) * ub)) + b[j, 0]

        # big-M constraint for y
        for j in range(Kf):
            ax = (plp.lpSum(a[j][h, 0] * xc[h][0] for h in range(nci))
                  + plp.lpSum(a[j][nci + h, 0] * xd[h][0] for h in range(sum_X_d)))
            prob += (v_pwl_x[j] <= ax + b[j] - Mcm[j, 0] * (1 - z_pwl_x[j]))
            prob += (v_pwl_x[j] >= ax + b[j] - Mcp[j, 0] * (1 - z_pwl_x[j]))
            prob += (v_pwl_x[j] <= Mcp[j, 0] * z_pwl_x[j])
            prob += (v_pwl_x[j] >= Mcm[j, 0] * z_pwl_x[j])

        # Set y
        prob += (y == plp.lpSum(v_pwl_x[j] for j in range(Kf)))

        # acquisition function related
        if nd > 0:
            # discrete variables
            sum_ham_dist = 0
            for i in range(N):
                ind_0 = list(np.where(X_curr_d[i, :] == 0)[0])
                ind_1 = list(np.where(X_curr_d[i, :] == 1)[0])
                sum_ham_dist += plp.lpSum(xd[h][0] for h in ind_0) + plp.lpSum(1 - xd[h][0] for h in ind_1)

            prob += cat_ham_dist_scaled == 1 / (nd * N) * sum_ham_dist
        else:
            prob += cat_ham_dist_scaled[0] == 0

        # continuous variables
        # big-M for exploration function
        M_x = 1.0e5 * np.ones((1, nci))

        # max_X_curr = np.zeros((1, nci))
        # min_X_curr = np.zeros((1, nci))
        # for h in range(nci):
        #     max_X_curr[0][h] = np.amax(X_curr_c[:, h])
        #     min_X_curr[0][h] = np.amin(X_curr_c[:, h])
        # for i in range(nci):
        #     if max_X_curr[0,i] > 0:
        #         M_x[0,i] = 2 * max_X_curr[0,i] - min_X_curr[0,i]
        #     else:
        #         M_x[0,i] = max_X_curr[0,i] - 2 * min_X_curr[0,i]

        if nc >0:
            if (N * nc < 120) and (N * nint < 120):
                for i in range(N):
                    for j in range(nc):
                        prob += delta_p[i][j] <= 1 - delta_m[i][j]
                        prob += xc[j][0] - X[i, j] >= beta[0] - M_x[0, j] * (1 - delta_p[i][j])
                        prob += -xc[j][0] + X[i, j] >= beta[0] - M_x[0, j] * (1 - delta_m[i][j])
                    prob += (plp.lpSum(delta_p[i][j] + delta_m[i][j] for j in range(nc)) >= 1)
            else:
                N_iter = N - 20
                for i in range(20):
                    for j in range(nc):
                        prob += delta_p[i][j] <= 1 - delta_m[i][j]
                        prob += xc[j][0] - X[N_iter + i, j] >= beta[0] - M_x[0, j] * (1 - delta_p[i][j])
                        prob += -xc[j][0] + X[N_iter + i, j] >= beta[0] - M_x[0, j] * (1 - delta_m[i][j])
                    prob += (plp.lpSum(delta_p[i][j] + delta_m[i][j] for j in range(nc)) >= 1)
        else:
            prob += beta[0] == 0

        if nint >0:
            if (N * nc < 120) and (N * nint < 120):
                for i in range(N):
                    for j in range(nint):
                        prob += delta_p_int[i][j] <= 1 - delta_m_int[i][j]
                        prob += (xc[nc+j][0] - X[i, nc+j])*dd_int[j] >= beta_int[0] - M_x[0, nc+j] * (1 - delta_p_int[i][j])
                        prob += (-xc[nc+j][0] + X[i, nc+j])*dd_int[j]  >= beta_int[0] - M_x[0, nc+j] * (1 - delta_m_int[i][j])
                    prob += (plp.lpSum(delta_p_int[i][j] + delta_m_int[i][j] for j in range(nint)) >= 1)
            else:
                N_iter = N - 20
                for i in range(20):
                    for j in range(nint):
                        prob += delta_p_int[i][j] <= 1 - delta_m_int[i][j]
                        prob += (xc[nc+j][0] - X[N_iter + i, nc+j])*dd_int[j] >= beta_int[0] - M_x[0, nc+j] * (1 - delta_p_int[i][j])
                        prob += (-xc[nc+j][0] + X[N_iter + i, nc+j])*dd_int[j]  >= beta_int[0] - M_x[0, nc+j] * (1 - delta_m_int[i][j])
                    prob += (plp.lpSum(delta_p_int[i][j] + delta_m_int[i][j] for j in range(nint)) >= 1)
        else:
            prob += beta_int[0] == 0


        if integer_cut and (nc < 1 and nint >0): # include relevant variables for integer cut (when continuous variable is not present but integer variables are present)
            M_intCut = 1.0e5
            for j in range(nint+sum_X_d):
                for i in range(N):
                    if j < nint:
                        if scale_vars:
                            prob += xc[nc+j][0] - X_curr_c[i,nc+j] + 0.5 <= M_intCut * as_z1[i][j]
                            prob += X_curr_c[i, nc + j] - xc[nc + j][0] - 0.5 <= M_intCut * (1 - as_z1[i][j])
                            prob += xc[nc+j][0] - X_curr_c[i,nc+j] - 0.5 <= M_intCut * (1 - as_z2[i][j])
                            prob += X_curr_c[i, nc + j] - xc[nc + j][0] + 0.5 <= M_intCut * as_z2[i][j]
                        else:
                            prob += xint[j][0] - X_curr_c[i, nc + j] + 0.5 <= M_intCut * as_z1[i][j]
                            prob += X_curr_c[i, nc + j] - xint[j][0] - 0.5 <= M_intCut * (1 - as_z1[i][j])
                            prob += xint[j][0] - X_curr_c[i, nc + j] - 0.5 <= M_intCut * (1 - as_z2[i][j])
                            prob += X_curr_c[i, nc + j] - xint[j][0] + 0.5 <= M_intCut * as_z2[i][j]
                    else:
                        prob += xd[j-nint][0] - X_curr_d[i,j-nint] + 0.5 <= M_intCut * as_z1[i][j]
                        prob += X_curr_d[i,j-nint] - xd[j-nint][0] - 0.5 <= M_intCut * (1 - as_z1[i][j])
                        prob += xd[j-nint][0] - X_curr_d[i,j-nint] - 0.5 <= M_intCut * (1 - as_z2[i][j])
                        prob += X_curr_d[i,j-nint] - xd[j-nint][0] + 0.5 <= M_intCut * as_z2[i][j]

                    prob += already_sampled[i][j] >= as_z1[i][j] + as_z2[i][j] - 1


            for i in range(N):
                prob += sum(already_sampled[i]) <= nint+sum_X_d - 1


        try:
            prob.solve(plp.GUROBI(timeLimit=timelimit, msg=0))
        except:
            prob.solve(plp.GLPK(timeLimit=timelimit, msg=0))
        status = prob.status

        if status == plp.LpStatusOptimal:  # See plp.constants.LpStatus. See more in constants.py in pulp/ folder
            xopt = np.zeros((nci + sum_X_d,))
            for i in range(nci):
                xopt[i,] = xc[i][0].varValue
            for i in range(nci, nci + sum_X_d):
                xopt[i,] = xd[i - nci][0].varValue
        else:
            xopt = np.nan

            # # to assess the results:
            # z_pwl_x_opt = np.zeros((Kf))
            # for i in range(Kf):
            #     z_pwl_x_opt[i,] = z_pwl_x[i].varValue
            # cat_ham_dist_scaled = cat_ham_dist_scaled[0].varValue
            # beta_opt = beta[0].varValue
            # y_opt = y[0].varValue
            # cost_opt = - delta_E * (cat_ham_dist_scaled + beta_opt) + y_opt / dF
            # cost_opt = - delta_E * beta_opt + y_opt

        # status_msg = plp.LpStatus[status]

        return xopt


    def acq_explore_intEncoded(self, X, a, b, N, omega, gamma, Kf, dF):
        """
        When integer variables are one-hot encoded
        Solve the acquisition step to obtain optimal point to query based on current samples and surrogate model
            - obtain the continuous, integer, and categorical variables, if exist, all in one-step
            - explore continuous, integer, and categorical variables


        Inputs:
                X: np array, training samples
                a: np array, coefficients of the PWA
                b: np array, coefficients of the PWA
                N: int, number of training samples
                omega: np array, coefficients of the PWL
                gamma: np array, coefficients of the PWL
                Kf: number of partitions
                dF: float, scale parameter, defined as: max(max(y_pred)-min(y_pred), self.prob.epsDeltaF)

        Outputs:
                xopt: np array, the optimum variables (next point to query)

        """
        lb = self.lb
        ub = self.ub
        nc = self.nc
        nint = self.nint
        nci_encoded = self.nci_encoded
        nint_encoded = self.nint_encoded
        int_interval = self.int_interval
        nd = self.nd
        X_d = self.X_d
        sum_X_d = self.sum_X_d

        # X_curr_c = X[:, :nc]
        X_curr_int = X[:, nc:nci_encoded]
        X_curr_d = X[:, nci_encoded:]

        delta_E = self.delta_E
        timelimit = self.timelimit

        isLin_eqConstrained = self.isLin_eqConstrained
        Aeq = self.Aeq
        beq = self.beq
        isLin_ineqConstrained = self.isLin_ineqConstrained
        Aineq = self.Aineq
        bineq = self.bineq

        prob = plp.LpProblem('acq_explore_intEncoded', plp.LpMinimize)
        xc = plp.LpVariable.dicts("xc", (range(nc), range(1)), cat=plp.LpContinuous)
        xint = plp.LpVariable.dicts("xint", (range(nint_encoded), range(1)), cat=plp.LpBinary)
        xd = plp.LpVariable.dicts("xd", (range(sum_X_d), range(1)), cat=plp.LpBinary)
        y = plp.LpVariable.dicts("y", range(1), cat=plp.LpContinuous)
        z_pwl_x = plp.LpVariable.dicts("z_pwl_x", range(Kf), cat=plp.LpBinary)
        v_pwl_x = plp.LpVariable.dicts("v_pwl_x", range(Kf), cat=plp.LpContinuous)
        if nc > 0:
            if (N * nc < 120):
                delta_p = plp.LpVariable.dicts("delta_p", (range(N), range(nc)), cat=plp.LpBinary)
                delta_m = plp.LpVariable.dicts("delta_m", (range(N), range(nc)), cat=plp.LpBinary)
            else:
                delta_p = plp.LpVariable.dicts("delta_p", (range(20), range(nc)), cat=plp.LpBinary)
                delta_m = plp.LpVariable.dicts("delta_m", (range(20), range(nc)), cat=plp.LpBinary)
        beta = plp.LpVariable.dicts("beta", range(1), lowBound=0, cat=plp.LpContinuous)
        int_ham_dist_scaled = plp.LpVariable.dicts("int_ham_dist_scaled", range(1), lowBound=0, cat=plp.LpContinuous)
        cat_ham_dist_scaled = plp.LpVariable.dicts("cat_ham_dist_scaled", range(1), lowBound=0, cat=plp.LpContinuous)

        # Objective function
        cost = - delta_E * (plp.lpSum(cat_ham_dist_scaled) + plp.lpSum(beta) + plp.lpSum(int_ham_dist_scaled)) + plp.lpSum(y) / dF
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
            prob += (plp.lpSum(xint[h][0] for h in range(int(round(sum(int_interval[:i]))), int(round(sum(int_interval[:i + 1]))))) == 1)


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
                         + plp.lpSum(Aineq[i, nci_encoded + h] * xd[h][0] for h in range(sum_X_d))
                         <= bineq[i][0])

        # Compute big-M for partition
        M = 1.0e5 * np.ones((Kf, Kf))
        # for j in range(Kf):
        #     for i in range(Kf):
        #         if not i == j:
        #             M[j, i] = sum((max(omega[i, h] - omega[j, h], 0) * ub[h]
        #                            - max(-omega[i, h] + omega[j, h], 0) * lb[h])
        #                           for h in range(nvar)) - gamma[j] + gamma[i]

        # Exclusive-or constraint
        prob += (plp.lpSum(z_pwl_x[i] for i in range(Kf)) == 1)

        # big-M constraint for PWL partition
        for j in range(Kf):
            for i in range(Kf):
                if not i == j:
                    prob += (plp.lpSum((omega[i, h] - omega[j, h]) * xc[h][0] for h in range(nc))
                             + plp.lpSum((omega[i, nc+h] - omega[j, nc+h]) * xint[h][0] for h in range(nint_encoded))
                             + plp.lpSum((omega[i, nci_encoded + h] - omega[j, nci_encoded + h]) * xd[h][0] for h in range(sum_X_d))
                             <= gamma[j] - gamma[i] + M[j, i] * (1 - z_pwl_x[j]))

        # Compute big-M for y
        Mcp = max(1.0e5,dF*10) * np.ones((Kf, 1))
        Mcm = -max(1.0e5,dF*10) * np.ones((Kf, 1))

        # for j in range(Kf):
        #     Mcp[j, 0] = np.sum(a[j][:, 0] * (a[j][:, 0] >= 0) * ub
        #                        - (-a[j][:, 0] * (a[j][:, 0] <= 0) * lb)) + b[j, 0]
        #     Mcm[j, 0] = np.sum(a[j][:, 0] * (a[j][:, 0] >= 0) * lb
        #                        - (-a[j][:, 0] * (a[j][:, 0] <= 0) * ub)) + b[j, 0]

        # big-M constraint for y
        for j in range(Kf):
            ax = (plp.lpSum(a[j][h, 0] * xc[h][0] for h in range(nc))
                  + + plp.lpSum(a[j][nc + h, 0] * xint[h][0] for h in range(nint_encoded))
                  + plp.lpSum(a[j][nci_encoded + h, 0] * xd[h][0] for h in range(sum_X_d)))
            prob += (v_pwl_x[j] <= ax + b[j] - Mcm[j, 0] * (1 - z_pwl_x[j]))
            prob += (v_pwl_x[j] >= ax + b[j] - Mcp[j, 0] * (1 - z_pwl_x[j]))
            prob += (v_pwl_x[j] <= Mcp[j, 0] * z_pwl_x[j])
            prob += (v_pwl_x[j] >= Mcm[j, 0] * z_pwl_x[j])

        # Set y
        prob += (y == plp.lpSum(v_pwl_x[j] for j in range(Kf)))

        # acquisition function related
        if nd > 0:
            # discrete variables
            cat_sum_ham_dist = 0
            for i in range(N):
                ind_0 = list(np.where(X_curr_d[i, :] == 0)[0])
                ind_1 = list(np.where(X_curr_d[i, :] == 1)[0])
                cat_sum_ham_dist += plp.lpSum(xd[h][0] for h in ind_0) + plp.lpSum(1 - xd[h][0] for h in ind_1)

            prob += cat_ham_dist_scaled == 1 / (nd * N) * cat_sum_ham_dist
        else:
            prob += cat_ham_dist_scaled[0] == 0


        if nint > 0:
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

        # max_X_curr = np.zeros((1, nci))
        # min_X_curr = np.zeros((1, nci))
        # for h in range(nci):
        #     max_X_curr[0][h] = np.amax(X_curr_c[:, h])
        #     min_X_curr[0][h] = np.amin(X_curr_c[:, h])
        # for i in range(nci):
        #     if max_X_curr[0,i] > 0:
        #         M_x[0,i] = 2 * max_X_curr[0,i] - min_X_curr[0,i]
        #     else:
        #         M_x[0,i] = max_X_curr[0,i] - 2 * min_X_curr[0,i]

        if nc > 0:
            if (N * nc < 120):
                for i in range(N):
                    for j in range(nc):
                        prob += delta_p[i][j] <= 1 - delta_m[i][j]
                        prob += xc[j][0] - X[i, j] >= beta[0] - M_x[0, j] * (1 - delta_p[i][j])
                        prob += -xc[j][0] + X[i, j] >= beta[0] - M_x[0, j] * (1 - delta_m[i][j])
                    prob += (plp.lpSum(delta_p[i][j] + delta_m[i][j] for j in range(nc)) >= 1)
            else:
                N_iter = N - 20
                for i in range(20):
                    for j in range(nc):
                        prob += delta_p[i][j] <= 1 - delta_m[i][j]
                        prob += xc[j][0] - X[N_iter + i, j] >= beta[0] - M_x[0, j] * (1 - delta_p[i][j])
                        prob += -xc[j][0] + X[N_iter + i, j] >= beta[0] - M_x[0, j] * (1 - delta_m[i][j])
                    prob += (plp.lpSum(delta_p[i][j] + delta_m[i][j] for j in range(nc)) >= 1)
        else:
            prob += beta[0] == 0


        try:
            prob.solve(plp.GUROBI(timeLimit=timelimit, msg=0))
        except:
            prob.solve(plp.GLPK(timeLimit=timelimit, msg=0))
        status = prob.status

        if status == plp.LpStatusOptimal:  # See plp.constants.LpStatus. See more in constants.py in pulp/ folder
            xopt = np.zeros((nci_encoded + sum_X_d,))
            for i in range(nc):
                xopt[i,] = xc[i][0].varValue
            for i in range(nc, nci_encoded):
                xopt[i,] = xint[i - nc][0].varValue
            for i in range(nci_encoded, nci_encoded + sum_X_d):
                xopt[i,] = xd[i - nci_encoded][0].varValue
        else:
            xopt = np.nan

            # # to assess the results:
            # z_pwl_x_opt = np.zeros((Kf))
            # for i in range(Kf):
            #     z_pwl_x_opt[i,] = z_pwl_x[i].varValue
            # cat_ham_dist_scaled = cat_ham_dist_scaled[0].varValue
            # beta_opt = beta[0].varValue
            # y_opt = y[0].varValue
            # cost_opt = - delta_E * (cat_ham_dist_scaled + beta_opt) + y_opt / dF
            # cost_opt = - delta_E * beta_opt + y_opt

        # status_msg = plp.LpStatus[status]

        return xopt


    def acq_surrogate(self, a, b, omega, gamma, Kf, dF):
        """
        When integer variables are NOT one-hot encoded
        Solve the acquisition step to obtain optimal point to query based on the surrogate model
            - obtain the continuous, integer, and categorical variables, if exist, all in one-step
            - without exploration, based on the surrogate model ONLY

        Inputs:
                a: np array, coefficients of the PWA
                b: np array, coefficients of the PWA
                N: int, number of training samples
                omega: np array, coefficients of the PWL
                gamma: np array, coefficients of the PWL
                Kf: number of partitions
                dF: float, scale parameter, defined as: max(max(y_pred)-min(y_pred), self.prob.epsDeltaF)

        Outputs:
                xopt: np array, the optimum variables (next point to query)

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

        timelimit = self.timelimit

        isLin_eqConstrained = self.isLin_eqConstrained
        Aeq = self.Aeq
        beq = self.beq
        isLin_ineqConstrained = self.isLin_ineqConstrained
        Aineq = self.Aineq
        bineq = self.bineq

        prob = plp.LpProblem('Acq_exp', plp.LpMinimize)
        xc = plp.LpVariable.dicts("xc", (range(nci), range(1)), cat=plp.LpContinuous)
        xd = plp.LpVariable.dicts("xd", (range(sum_X_d), range(1)), cat=plp.LpBinary)
        xint = plp.LpVariable.dicts("xint", (range(nint), range(1)), cat=plp.LpInteger)
        y = plp.LpVariable.dicts("y", range(1), cat=plp.LpContinuous)
        z_pwl_x = plp.LpVariable.dicts("z_pwl_x", range(Kf), cat=plp.LpBinary)
        v_pwl_x = plp.LpVariable.dicts("v_pwl_x", range(Kf), cat=plp.LpContinuous)

        # Objective function
        cost =  plp.lpSum(y) / dF
        prob += cost

        # up and lower bound of xc
        for h in range(nc):
            prob += xc[h][0] <= ub[h]
            prob += xc[h][0] >= lb[h]

        # up and lower bound of xint
        for h in range(nint):
            prob += xint[h][0] <= np.round(ub[nc + h])
            prob += xint[h][0] >= np.round(lb[nc + h])

        # relationship between xint and xint_scaled
        for h in range(nint):
            if scale_vars:
                prob += xc[nc + h][0] * dd_int[h] + d0_int[h] == xint[h][0]
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
                         + plp.lpSum(Aineq[i, nci + h] * xd[h][0] for h in range(sum_X_d))
                         <= bineq[i][0])

        # Compute big-M for partition
        M = 1.0e5 * np.ones((Kf, Kf))
        # for j in range(Kf):
        #     for i in range(Kf):
        #         if not i == j:
        #             M[j, i] = sum((max(omega[i, h] - omega[j, h], 0) * ub[h]
        #                            - max(-omega[i, h] + omega[j, h], 0) * lb[h])
        #                           for h in range(nvar)) - gamma[j] + gamma[i]

        # Exclusive-or constraint
        prob += (plp.lpSum(z_pwl_x[i] for i in range(Kf)) == 1)

        # big-M constraint for PWL partition
        for j in range(Kf):
            for i in range(Kf):
                if not i == j:
                    prob += (plp.lpSum((omega[i, h] - omega[j, h]) * xc[h][0] for h in range(nci))
                             + plp.lpSum((omega[i, nci + h] - omega[j, nci + h]) * xd[h][0] for h in range(sum_X_d))
                             <= gamma[j] - gamma[i] + M[j, i] * (1 - z_pwl_x[j]))

        # Compute big-M for y
        Mcp = max(1.0e5,dF*10) * np.ones((Kf, 1))
        Mcm = -max(1.0e5,dF*10) * np.ones((Kf, 1))

        # for j in range(Kf):
        #     Mcp[j, 0] = np.sum(a[j][:, 0] * (a[j][:, 0] >= 0) * ub
        #                        - (-a[j][:, 0] * (a[j][:, 0] <= 0) * lb)) + b[j, 0]
        #     Mcm[j, 0] = np.sum(a[j][:, 0] * (a[j][:, 0] >= 0) * lb
        #                        - (-a[j][:, 0] * (a[j][:, 0] <= 0) * ub)) + b[j, 0]

        # big-M constraint for y
        for j in range(Kf):
            ax = (plp.lpSum(a[j][h, 0] * xc[h][0] for h in range(nci))
                  + plp.lpSum(a[j][nci + h, 0] * xd[h][0] for h in range(sum_X_d)))
            prob += (v_pwl_x[j] <= ax + b[j] - Mcm[j, 0] * (1 - z_pwl_x[j]))
            prob += (v_pwl_x[j] >= ax + b[j] - Mcp[j, 0] * (1 - z_pwl_x[j]))
            prob += (v_pwl_x[j] <= Mcp[j, 0] * z_pwl_x[j])
            prob += (v_pwl_x[j] >= Mcm[j, 0] * z_pwl_x[j])

        # Set y
        prob += (y == plp.lpSum(v_pwl_x[j] for j in range(Kf)))


        try:
            prob.solve(plp.GUROBI(timeLimit=timelimit, msg=0))
        except:
            prob.solve(plp.GLPK(timeLimit=timelimit, msg=0))
        status = prob.status

        if status == plp.LpStatusOptimal:  # See plp.constants.LpStatus. See more in constants.py in pulp/ folder
            xopt = np.zeros((nci + sum_X_d,))
            for i in range(nci):
                xopt[i,] = xc[i][0].varValue
            for i in range(nci, nci + sum_X_d):
                xopt[i,] = xd[i - nci][0].varValue

            # # to assess the results:
            # z_pwl_x_opt = np.zeros((Kf))
            # for i in range(Kf):
            #     z_pwl_x_opt[i,] = z_pwl_x[i].varValue
            # cat_ham_dist_scaled = cat_ham_dist_scaled[0].varValue
            # beta_opt = beta[0].varValue
            # y_opt = y[0].varValue
            # cost_opt = - delta_E * (cat_ham_dist_scaled + beta_opt) + y_opt / dF
            # cost_opt = - delta_E * beta_opt + y_opt

        # status_msg = plp.LpStatus[status]

        return xopt


    def acq_surrogate_intEncoded(self, a, b, omega, gamma, Kf, dF):
        """
        When integer variables are one-hot encoded
        Solve the acquisition step to obtain optimal point to query based on the surrogate model
            - obtain the continuous, integer, and categorical variables, if exist, all in one-step
            - without exploration, based on the surrogate model ONLY

        Inputs:
                a: np array, coefficients of the PWA
                b: np array, coefficients of the PWA
                N: int, number of training samples
                omega: np array, coefficients of the PWL
                gamma: np array, coefficients of the PWL
                Kf: number of partitions
                dF: float, scale parameter, defined as: max(max(y_pred)-min(y_pred), self.prob.epsDeltaF)

        Outputs:
                xopt: np array, the optimum variables (next point to query)

        """
        lb = self.lb
        ub = self.ub
        nc = self.nc
        nint = self.nint
        nint_encoded = self.nint_encoded
        int_interval = self.int_interval
        nci_encoded = self.nci_encoded
        nd = self.nd
        X_d = self.X_d
        sum_X_d = self.sum_X_d

        timelimit = self.timelimit

        isLin_eqConstrained = self.isLin_eqConstrained
        Aeq = self.Aeq
        beq = self.beq
        isLin_ineqConstrained = self.isLin_ineqConstrained
        Aineq = self.Aineq
        bineq = self.bineq

        prob = plp.LpProblem('Acq_exp', plp.LpMinimize)
        xc = plp.LpVariable.dicts("xc", (range(nc), range(1)), cat=plp.LpContinuous)
        xint = plp.LpVariable.dicts("xint", (range(nint_encoded), range(1)), cat=plp.LpBinary)
        xd = plp.LpVariable.dicts("xd", (range(sum_X_d), range(1)), cat=plp.LpBinary)
        y = plp.LpVariable.dicts("y", range(1), cat=plp.LpContinuous)
        z_pwl_x = plp.LpVariable.dicts("z_pwl_x", range(Kf), cat=plp.LpBinary)
        v_pwl_x = plp.LpVariable.dicts("v_pwl_x", range(Kf), cat=plp.LpContinuous)

        # Objective function
        cost =  plp.lpSum(y) / dF
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
            prob += (plp.lpSum(xint[h][0] for h in range(int(round(sum(int_interval[:i]))), int(round(sum(int_interval[:i + 1]))))) == 1)

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
                         + plp.lpSum(Aineq[i, nci_encoded + h] * xd[h][0] for h in range(sum_X_d))
                         <= bineq[i][0])

        # Compute big-M for partition
        M = 1.0e5 * np.ones((Kf, Kf))
        # for j in range(Kf):
        #     for i in range(Kf):
        #         if not i == j:
        #             M[j, i] = sum((max(omega[i, h] - omega[j, h], 0) * ub[h]
        #                            - max(-omega[i, h] + omega[j, h], 0) * lb[h])
        #                           for h in range(nvar)) - gamma[j] + gamma[i]

        # Exclusive-or constraint
        prob += (plp.lpSum(z_pwl_x[i] for i in range(Kf)) == 1)

        # big-M constraint for PWL partition
        for j in range(Kf):
            for i in range(Kf):
                if not i == j:
                    prob += (plp.lpSum((omega[i, h] - omega[j, h]) * xc[h][0] for h in range(nc))
                             + plp.lpSum(
                                (omega[i, nc + h] - omega[j, nc + h]) * xint[h][0] for h in range(nint_encoded))
                             + plp.lpSum((omega[i, nci_encoded + h] - omega[j, nci_encoded + h]) * xd[h][0] for h in
                                         range(sum_X_d))
                             <= gamma[j] - gamma[i] + M[j, i] * (1 - z_pwl_x[j]))

        # Compute big-M for y
        Mcp = max(1.0e5,dF*10) * np.ones((Kf, 1))
        Mcm = -max(1.0e5,dF*10) * np.ones((Kf, 1))

        # for j in range(Kf):
        #     Mcp[j, 0] = np.sum(a[j][:, 0] * (a[j][:, 0] >= 0) * ub
        #                        - (-a[j][:, 0] * (a[j][:, 0] <= 0) * lb)) + b[j, 0]
        #     Mcm[j, 0] = np.sum(a[j][:, 0] * (a[j][:, 0] >= 0) * lb
        #                        - (-a[j][:, 0] * (a[j][:, 0] <= 0) * ub)) + b[j, 0]

        # big-M constraint for y
        for j in range(Kf):
            ax = (plp.lpSum(a[j][h, 0] * xc[h][0] for h in range(nc))
                  + + plp.lpSum(a[j][nc + h, 0] * xint[h][0] for h in range(nint_encoded))
                  + plp.lpSum(a[j][nci_encoded + h, 0] * xd[h][0] for h in range(sum_X_d)))
            prob += (v_pwl_x[j] <= ax + b[j] - Mcm[j, 0] * (1 - z_pwl_x[j]))
            prob += (v_pwl_x[j] >= ax + b[j] - Mcp[j, 0] * (1 - z_pwl_x[j]))
            prob += (v_pwl_x[j] <= Mcp[j, 0] * z_pwl_x[j])
            prob += (v_pwl_x[j] >= Mcm[j, 0] * z_pwl_x[j])

        # Set y
        prob += (y == plp.lpSum(v_pwl_x[j] for j in range(Kf)))


        try:
            prob.solve(plp.GUROBI(timeLimit=timelimit, msg=0))
        except:
            prob.solve(plp.GLPK(timeLimit=timelimit, msg=0))
        status = prob.status

        if status == plp.LpStatusOptimal:  # See plp.constants.LpStatus. See more in constants.py in pulp/ folder
            xopt = np.zeros((nci_encoded + sum_X_d,))
            for i in range(nc):
                xopt[i,] = xc[i][0].varValue
            for i in range(nc, nci_encoded):
                xopt[i,] = xint[i - nc][0].varValue
            for i in range(nci_encoded, nci_encoded + sum_X_d):
                xopt[i,] = xd[i - nci_encoded][0].varValue

            # # to assess the results:
            # z_pwl_x_opt = np.zeros((Kf))
            # for i in range(Kf):
            #     z_pwl_x_opt[i,] = z_pwl_x[i].varValue
            # cat_ham_dist_scaled = cat_ham_dist_scaled[0].varValue
            # beta_opt = beta[0].varValue
            # y_opt = y[0].varValue
            # cost_opt = - delta_E * (cat_ham_dist_scaled + beta_opt) + y_opt / dF
            # cost_opt = - delta_E * beta_opt + y_opt

        # status_msg = plp.LpStatus[status]

        return xopt