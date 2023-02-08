"""
Function used to express preferences (for numerical benchmarks)

Note: the exact numerical results are assumed to be unknown and only used to express preferences

Reference code: 'glis_function1' version 3.0 in glis package by A. Bemporad, 2021
                http://cse.lab.imtlucca.it/~bemporad/glis/index.html

"""

from numpy import vstack, append, array
from numpy import sum as npsum
import numpy as np

class PWASp_fun1:
    """
    preference query function for numerical benchmark

    Reference code: 'glis_function1' version 3.0 in glis package by A. Bemporad, 2021
    """

    def __init__(self, f, comparetol, Aeq,beq,Aineq, bineq):
        self.itest = 0
        self.Xtest = []
        self.Ftest = []
        self.Festest = []
        self.f = f
        self.comparetol = comparetol
        self.Aeq = Aeq
        self.beq = beq
        self.Aineq = Aineq
        self.bineq = bineq

    def clear(self):
        self.itest = 0
        self.Xtest = []
        self.Ftest = []
        self.Festest = []
        return

    def eval(self, x, y, x_encoded,y_encoded):

        xfound = False
        yfound = False

        itest = self.itest
        if itest > 0:
            Xtest = self.Xtest
            Ftest = self.Ftest
            Festest = self.Festest
        else:
            fx = self.f(x)
            fes_known_x = self.feas_check(x_encoded)
            itest = 1
            Xtest = array([x])
            Ftest = array([fx])
            Festest = array([fes_known_x])
            xfound = True

        for i in range(itest):
            if not (xfound) and npsum(abs(Xtest[i, :] - x)) <= 1e-10:
                xfound = True
                fx = Ftest[i]
                fes_known_x = Festest[i]

            if not (yfound) and npsum(abs(Xtest[i, :] - y)) <= 1e-10:
                yfound = True
                fy = Ftest[i]
                fes_known_y = Festest[i]

        if not (xfound):
            fx = self.f(x)
            fes_known_x = self.feas_check(x_encoded)
            Xtest = vstack((Xtest, x))
            Ftest = append(Ftest, fx)
            Festest = append(Festest, fes_known_x)
            itest = itest + 1

        if not (yfound):
            fy = self.f(y)
            fes_known_y = self.feas_check(y_encoded)
            Xtest = vstack((Xtest, y))
            Ftest = append(Ftest, fy)
            Festest = append(Festest, fes_known_y)
            itest = itest + 1

        # Make comparison
        if fx < fy - self.comparetol:
            if fes_known_x == 1 or (fes_known_y == 0 and fes_known_x == 0):
                out = -1
            else:
                out = 1
        elif fx > fy + self.comparetol:
            if fes_known_y == 1 or (fes_known_y == 0 and fes_known_x == 0):
                out = 1
            else:
                out = -1
        else:
            out = 0

        self.Xtest = Xtest
        self.Ftest = Ftest
        self.Festest = Festest
        self.itest = itest

        return out

    def value(self, x):
        # Compute function value, from available ones if available
        #
        # (C) 2019 A. Bemporad, September 22, 2019

        j = 0
        while j < self.itest:
            if npsum(abs(self.Xtest[j, :] - x)) <= 1e-10:
                val = self.Ftest[j]
                return val
            j = j + 1

        # Value does not exist, compute it
        val = self.f(x)
        self.Xtest = vstack((self.Xtest, x))
        self.Ftest = append(self.Ftest, val)
        self.itest = self.itest + 1
        return val

    def feas_check(self, x):
        # check if the decision variable is feasible or not ( for known constraints, which is needed to help express preferences)
        isfeas = True
        if self.Aineq.any():
            isfeas = isfeas and all(self.Aineq.dot(x) <= self.bineq.flatten("c")+ 1.0e-8)
        if self.Aeq.any():
            isfeas = isfeas and all(self.Aeq.dot(x) <= self.beq.flatten("c") + 1.0e-8)
            isfeas = isfeas and all(-self.Aeq.dot(x) <= -self.beq.flatten("c") + 1.0e-8)

        return isfeas