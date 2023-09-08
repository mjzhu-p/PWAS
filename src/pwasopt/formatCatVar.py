"""
function used to format the linear equality and inequality constraints after one-hot encode the categorical variables
    - used if categorical variable are involved in the constraints numerically

(C) 2021-2023 Mengjia Zhu, Alberto Bemporad
"""

import numpy as np

def formatCatVar_encoded(nc,nint,nd,lb,ub,X_d =[],Aeq=[],Aineq=[],isLin_eqConstrained=False,isLin_ineqConstrained=False):
    nci = nc + nint
    sum_X_d = int(round(sum(X_d)))
    cat_interval = np.round(ub[nci:] - lb[nci:] + 1)

    if isLin_eqConstrained:
        Aeq_cat_encode = np.zeros((Aeq.shape[0], nci+sum_X_d))
        Aeq_cat_encode[:, :nci] = Aeq[:, :nci]

        for i in range(nd):
            Aeq_cat_encode[:, nci + int(round(np.sum(cat_interval[:i]))):nci + int(round(np.sum(cat_interval[:i + 1])))] = \
                Aeq[:, nci + i].reshape((-1,1)).dot(np.arange(lb[nci +i], ub[nci + i] + 1).reshape((1,-1)))

    if isLin_ineqConstrained:
        Aineq_cat_encode = np.zeros((Aineq.shape[0], nci + sum_X_d))
        Aineq_cat_encode[:, :nci] = Aineq[:, :nci]

        for i in range(nd):
            Aineq_cat_encode[:,nci + int(round(np.sum(cat_interval[:i]))):nci + int(round(np.sum(cat_interval[:i + 1])))] = \
                Aineq[:, nci + i].reshape((-1, 1)).dot(np.arange(lb[nci + i], ub[nci + i] + 1).reshape((1, -1)))

    return Aeq_cat_encode,Aineq_cat_encode
