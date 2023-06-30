"""
Encoder for categorical variables

(C) 2021-2023 Mengjia Zhu, Alberto Bemporad
"""

import numpy as np
from sklearn.preprocessing import OneHotEncoder

import sys

class cat_encoder:

    def __init__(self, prob):
        """
        Obtain and constructs all necessary attributes for self
        """
        self.nd = prob.nd
        self.X_d = prob.X_d
        self.nci = prob.nci
        self.nci_encoded = prob.nci_encoded

    def cat_encoder(self):
        """
        Generate the one-hot encoder for the categorical variables

        Inputs:
                nd: number of categorical variables to be encoded
                X_d: number of options for each categorical variable
                (for the current setup, assume X_d is known, if its unknown, can be trained from data)
                (which is not the target problem type of this algorithm (experiments are expensive to perform)

        Output:
                encoder: the one-hot encoder for the categorical variables
        """
        nd = self.nd
        X_d = self.X_d
        max_cat = max(X_d)

        xenc_gen = np.array(np.zeros([max_cat, nd]))

        enc = OneHotEncoder(drop=None, sparse=False)

        if nd != len(X_d):
            errstr_nd = "The number of discrete variables is not consistent with the length of X_d (the list indicates number of categorials for each discrete variables), please double check"
            print(errstr_nd)
            sys.exit(1)

        for ind in range(nd):
            x_d_ind = np.arange(X_d[ind])
            xenc_gen[:X_d[ind], ind] = x_d_ind

        encoder = enc.fit(xenc_gen)

        return encoder


    def encode(self, x, encoder):
        """
        Transform the categorical variables in x using the generated one-hot encoder (from 'cat_encoder' function)

        Inputs:
                x: the varaible to be encoded
                encoder: the one-hot encoder generated

        Outputs:
                xencoded: the one-hot encoded version of x

        """

        nci_encoded = self.nci_encoded
        xencoded = x[:, :nci_encoded]
        xtoencode = x[:, nci_encoded:]
        x_d_encoded = encoder.transform(xtoencode)
        xencoded = np.hstack((xencoded, x_d_encoded))

        return xencoded


    def decode(self, x, encoder):
        """
        Reverse one-hot encoding (transform the one-hot encoded variables in x to the original form)

        Inputs:
                x: the varaible to be decoded
                encoder: the one-hot encoder generated

        Outputs:
                xdecoded: the decoded version of x


        """
        nci = self.nci
        xdecoded = x[:, :nci]
        xtodecode = x[:,nci:]
        x_d_decoded = encoder.inverse_transform(xtodecode)
        xdecoded = np.hstack((xdecoded, x_d_decoded))

        return xdecoded