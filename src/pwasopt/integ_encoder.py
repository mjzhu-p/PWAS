"""
Encoder for integer variables

(C) 2021-2023 Mengjia Zhu, Alberto Bemporad
"""

import numpy as np
from sklearn.preprocessing import OneHotEncoder

import sys

class integ_encoder:

    def __init__(self, prob):
        """
        Obtain and constructs all necessary attributes for self
        """
        self.nint = prob.nint
        self.nc = prob.nc
        self.nci = prob.nci
        self.lb = prob.lb_original
        self.ub = prob.ub_original
        self.int_interval = prob.int_interval
        self.nci_encoded = prob.nci_encoded


    def integ_encoder(self):
        """
        Generate the one-hot encoder for the integer variables

        Inputs:
                nc: number of continuous variables
                nint: number of integer variables to be enoded
                int_interval: number of possible integer values for each integer variable

        Output:
                encoder: the one-hot encoder for the integer variables
        """
        nc = self.nc
        nint = self.nint
        int_interval = self.int_interval
        max_interval = int(round(max(int_interval)))
        lb = self.lb
        ub = self.ub

        xenc_gen = np.array(np.zeros([max_interval, nint]))

        enc = OneHotEncoder(drop=None, sparse=False)

        if nint != int_interval.shape[0]:
            errstr_nint = "The number of integer variables is not consistent with the length of int_interval (the list indicates number of possible integer values for each integer variables), please double check"
            print(errstr_nint)
            sys.exit(1)

        for i in range(nint):
            x_int_ind = np.arange(lb[nc + i], ub[nc + i] + 1).reshape((1,-1))
            xenc_gen[:int(int_interval[i]), i] = x_int_ind

        encoder = enc.fit(xenc_gen)

        return encoder


    def encode(self, x, encoder):
        """
        Transform the integer variables in x using the generated one-hot encoder (from 'integ_encoder' function)

        Inputs:
                x: the varaible to be encoded
                encoder: the one-hot encoder generated

        Outputs:
                xencoded: the one-hot encoded version of x

        """
        nc = self.nc
        nci = self.nci
        xencoded_nc = x[:, :nc]
        xencoded_nd = x[:, nci:]
        xtoencode = x[:, nc:nci]
        x_d_encoded = encoder.transform(xtoencode)
        xencoded = np.hstack((xencoded_nc, x_d_encoded, xencoded_nd))

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
        nc = self.nc
        # nci = self.nci
        nci_encoded = self.nci_encoded
        xdecoded_nc = x[:, :nc]
        xdecoded_nd = x[:, nci_encoded:]
        xtodecode = x[:,nc:nci_encoded]
        x_d_decoded = encoder.inverse_transform(xtodecode)
        xdecoded = np.hstack((xdecoded_nc, x_d_decoded, xdecoded_nd))

        return xdecoded