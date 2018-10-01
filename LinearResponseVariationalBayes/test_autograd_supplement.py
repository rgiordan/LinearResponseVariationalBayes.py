#!/usr/bin/env python3

import autograd
import autograd.numpy as np
from autograd.test_util import check_grads
import autograd.numpy.random as npr

from LinearResponseVariationalBayes import AutogradSupplement

import unittest

class TestAutogradSupplement(unittest.TestCase):
    def test_supplemental_functions(self):
        def fun(x):
            sign, logdet = np.linalg.slogdet(x)
            return logdet

        D = 6
        mat = npr.randn(D, D)
        mat[0, 0] += 1  # Make sure the matrix is not symmetric
        check_grads(fun, modes=['fwd'])(mat)
        check_grads(fun, modes=['rev'])(-mat)

if __name__ == '__main__':
    unittest.main()
