#!/usr/bin/env python3

import autograd
import autograd.numpy as np
from autograd.test_util import check_grads
import autograd.numpy.random as npr

from LinearResponseVariationalBayes import AutogradSupplement

import unittest

npr.seed(1)


class TestAutogradSupplement(unittest.TestCase):
    def test_supplemental_functions(self):
        def fun(x):
            sign, logdet = np.linalg.slogdet(x)
            return logdet

        D = 3
        mat = npr.randn(D, D)
        mat[0, 0] += 1  # Make sure the matrix is not symmetric
        mat = mat + mat.T + 10 * np.eye(D)

        print('mat', mat)
        print('inv mat', np.linalg.inv(mat))
        print('Here')
        check_grads(fun, modes=['fwd'])(mat)
        #check_grads(fun, modes=['fwd'])(-mat)

if __name__ == '__main__':
    unittest.main()
