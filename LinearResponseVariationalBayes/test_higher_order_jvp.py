#!/usr/bin/env python3

import autograd
import autograd.numpy as np

import LinearResponseVariationalBayes as vb
import LinearResponseVariationalBayes.SparseObjectives as obj_lib
import LinearResponseVariationalBayes.HighOrderJVP as jvp_lib

import numpy.testing as np_test
import unittest

class Target(object):
    def __init__(self, dim):
        self.dim = dim
        self.a = np.random.random((dim, dim, dim))

    def obj(self, x):
        return np.einsum('ijk,i,j,k', self.a, x, x, x) / 6

class TestGenerator(unittest.TestCase):
    def test_generator(self):
        target = Target(3)
        x = np.random.random(target.dim)

        v1 = np.random.random(target.dim)
        v2 = np.random.random(target.dim)
        v3 = np.random.random(target.dim)

        target_grad = autograd.grad(target.obj)
        target_hess = autograd.hessian(target.obj)
        target_d3 = autograd.jacobian(target_hess)

        # Test append_jvp.
        jvp_gen = jvp_lib.JacobianVectorProducts(target.obj)
        fun_jvp = jvp_gen.append_jvp(target.obj)
        np_test.assert_array_almost_equal(
            np.einsum('i,i', target_grad(x), v1),
            fun_jvp(x, v1))

        fun_jvp2 = jvp_gen.append_jvp(fun_jvp)
        np_test.assert_array_almost_equal(
            np.einsum('ij,i,j', target_hess(x), v1, v2),
            fun_jvp2(x, v1, v2))

        fun_jvp3 = jvp_gen.append_jvp(fun_jvp2)
        np_test.assert_array_almost_equal(
            np.einsum('ijk,i,j,k', target_d3(x), v1, v2, v3),
            fun_jvp3(x, v1, v2, v3))

        # Test the pre-built jvps.
        np_test.assert_array_almost_equal(
            target.obj(x),
            jvp_gen.jvp_funs[0](x))
        np_test.assert_array_almost_equal(
            np.einsum('i,i', target_grad(x), v1),
            jvp_gen.jvp_funs[1](x, v1))
        np_test.assert_array_almost_equal(
            np.einsum('ij,i,j', target_hess(x), v1, v2),
            jvp_gen.jvp_funs[2](x, v1, v2))
        np_test.assert_array_almost_equal(
            np.einsum('ijk,i,j,k', target_d3(x), v1, v2, v3),
            jvp_gen.jvp_funs[3](x, v1, v2, v3))


if __name__ == '__main__':
    unittest.main()
