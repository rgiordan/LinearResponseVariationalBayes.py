#!/usr/bin/python3

from autograd import numpy as np
import scipy as sp
import numpy.testing as np_test
import unittest
import VariationalBayes as vb
import VariationalBayes.SparseObjectives as obj_lib

class Model(object):
    def __init__(self, dim):
        self.dim = dim
        self.x = vb.VectorParam('x', size=dim, lb=-2.0, ub=5.0)
        self.y = vb.VectorParam('y', size=dim, lb=-2.0, ub=5.0)
        self.a_mat = np.full((dim, dim), 0.1) + np.eye(dim)
        self.set_ones()

        self.preconditioner = np.eye(dim)

    def set_random(self):
        self.x.set_free(np.random.random(self.x.free_size()))

    def set_ones(self):
        #self.x.set_vector(np.ones(self.dim))
        self.x.set_vector(np.linspace(0., 1., self.dim))

    def set_zeros(self):
        self.x.set_vector(np.zeros(self.dim))

    def f_of_x(self, x):
        return np.matmul(x.T, np.matmul(self.a_mat, x))

    def f(self):
        return self.f_of_x(self.x.get())

    def f_conditioned(self):
        # Note that
        # df / dy = (dx' / dy) df / dx = (dy / dx')^{-1} df / dx
        # So the transform should multiply by the inverse of the preconditioner.
        y_free = np.matmul(self.preconditioner, self.x.get_free())
        self.y.set_free(y_free)
        return self.f_of_x(self.y.get())



class TestObjectiveClass(unittest.TestCase):
    # For every parameter type, execute all the required methods.
    def test_objective(self):
        model = Model(dim=3)
        objective = obj_lib.Objective(par=model.x, fun=model.f)

        model.set_ones()
        x_free = model.x.get_free()
        x_vec = model.x.get_vector()

        model.set_zeros()
        self.assertTrue(objective.fun_free(x_free) > 0.0)
        np_test.assert_array_almost_equal(
            objective.fun_free(x_free), objective.fun_vector(x_vec))

        grad = objective.fun_free_grad(x_free)
        hess = objective.fun_free_hessian(x_free)
        np_test.assert_array_almost_equal(
            np.matmul(hess, grad), objective.fun_free_hvp(x_free, grad))

        model.set_zeros()
        self.assertTrue(objective.fun_vector(x_vec) > 0.0)
        grad = objective.fun_vector_grad(x_vec)
        hess = objective.fun_vector_hessian(x_vec)
        np_test.assert_array_almost_equal(
            np.matmul(hess, grad), objective.fun_vector_hvp(x_free, grad))

        # Check the preconditioning
        preconditioner = 2.0 * np.eye(model.dim)
        preconditioner[model.dim - 1, 0] = 0.1 # Add asymmetry for testing!
        objective.preconditioner = preconditioner

        # Test by evaluating an objective direclty in terms of the
        # transformed variable.
        model.preconditioner = preconditioner
        direct_objective = \
            obj_lib.Objective(par=model.x, fun=model.f_conditioned)

        y = np.matmul(preconditioner, x_free)
        np_test.assert_array_almost_equal(
            direct_objective.fun_free(x_free),
            objective.fun_free(y))

        np_test.assert_array_almost_equal(
            objective.fun_free_cond(x_free),
            objective.fun_free(y))

        grad_cond = objective.fun_free_grad_cond(y)
        np_test.assert_array_almost_equal(
            direct_objective.fun_free_grad(x_free), grad_cond)

        hessian_cond = objective.fun_free_hessian_cond(y)
        np_test.assert_array_almost_equal(
            direct_objective.fun_free_hessian(x_free),
            hessian_cond)

        np_test.assert_array_almost_equal(
            direct_objective.fun_free_hvp(x_free, grad),
            objective.fun_free_hvp_cond(y, grad))


class TestSparsePacking(unittest.TestCase):
    def test_packing(self):
        dense_mat = np.zeros((3, 3))
        dense_mat[0, 0] = 2.0
        dense_mat[0, 1] = 3.0
        dense_mat[2, 1] = 4.0

        sparse_mat = sp.sparse.csr_matrix(dense_mat)
        sparse_mat_packed = obj_lib.pack_csr_matrix(sparse_mat)
        sparse_mat_unpacked = obj_lib.unpack_csr_matrix(sparse_mat_packed)

        np_test.assert_array_almost_equal(
            dense_mat, sparse_mat_unpacked.todense())


if __name__ == '__main__':
    unittest.main()
