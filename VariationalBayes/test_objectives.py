#!/usr/bin/python3

import autograd
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
        # self.x = vb.VectorParam('x', size=dim)
        # self.y = vb.VectorParam('y', size=dim)
        self.a_mat = np.full((dim, dim), 0.1) + np.eye(dim)
        self.set_inits()

        self.preconditioner = np.eye(dim)

    def set_random(self):
        self.x.set_free(np.random.random(self.x.free_size()))

    def set_inits(self):
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

        model.set_inits()
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

        # Test the preconditioning
        preconditioner = 2.0 * np.eye(model.dim)
        preconditioner[model.dim - 1, 0] = 0.1 # Add asymmetry for testing!
        objective.preconditioner = preconditioner

        np_test.assert_array_almost_equal(
            objective.fun_free_cond(x_free),
            objective.fun_free(np.matmul(preconditioner, x_free)),
            err_msg='Conditioned function values')

        fun_free_cond_grad = autograd.grad(objective.fun_free_cond)
        grad_cond = objective.fun_free_grad_cond(x_free)
        np_test.assert_array_almost_equal(
            fun_free_cond_grad(x_free), grad_cond,
            err_msg='Conditioned gradient values')

        fun_free_cond_hessian = autograd.hessian(objective.fun_free_cond)
        hess_cond = objective.fun_free_hessian_cond(x_free)
        np_test.assert_array_almost_equal(
            fun_free_cond_hessian(x_free), hess_cond,
            err_msg='Conditioned Hessian values')

        fun_free_cond_hvp = autograd.hessian_vector_product(
            objective.fun_free_cond)
        np_test.assert_array_almost_equal(
            fun_free_cond_hvp(x_free, grad_cond),
            objective.fun_free_hvp_cond(x_free, grad_cond),
            err_msg='Conditioned Hessian vector product values')


    def test_optimization(self):
        model = Model(dim=3)
        objective = obj_lib.Objective(par=model.x, fun=model.f)
        preconditioner = 2.0 * np.eye(model.dim)
        preconditioner[model.dim - 1, 0] = 0.1 # Add asymmetry for testing!
        objective.preconditioner = preconditioner

        model.set_inits()
        x0 = model.x.get_free()
        y0 = np.linalg.solve(preconditioner, x0)

        # Unconditioned
        opt_result = sp.optimize.minimize(
            fun=objective.fun_free,
            jac=objective.fun_free_grad,
            hessp=objective.fun_free_hvp,
            x0=x0,
            method='trust-ncg',
            options={'maxiter': 100, 'disp': False, 'gtol': 1e-6 })
        self.assertTrue(opt_result.success)
        model.x.set_free(opt_result.x)
        np_test.assert_array_almost_equal(
            np.zeros(model.dim), model.x.get_vector(),
            err_msg='Trust-NCG Unconditioned')

        # Conditioned:
        opt_result = sp.optimize.minimize(
            fun=objective.fun_free_cond,
            jac=objective.fun_free_grad_cond,
            hessp=objective.fun_free_hvp_cond,
            x0=y0,
            method='trust-ncg',
            options={'maxiter': 100, 'disp': False, 'gtol': 1e-6 })
        self.assertTrue(opt_result.success)
        model.x.set_free(objective.uncondition_x(opt_result.x))
        np_test.assert_array_almost_equal(
            np.zeros(model.dim), model.x.get_vector(),
            err_msg='Trust-NCG')

        opt_result = sp.optimize.minimize(
            fun=lambda par: objective.fun_free_cond(par, verbose=False),
            jac=objective.fun_free_grad_cond,
            x0=y0,
            method='BFGS',
            options={'maxiter': 100, 'disp': False, 'gtol': 1e-6 })
        self.assertTrue(opt_result.success)
        model.x.set_free(objective.uncondition_x(opt_result.x))
        np_test.assert_array_almost_equal(
            np.zeros(model.dim), model.x.get_vector(), err_msg='BFGS')


        opt_result = sp.optimize.minimize(
            fun=lambda par: objective.fun_free_cond(par, verbose=False),
            jac=objective.fun_free_grad_cond,
            hess=objective.fun_free_hessian_cond,
            x0=y0,
            method='Newton-CG',
            options={'maxiter': 100, 'disp': False })
        self.assertTrue(opt_result.success)
        model.x.set_free(objective.uncondition_x(opt_result.x))
        np_test.assert_array_almost_equal(
            np.zeros(model.dim), model.x.get_vector(), err_msg='Newton')



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
