#!/usr/bin/env python3

import LinearResponseVariationalBayes as vb
import LinearResponseVariationalBayes.SparseObjectives as obj_lib
import LinearResponseVariationalBayes.OptimizationUtils as opt_lib
import autograd.numpy as np
import numpy.testing as np_test
import unittest

class QuadraticModel(object):
    def __init__(self, dim):
        self.dim = dim
        self.param = vb.VectorParam('theta', size=dim)

        vec = np.linspace(0.1, 0.3, num=dim)
        self.matrix = np.outer(vec, vec) + np.eye(dim)
        self.vec = vec

        self.objective = obj_lib.Objective(self.param, self.get_objective)

    def get_objective(self):
        theta = self.param.get()
        objective = 0.5 * theta.T @ self.matrix @ theta + self.vec @ theta
        return objective

    # Testing functions that use the fact that the optimum has a closed form.
    def get_true_optimum_theta(self):
        theta = -1 * np.linalg.solve(self.matrix, self.vec)
        return theta

    def get_true_optimum(self):
        # ...in the free parameterization.
        theta = self.get_true_optimum_theta()
        self.param.set_vector(theta)
        return self.param.get_free()


class TestOptimizationUtils(unittest.TestCase):
    def test_optimzation_utils(self):
        model = QuadraticModel(3)
        init_x = np.zeros(model.dim)

        opt_x, opt_result = opt_lib.minimize_objective_bfgs(
            model.objective, init_x, precondition=False)
        np_test.assert_array_almost_equal(
            model.get_true_optimum(), opt_result.x)
        np_test.assert_array_almost_equal(
            model.get_true_optimum(), opt_x)

        opt_x, opt_result = opt_lib.minimize_objective_trust_ncg(
            model.objective, init_x, precondition=False)
        np_test.assert_array_almost_equal(
            model.get_true_optimum(), opt_result.x)
        np_test.assert_array_almost_equal(
            model.get_true_optimum(), opt_x)

        hessian, inv_hess_sqrt, hessian_corrected = \
            opt_lib.set_objective_preconditioner(model.objective, init_x)
        np_test.assert_array_almost_equal(model.matrix, hessian)

        opt_x, opt_result = opt_lib.minimize_objective_bfgs(
            model.objective, init_x, precondition=True)
        np_test.assert_array_almost_equal(model.get_true_optimum(), opt_x)

        opt_x, opt_result = opt_lib.minimize_objective_trust_ncg(
            model.objective, init_x, precondition=True)
        np_test.assert_array_almost_equal(model.get_true_optimum(), opt_x)

    def test_repeated_optimization(self):
        model = QuadraticModel(3)
        init_x = np.zeros(model.dim)

        def initial_optimization(x):
            new_x = np.random.random(len(x))
            return new_x, new_x

        def take_gradient_step(x):
            grad = model.objective.fun_free_grad(x)
            return x - 0.5 * grad, x

        opt_x, converged, x_conv, f_conv, grad_conv, obj_opt, opt_results = \
            opt_lib.repeatedly_optimize(
                model.objective,
                take_gradient_step,
                init_x,
                initial_optimization_fun=initial_optimization,
                keep_intermediate_optimizations=True,
                max_iter=1000)

        np_test.assert_array_almost_equal(
            model.get_true_optimum(), opt_x, decimal=4)


class TestMatrixSquareRoot(unittest.TestCase):
    def test_sym_matrix_inv_sqrt(self):
        a_vec = np.array([1, 2, 3])
        b_vec = np.array([0, 1, 3])
        c_vec = np.array([0, 3, 3])
        # a_vec = a_vec /np.linalg.norm(a_vec)
        # b_vec = b_vec /np.linalg.norm(b_vec)
        # c_vec = c_vec / np.linalg.norm(c_vec)
        a = np.outer(a_vec, a_vec) + \
            np.outer(b_vec, b_vec) + \
            np.outer(c_vec, c_vec)

        # Test with no eigenvalue trimming
        a_inv_sqrt, a_corrected = opt_lib.get_sym_matrix_inv_sqrt(a)
        np_test.assert_array_almost_equal(
            np.linalg.inv(a), a_inv_sqrt @ a_inv_sqrt.T)
        np_test.assert_array_almost_equal(a, a_corrected)

        # Check the eigenvalue trimming.
        eig_val, eig_vec = np.linalg.eigh(a)
        min_ev = eig_val[0] + 0.5 * (eig_val[1] - eig_val[0])
        max_ev = eig_val[2] - 0.5 * (eig_val[2] - eig_val[1])

        a_inv_sqrt, a_corrected = opt_lib.get_sym_matrix_inv_sqrt(
            a, ev_min=min_ev)
        eig_val_test, _ = np.linalg.eigh(np.linalg.inv(a_inv_sqrt @ a_inv_sqrt.T))
        np_test.assert_array_almost_equal(min_ev, eig_val_test[0])
        np_test.assert_array_almost_equal(eig_val[1:2], eig_val_test[1:2])

        a_inv_sqrt, a_corrected = opt_lib.get_sym_matrix_inv_sqrt(
            a, ev_max=max_ev)
        eig_val_test, _ = np.linalg.eigh(np.linalg.inv(a_inv_sqrt @ a_inv_sqrt.T))
        np_test.assert_array_almost_equal(max_ev, eig_val_test[2])
        np_test.assert_array_almost_equal(eig_val[0:1], eig_val_test[0:1])

        a_inv_sqrt, a_corrected = opt_lib.get_sym_matrix_inv_sqrt(
            a, ev_min=min_ev, ev_max=max_ev)
        eig_val_test, _ = np.linalg.eigh(np.linalg.inv(a_inv_sqrt @ a_inv_sqrt.T))
        np_test.assert_array_almost_equal(min_ev, eig_val_test[0])
        np_test.assert_array_almost_equal(max_ev, eig_val_test[2])
        np_test.assert_array_almost_equal(eig_val[1], eig_val_test[1])


if __name__ == '__main__':
    unittest.main()
