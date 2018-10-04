#!/usr/bin/env python3

import autograd
import autograd.numpy as np

import LinearResponseVariationalBayes as vb
import LinearResponseVariationalBayes.SparseObjectives as obj_lib
import LinearResponseVariationalBayes.ModelSensitivity as sens_lib

import numpy.testing as np_test
from numpy.testing import assert_array_almost_equal

from copy import deepcopy

import math

from LinearResponseVariationalBayes.ModelSensitivity import append_jvp
from LinearResponseVariationalBayes.ModelSensitivity import DerivativeTerm
from LinearResponseVariationalBayes.ModelSensitivity import generate_two_term_derivative_array

from LinearResponseVariationalBayes.ModelSensitivity import consolidate_terms
from LinearResponseVariationalBayes.ModelSensitivity import evaluate_terms

from LinearResponseVariationalBayes.ModelSensitivity import get_taylor_base_terms
from LinearResponseVariationalBayes.ModelSensitivity import evaluate_dketa_depsk

from LinearResponseVariationalBayes.ModelSensitivity import differentiate_terms

from LinearResponseVariationalBayes.ModelSensitivity import ParametricSensitivityTaylorExpansion

import scipy as sp

import unittest

# This class will be used for testing.
class QuadraticModel(object):
    def __init__(self, dim):
        # Put lower bounds so we're testing the contraining functions
        # and so that derivatives of all orders are nonzero.
        self.dim = dim
        self.param = vb.VectorParam('theta', size=dim, lb=-10.0)

        # Make a copy of the parameter that we can use to differentiates
        # the conversion from vector to free parameters without leaving ArrayBox
        # types around.
        self.param_copy = deepcopy(self.param)

        self.hyper_param = vb.VectorParam('lambda', size=dim, lb=-2.0)
        self.hyper_param.set_vector(np.linspace(0.5, 10.0, num=dim))

        vec = np.linspace(0.1, 0.3, num=dim)
        self.matrix = np.outer(vec, vec) + np.eye(dim)

        self.output_param = vb.VectorParam('theta_sq', size=dim, lb=0.0)

        self.objective = obj_lib.Objective(self.param, self.get_objective)

    def get_hyper_par_objective(self):
        # Only the part of the objective that dependson the hyperparameters.
        theta = self.param.get()
        return self.hyper_param.get() @ theta

    def get_objective(self):
        theta = self.param.get()
        objective = 0.5 * theta.T @ self.matrix @ theta
        shift = self.get_hyper_par_objective()
        return objective + shift

    # Testing functions that use the fact that the optimum has a closed form.
    def get_true_optimum_theta(self, hyper_param_val):
        theta = -1 * np.linalg.solve(self.matrix, hyper_param_val)
        return theta

    def get_true_optimum(self, hyper_param_val):
        # ...in the free parameterization.
        theta = self.get_true_optimum_theta(hyper_param_val)
        self.param_copy.set_vector(theta)
        return self.param_copy.get_free()

    # For approximating the value of the output parameter:
    def get_output_param_vector(self):
        theta = self.param.get_vector()
        self.output_param.set_vector(theta ** 2)
        return self.output_param.get_vector()

    def get_true_output_param_vector(self, hyper_param_val):
        theta = self.get_true_optimum_theta(hyper_param_val)
        return theta ** 2


class TestTaylorExpansion(unittest.TestCase):
    def test_everything(self):
        # TODO: split some of these out into standalone tests.

        #################################
        # Set up the ground truth.

        def wrap_objective_2par(fun):
            def wrapped_fun(eta, eps, *argv):
                result = fun(eta, eps, *argv)
                sensitivity_objective.par1.set_free(eta)
                sensitivity_objective.par2.set_vector(eps)
                return result
            return wrapped_fun

        model = QuadraticModel(3)
        eps0 = model.hyper_param.get_vector()
        eta0 = model.get_true_optimum(eps0)
        hess0 = model.objective.fun_free_hessian(eta0)

        eps1 = eps0 + 1e-1
        eta1 = model.get_true_optimum(eps1)

        # The TwoParameterObjective class computes cross Hessians and
        # so will be useful for testing more generic JVPs.
        sensitivity_objective = obj_lib.TwoParameterObjective(
            model.param, model.hyper_param, model.get_objective)

        # Get the exact derivatives using the closed-form optimum.
        true_deta_deps = autograd.jacobian(model.get_true_optimum)
        true_d2eta_deps2 = autograd.jacobian(true_deta_deps)
        true_d3eta_deps3 = autograd.jacobian(true_d2eta_deps2)
        true_d4eta_deps4 = autograd.jacobian(true_d3eta_deps3)

        # Sanity check using standard first-order approximation.
        d2f_deta_deps = \
            sensitivity_objective.fun_hessian_free1_vector2(eta0, eps0)
        assert_array_almost_equal(
            true_deta_deps(eps0),
            -1 * np.linalg.solve(hess0, d2f_deta_deps))


        ########################
        # Test append_jvp.

        def objective_2par(eta, eps):
            return sensitivity_objective.eval_fun(
                eta, eps, val1_is_free=True, val2_is_free=False)

        dobj_deta = wrap_objective_2par(
            append_jvp(objective_2par, num_base_args=2, argnum=0))
        d2obj_deta_deta = wrap_objective_2par(
            append_jvp(dobj_deta, num_base_args=2, argnum=0))

        v1 = np.random.random(len(eta0))
        v2 = np.random.random(len(eta0))
        v3 = np.random.random(len(eta0))
        w1 = np.random.random(len(eps0))
        w2 = np.random.random(len(eps0))
        w3 = np.random.random(len(eps0))

        # Check the first argument
        assert_array_almost_equal(
            np.einsum('i,i', model.objective.fun_free_grad(eta0), v1),
            dobj_deta(eta0, eps0, v1))
        assert_array_almost_equal(
            np.einsum('ij,i,j', model.objective.fun_free_hessian(eta0), v1, v2),
            d2obj_deta_deta(eta0, eps0, v1, v2))

        # Check the second argument
        hyperparam_obj = obj_lib.Objective(model.hyper_param, model.get_objective)

        dobj_deps = wrap_objective_2par(append_jvp(objective_2par, num_base_args=2, argnum=1))
        d2obj_deps_deps = wrap_objective_2par(append_jvp(dobj_deps, num_base_args=2, argnum=1))

        assert_array_almost_equal(
            np.einsum('i,i', hyperparam_obj.fun_vector_grad(eps0), w1),
            dobj_deps(eta0, eps0, w1))

        assert_array_almost_equal(
            np.einsum('ij,i,j', hyperparam_obj.fun_vector_hessian(eps0), w1, w2),
            d2obj_deps_deps(eta0, eps0, w1, w2))

        # Check mixed arguments
        d2obj_deps_deta = wrap_objective_2par(append_jvp(dobj_deps, num_base_args=2, argnum=0))
        d2obj_deta_deps = wrap_objective_2par(append_jvp(dobj_deta, num_base_args=2, argnum=1))

        assert_array_almost_equal(
            d2obj_deps_deta(eta0, eps0, v1, w1),
            d2obj_deta_deps(eta0, eps0, w1, v1))

        assert_array_almost_equal(
            np.einsum('ij,i,j',
                      sensitivity_objective.fun_hessian_free1_vector2(eta0, eps0), v1, w1),
            d2obj_deps_deta(eta0, eps0, v1, w1))

        # Check derivatives of vectors.
        @wrap_objective_2par
        def grad_obj(eta, eps):
            return sensitivity_objective.fun_grad1(
                eta, eps, val1_is_free=True, val2_is_free=False)

        grad_obj(eta0, eps0)

        dg_deta = wrap_objective_2par(append_jvp(grad_obj, num_base_args=2, argnum=0))

        assert_array_almost_equal(
            hess0 @ v1, dg_deta(eta0, eps0, v1))

        ########################
        # Test derivative terms.

        # Again, first some ground truth.
        def eval_deta_deps(eta, eps, v1):
            assert np.max(np.sum(eps - eps0)) < 1e-8
            assert np.max(np.sum(eta - eta0)) < 1e-8
            return -1 * np.linalg.solve(hess0, d2f_deta_deps @ v1)

        dg_deta = wrap_objective_2par(append_jvp(grad_obj, num_base_args=2, argnum=0))
        dg_deps = wrap_objective_2par(append_jvp(grad_obj, num_base_args=2, argnum=1))

        d2g_deta_deta = wrap_objective_2par(append_jvp(dg_deta, num_base_args=2, argnum=0))
        d2g_deta_deps = wrap_objective_2par(append_jvp(dg_deta, num_base_args=2, argnum=1))
        d2g_deps_deta = wrap_objective_2par(append_jvp(dg_deps, num_base_args=2, argnum=0))
        d2g_deps_deps = wrap_objective_2par(append_jvp(dg_deps, num_base_args=2, argnum=1))

        # This is a manual version of the second derivative.
        def eval_d2eta_deps2(eta, eps, delta_eps):
            assert np.max(np.sum(eps - eps0)) < 1e-8
            assert np.max(np.sum(eta - eta0)) < 1e-8

            deta_deps = -1 * np.linalg.solve(hess0, dg_deps(eta, eps, delta_eps))

            # Then the terms in the second derivative.
            d2_terms = \
                d2g_deps_deps(eta, eps, delta_eps, delta_eps) + \
                d2g_deps_deta(eta, eps, delta_eps, deta_deps) + \
                d2g_deta_deps(eta, eps, deta_deps, delta_eps) + \
                d2g_deta_deta(eta, eps, deta_deps, deta_deps)
            d2eta_deps2 = -1 * np.linalg.solve(hess0, d2_terms)
            return d2eta_deps2


        eval_g_derivs = generate_two_term_derivative_array(grad_obj, order=5)

        assert_array_almost_equal(
            hess0 @ v1,
            eval_g_derivs[1][0](eta0, eps0, v1))

        d2g_deta_deta(eta0, eps0, v1, v2)
        eval_g_derivs[2][0](eta0, eps0, v1, v2)

        assert_array_almost_equal(
            d2g_deta_deta(eta0, eps0, v1, v2),
            eval_g_derivs[2][0](eta0, eps0, v1, v2))

        assert_array_almost_equal(
            d2g_deta_deps(eta0, eps0, v1, v2),
            eval_g_derivs[1][1](eta0, eps0, v1, v2))

        dterm = DerivativeTerm(
            eps_order=1,
            eta_orders=[1, 0],
            prefactor=1.5,
            eval_eta_derivs=[ eval_deta_deps ],
            eval_g_derivs=eval_g_derivs)

        deps = eps1 - eps0

        assert_array_almost_equal(
            dterm.prefactor * d2g_deta_deps(
                eta0, eps0, eval_deta_deps(eta0, eps0, deps), deps),
            dterm.evaluate(eta0, eps0, deps))

        dterms = [
            DerivativeTerm(
                eps_order=2,
                eta_orders=[0, 0],
                prefactor=1.5,
                eval_eta_derivs=[ eval_deta_deps ],
                eval_g_derivs=eval_g_derivs),
            DerivativeTerm(
                eps_order=1,
                eta_orders=[1, 0],
                prefactor=2,
                eval_eta_derivs=[ eval_deta_deps ],
                eval_g_derivs=eval_g_derivs),
            DerivativeTerm(
                eps_order=1,
                eta_orders=[1, 0],
                prefactor=3,
                eval_eta_derivs=[ eval_deta_deps ],
                eval_g_derivs=eval_g_derivs) ]


        dterms_combined = consolidate_terms(dterms)
        assert len(dterms) == 3
        assert len(dterms_combined) == 2

        assert_array_almost_equal(
            evaluate_terms(dterms, eta0, eps0, deps),
            evaluate_terms(dterms_combined, eta0, eps0, deps))

        dterms1 = get_taylor_base_terms(eval_g_derivs)

        assert_array_almost_equal(
            dg_deps(eta0, eps0, deps),
            dterms1[0].evaluate(eta0, eps0, deps))

        assert_array_almost_equal(
            np.einsum('ij,j', true_deta_deps(eps0), deps),
            evaluate_dketa_depsk(hess0, dterms1, eta0, eps0, deps))

        assert_array_almost_equal(
            eval_deta_deps(eta0, eps0, deps),
            evaluate_dketa_depsk(hess0, dterms1, eta0, eps0, deps))

        dterms2 = differentiate_terms(hess0, dterms1)
        assert np.linalg.norm(evaluate_dketa_depsk(hess0, dterms2, eta0, eps0, deps)) > 0
        assert_array_almost_equal(
            np.einsum('ijk,j, k', true_d2eta_deps2(eps0), deps, deps),
            evaluate_dketa_depsk(hess0, dterms2, eta0, eps0, deps))

        dterms3 = differentiate_terms(hess0, dterms2)
        assert np.linalg.norm(evaluate_dketa_depsk(hess0, dterms3, eta0, eps0, deps)) > 0

        assert_array_almost_equal(
            np.einsum('ijkl,j,k,l', true_d3eta_deps3(eps0), deps, deps, deps),
            evaluate_dketa_depsk(hess0, dterms3, eta0, eps0, deps))

        ###################################
        # Test the Taylor series itself.

        taylor_expansion = \
            ParametricSensitivityTaylorExpansion(
                objective_functor=model.get_objective,
                input_par=model.param,
                hyper_par=model.hyper_param,
                input_val0=eta0,
                hyper_val0=eps0,
                order=3,
                input_is_free=True,
                hyper_is_free=False,
                hess0=hess0)

        taylor_expansion.print_terms(k=3)

        d1 = np.einsum('ij,j', true_deta_deps(eps0), deps)
        d2 = np.einsum('ijk,j,k', true_d2eta_deps2(eps0), deps, deps)
        d3 = np.einsum('ijkl,j,k,l', true_d3eta_deps3(eps0), deps, deps, deps)

        assert_array_almost_equal(
            d1, taylor_expansion.evaluate_dkinput_dhyperk(deps, k=1))

        assert_array_almost_equal(
            d2, taylor_expansion.evaluate_dkinput_dhyperk(deps, k=2))

        assert_array_almost_equal(
            d3, taylor_expansion.evaluate_dkinput_dhyperk(deps, k=3))

        assert_array_almost_equal(
            eta0 + d1, taylor_expansion.evaluate_taylor_series(deps, max_order=1))

        assert_array_almost_equal(
            eta0 + d1 + 0.5 * d2,
            taylor_expansion.evaluate_taylor_series(deps, max_order=2))

        assert_array_almost_equal(
            eta0 + d1 + d2 / 2 + d3 / 6,
            taylor_expansion.evaluate_taylor_series(deps, max_order=3))

        assert_array_almost_equal(
            eta0 + d1 + d2 / 2 + d3 / 6,
            taylor_expansion.evaluate_taylor_series(deps))


class ParametricSensitivityLinearApproximation(unittest.TestCase):
    def test_quadratic_model(self):
        model = QuadraticModel(3)

        opt_output = sp.optimize.minimize(
            fun=model.objective.fun_free,
            jac=model.objective.fun_free_grad,
            x0=np.zeros(model.dim),
            method='BFGS')

        hyper_param_val = model.hyper_param.get_vector()
        theta0 = model.get_true_optimum(hyper_param_val)
        np_test.assert_array_almost_equal(theta0, opt_output.x)
        model.param.set_free(theta0)

        parametric_sens = sens_lib.ParametricSensitivityLinearApproximation(
            objective_functor=model.get_objective,
            input_par=model.param,
            hyper_par=model.hyper_param,
            input_val0=theta0,
            hyper_val0=hyper_param_val)

        epsilon = 0.01
        new_hyper_param_val = hyper_param_val + epsilon

        # Check the optimal parameters
        pred_diff = \
            parametric_sens.predict_input_par_from_hyperparameters(
                new_hyper_param_val) - \
            theta0
        true_diff = model.get_true_optimum(new_hyper_param_val) - theta0
        self.assertTrue(
            np.linalg.norm(true_diff - pred_diff) <= \
            epsilon * np.linalg.norm(true_diff))

        # Check the Jacobian.
        get_dinput_dhyper = autograd.jacobian(
            model.get_true_optimum)
        np_test.assert_array_almost_equal(
            get_dinput_dhyper(hyper_param_val),
            parametric_sens.get_dinput_dhyper())

        # Check that the sensitivity works when specifying
        # hyper_par_objective_fun.
        # I think it suffices to just check the derivatives.
        model.param.set_free(theta0)
        model.hyper_param.set_vector(hyper_param_val)
        parametric_sens2 = sens_lib.ParametricSensitivityLinearApproximation(
            objective_functor=model.get_objective,
            input_par=model.param,
            hyper_par=model.hyper_param,
            input_val0=theta0,
            hyper_val0=hyper_param_val,
            hyper_par_objective_functor=model.get_hyper_par_objective)

        np_test.assert_array_almost_equal(
            get_dinput_dhyper(hyper_param_val),
            parametric_sens2.get_dinput_dhyper())


class TestParametricSensitivity(unittest.TestCase):
    # Note: this class and test should be deprecated in favor of
    # ParametricSensitivityLinearApproximation.
    def test_quadratic_model(self):
        model = QuadraticModel(3)

        opt_output = sp.optimize.minimize(
            fun=model.objective.fun_free,
            jac=model.objective.fun_free_grad,
            x0=np.zeros(model.dim),
            method='BFGS')

        hyper_param_val = model.hyper_param.get_vector()
        theta0 = model.get_true_optimum(hyper_param_val)
        np_test.assert_array_almost_equal(theta0, opt_output.x)
        model.param.set_free(theta0)
        output_par0 = model.get_output_param_vector()

        np_test.assert_array_almost_equal(
            output_par0,
            model.param.get_vector() ** 2)

        parametric_sens = obj_lib.ParametricSensitivity(
            objective_fun=model.get_objective,
            input_par=model.param,
            output_par=model.output_param,
            hyper_par=model.hyper_param,
            input_to_output_converter=model.get_output_param_vector)

        epsilon = 0.01
        new_hyper_param_val = hyper_param_val + epsilon

        # Check the optimal parameters
        pred_diff = \
            parametric_sens.predict_input_par_from_hyperparameters(
                new_hyper_param_val) - \
            theta0
        true_diff = model.get_true_optimum(new_hyper_param_val) - theta0
        self.assertTrue(
            np.linalg.norm(true_diff - pred_diff) <= \
            epsilon * np.linalg.norm(true_diff))

        # Check the output parameters
        pred_diff = \
            parametric_sens.predict_output_par_from_hyperparameters(
                new_hyper_param_val, linear=False) - \
            output_par0
        true_diff = \
            model.get_true_output_param_vector(new_hyper_param_val) - \
            output_par0
        self.assertTrue(
            np.linalg.norm(true_diff - pred_diff) <= \
            epsilon * np.linalg.norm(true_diff))

        pred_diff = \
            parametric_sens.predict_output_par_from_hyperparameters(
                new_hyper_param_val, linear=True) - \
            output_par0
        true_diff = \
            model.get_true_output_param_vector(new_hyper_param_val) - output_par0
        self.assertTrue(
            np.linalg.norm(true_diff - pred_diff) <= \
            epsilon * np.linalg.norm(true_diff))

        # Check the derivatives
        get_dinput_dhyper = autograd.jacobian(
            model.get_true_optimum)
        get_doutput_dhyper = autograd.jacobian(
            model.get_true_output_param_vector)

        np_test.assert_array_almost_equal(
            get_dinput_dhyper(hyper_param_val),
            parametric_sens.get_dinput_dhyper())

        np_test.assert_array_almost_equal(
            get_doutput_dhyper(hyper_param_val),
            parametric_sens.get_doutput_dhyper())

        # Check that the sensitivity works when specifying
        # hyper_par_objective_fun.
        # I think it suffices to just check the derivatives.
        model.param.set_free(theta0)
        model.hyper_param.set_vector(hyper_param_val)
        parametric_sens2 = obj_lib.ParametricSensitivity(
            objective_fun=model.get_objective,
            input_par=model.param,
            output_par=model.output_param,
            hyper_par=model.hyper_param,
            optimal_input_par=theta0,
            input_to_output_converter=model.get_output_param_vector,
            hyper_par_objective_fun=model.get_hyper_par_objective)

        np_test.assert_array_almost_equal(
            get_dinput_dhyper(hyper_param_val),
            parametric_sens2.get_dinput_dhyper())

        np_test.assert_array_almost_equal(
            get_doutput_dhyper(hyper_param_val),
            parametric_sens2.get_doutput_dhyper())


if __name__ == '__main__':
    unittest.main()
