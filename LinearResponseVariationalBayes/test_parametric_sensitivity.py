# #!/usr/bin/env python3
#
# import autograd
# from autograd import numpy as np
# import scipy as sp
# import numpy.testing as np_test
# import unittest
# import LinearResponseVariationalBayes as vb
# import LinearResponseVariationalBayes.SparseObjectives as obj_lib
# from copy import deepcopy
#
# class QuadraticModel(object):
#     def __init__(self, dim):
#         # Put lower bounds so we're testing the contraining functions.
#         self.dim = dim
#         self.param = vb.VectorParam('theta', size=dim, lb=-10.0)
#
#         # Make a copy of the parameter that we can use to differentiates
#         # the conversion from vector to free parameters without leaving ArrayBox
#         # types around.
#         self.param_copy = deepcopy(self.param)
#
#         self.hyper_param = vb.VectorParam('lambda', size=dim, lb=-2.0)
#         self.hyper_param.set_vector(np.linspace(0.5, 10.0, num=dim))
#
#         vec = np.linspace(0.1, 0.3, num=dim)
#         self.matrix = np.outer(vec, vec) + np.eye(dim)
#
#         self.output_param = vb.VectorParam('theta_sq', size=dim, lb=0.0)
#
#         self.objective = obj_lib.Objective(self.param, self.get_objective)
#
#     def get_hyper_par_objective(self):
#         # Only the part of the objective that dependson the hyperparameters.
#         theta = self.param.get()
#         return self.hyper_param.get() @ theta
#
#     def get_objective(self):
#         theta = self.param.get()
#         objective = 0.5 * theta.T @ self.matrix @ theta
#         shift = self.get_hyper_par_objective()
#         return objective + shift
#
#     def get_output_param_vector(self):
#         theta = self.param.get_vector()
#         self.output_param.set_vector(theta ** 2)
#         return self.output_param.get_vector()
#
#     # Testing functions that use the fact that the optimum has a closed form.
#     def get_true_optimum_theta(self, hyper_param_val):
#         theta = -1 * np.linalg.solve(self.matrix, hyper_param_val)
#         return theta
#
#     def get_true_optimum(self, hyper_param_val):
#         # ...in the free parameterization.
#         theta = self.get_true_optimum_theta(hyper_param_val)
#         self.param_copy.set_vector(theta)
#         return self.param_copy.get_free()
#
#     def get_true_output_param_vector(self, hyper_param_val):
#         theta = self.get_true_optimum_theta(hyper_param_val)
#         return theta ** 2
#
#
# class TestParametricSensitivity(unittest.TestCase):
#     def test_quadratic_model(self):
#         model = QuadraticModel(3)
#
#         opt_output = sp.optimize.minimize(
#             fun=model.objective.fun_free,
#             jac=model.objective.fun_free_grad,
#             x0=np.zeros(model.dim),
#             method='BFGS')
#
#         hyper_param_val = model.hyper_param.get_vector()
#         theta0 = model.get_true_optimum(hyper_param_val)
#         np_test.assert_array_almost_equal(theta0, opt_output.x)
#         model.param.set_free(theta0)
#         output_par0 = model.get_output_param_vector()
#
#         np_test.assert_array_almost_equal(
#             output_par0,
#             model.param.get_vector() ** 2)
#
#         parametric_sens = obj_lib.ParametricSensitivity(
#             objective_fun=model.get_objective,
#             input_par=model.param,
#             output_par=model.output_param,
#             hyper_par=model.hyper_param,
#             input_to_output_converter=model.get_output_param_vector)
#
#         epsilon = 0.01
#         new_hyper_param_val = hyper_param_val + epsilon
#
#         # Check the optimal parameters
#         pred_diff = \
#             parametric_sens.predict_input_par_from_hyperparameters(
#                 new_hyper_param_val) - \
#             theta0
#         true_diff = model.get_true_optimum(new_hyper_param_val) - theta0
#         self.assertTrue(
#             np.linalg.norm(true_diff - pred_diff) <= \
#             epsilon * np.linalg.norm(true_diff))
#
#         # Check the output parameters
#         pred_diff = \
#             parametric_sens.predict_output_par_from_hyperparameters(
#                 new_hyper_param_val, linear=False) - \
#             output_par0
#         true_diff = \
#             model.get_true_output_param_vector(new_hyper_param_val) - \
#             output_par0
#         self.assertTrue(
#             np.linalg.norm(true_diff - pred_diff) <= \
#             epsilon * np.linalg.norm(true_diff))
#
#         pred_diff = \
#             parametric_sens.predict_output_par_from_hyperparameters(
#                 new_hyper_param_val, linear=True) - \
#             output_par0
#         true_diff = \
#             model.get_true_output_param_vector(new_hyper_param_val) - output_par0
#         self.assertTrue(
#             np.linalg.norm(true_diff - pred_diff) <= \
#             epsilon * np.linalg.norm(true_diff))
#
#         # Check the derivatives
#         get_dinput_dhyper = autograd.jacobian(
#             model.get_true_optimum)
#         get_doutput_dhyper = autograd.jacobian(
#             model.get_true_output_param_vector)
#
#         np_test.assert_array_almost_equal(
#             get_dinput_dhyper(hyper_param_val),
#             parametric_sens.get_dinput_dhyper())
#
#         np_test.assert_array_almost_equal(
#             get_doutput_dhyper(hyper_param_val),
#             parametric_sens.get_doutput_dhyper())
#
#         # Check that the sensitivity works when specifying
#         # hyper_par_objective_fun.
#         # I think it suffices to just check the derivatives.
#         model.param.set_free(theta0)
#         model.hyper_param.set_vector(hyper_param_val)
#         parametric_sens2 = obj_lib.ParametricSensitivity(
#             objective_fun=model.get_objective,
#             input_par=model.param,
#             output_par=model.output_param,
#             hyper_par=model.hyper_param,
#             optimal_input_par=theta0,
#             input_to_output_converter=model.get_output_param_vector,
#             hyper_par_objective_fun=model.get_hyper_par_objective)
#
#         np_test.assert_array_almost_equal(
#             get_dinput_dhyper(hyper_param_val),
#             parametric_sens2.get_dinput_dhyper())
#
#         np_test.assert_array_almost_equal(
#             get_doutput_dhyper(hyper_param_val),
#             parametric_sens2.get_doutput_dhyper())
#
#
# if __name__ == '__main__':
#     unittest.main()
