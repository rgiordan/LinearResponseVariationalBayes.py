#!/usr/bin/python3
import VariationalBayes as vb
import LogisticGLMM_lib as logit_glmm
import VariationalBayes.SparseObjectives as obj_lib

import autograd
import unittest
import numpy.testing as np_test
import numpy as np


def simulate_data(N, K, NG):
    np.random.seed(42)
    true_beta = np.array(range(K))
    true_beta = true_beta - np.mean(true_beta)
    true_mu = 0.
    true_tau = 40.0

    x_mat, y_g_vec, y_vec, true_rho, true_u = \
        logit_glmm.simulate_data(N, NG, true_beta, true_mu, true_tau)

    return x_mat, y_g_vec, y_vec

class TestModel(unittest.TestCase):

    # For every parameter type, execute all the required methods.
    def test_create_model(self):
        N = 50
        K = 2
        NG = 7

        x_mat, y_g_vec, y_vec = simulate_data(N, K, NG)
        prior_par = logit_glmm.get_default_prior_params(K)
        glmm_par = logit_glmm.get_glmm_parameters(K=K, NG=NG)

        model = logit_glmm.LogisticGLMM(
            glmm_par, prior_par, x_mat, y_vec, y_g_vec, num_gh_points=4)

        objective = obj_lib.Objective(model.glmm_par, model.get_kl)
        free_par = np.random.random(model.glmm_par.free_size())
        model.glmm_par.set_free(free_par)
        objective.fun_free(free_par)
        obj_grad = objective.fun_free_grad(free_par)
        obj_hess = objective.fun_free_hessian(free_par)
        obj_hvp = objective.fun_free_hvp(free_par, obj_grad)

        np_test.assert_array_almost_equal(
            np.matmul(obj_hess, obj_grad), obj_hvp,
            err_msg='Hessian vector product equals Hessian')

        moment_wrapper = logit_glmm.MomentWrapper(glmm_par)
        moment_wrapper.get_moment_vector_from_free(free_par)

    def test_sparse_model(self):
        N = 17
        K = 2
        NG = 7

        x_mat, y_g_vec, y_vec = simulate_data(N, K, NG)
        prior_par = logit_glmm.get_default_prior_params(K)
        glmm_par = logit_glmm.get_glmm_parameters(K=K, NG=NG)

        model = logit_glmm.LogisticGLMM(
            glmm_par, prior_par, x_mat, y_vec, y_g_vec, num_gh_points=4)
        moment_wrapper = logit_glmm.MomentWrapper(glmm_par)
        objective = obj_lib.Objective(model.glmm_par, model.get_kl)

        free_par = np.random.random(model.glmm_par.free_size())

        model.glmm_par.set_free(free_par)
        group_model = logit_glmm.SubGroupsModel(model)
        global_model = logit_glmm.GlobalModel(model)
        # sparse_model = logit_glmm.SparseModelObjective(
        #     glmm_par, prior_par, x_mat, y_vec, y_g_vec, num_gh_points=4,
        #     num_groups=1)

        # global_model.glmm_par.set_free(free_par)
        # group_model.glmm_par.set_free(free_par)

        ################################
        # Test get_data_for_groups

        g = 2
        group_rows, y_g_select = group_model.get_data_for_groups([g])
        np_test.assert_array_almost_equal(
            x_mat[group_rows, :], x_mat[y_g_vec == g, :])
        np_test.assert_array_almost_equal(
            y_vec[group_rows], y_vec[y_g_vec == g])
        self.assertEqual(len(y_g_select), sum(group_rows))

        ########################
        # Test the objectives.

        # Testing the sparse objective.
        g = 2
        global_model.set_global_parameters()
        group_model.set_group_parameters([g])
        single_group_model = logit_glmm.LogisticGLMM(
            group_model.group_par, prior_par,
            x_mat[group_rows, :], y_vec[group_rows], y_g_select,
            num_gh_points = model.num_gh_points)
        np_test.assert_array_almost_equal(
            single_group_model.glmm_par.get_vector(),
            group_model.group_par.get_vector(),
            err_msg='Group model parameter equality')

        # Checking that a single group is equal.
        single_group_kl = single_group_model.get_kl()
        sparse_kl = \
            global_model.get_global_kl() + \
            group_model.get_group_kl()

        np_test.assert_array_almost_equal(
            single_group_kl, sparse_kl,
            err_msg="Group model kl equality")

        # Checking the full kl is equal.
        sparse_kl = global_model.get_global_kl()
        for g in range(NG):
            group_model.set_group_parameters([g])
            sparse_kl += group_model.get_group_kl()

        np_test.assert_array_almost_equal(
            objective.fun_free(free_par), sparse_kl)

        # Check that the vector Hessian is equal.
        model.glmm_par.set_free(free_par)
        global_model.glmm_par.set_free(free_par)
        group_model.glmm_par.set_free(free_par)

        sparse_vector_hess = \
            group_model.get_sparse_kl_vec_hessian(print_every_n=-1) + \
            global_model.get_sparse_kl_vec_hessian()
        sparse_vector_hess_dense = \
            np.array(sparse_vector_hess.todense())
        full_vector_hess = \
            objective.fun_vector_hessian(model.glmm_par.get_vector())

        np_test.assert_array_almost_equal(
            full_vector_hess,
            sparse_vector_hess_dense,
            err_msg='Sparse vector Hessian equality')

        # Check that the free Hessian is equal.
        #sparse_hess = sparse_model.get_free_hessian(sparse_vector_hess)
        sparse_hess = glmm_lib.get_free_hessian(
            model, group_model=group_model, global_model=global_model)
        full_hess = objective.fun_free_hessian(
            model.glmm_par.get_free())

        np_test.assert_array_almost_equal(
            full_hess,
            np.asarray(sparse_hess.todense()),
            err_msg='Sparse free Hessian equality')

        # Check that the weight jacobian is equal.
        sparse_jac = group_model.get_sparse_weight_vector_jacobian()
        def get_data_terms_vec(glmm_par_vec):
            model.glmm_par.set_vector(glmm_par_vec)
            return model.get_data_log_lik_terms()
        def get_data_terms_free(glmm_par_free):
            model.glmm_par.set_free(glmm_par_free)
            return model.get_data_log_lik_terms()

        get_full_jac = autograd.jacobian(get_data_terms_vec)
        full_jac = get_full_jac(model.glmm_par.get_vector())

        self.assertEqual(full_jac.shape, sparse_jac.shape)
        np_test.assert_array_almost_equal(
            full_jac,
            np.asarray(sparse_jac.todense()),
            err_msg='Sparse vector Jacobian equality')

        sparse_free_jac = \
            glmm_lib.get_sparse_weight_free_jacobian(group_model)
        get_full_free_jac = autograd.jacobian(get_data_terms_free)
        full_free_jac = get_full_free_jac(model.glmm_par.get_free())

        self.assertEqual(full_free_jac.shape, sparse_free_jac.shape)
        np_test.assert_array_almost_equal(
            full_free_jac,
            np.asarray(sparse_free_jac.todense()),
            err_msg='Sparse free Jacobian equality')




if __name__ == '__main__':
    unittest.main()
