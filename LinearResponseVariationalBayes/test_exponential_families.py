#!/usr/bin/env python3

import autograd.numpy as np
from autograd import grad, jacobian, hessian
from autograd.util import quick_grad_check
import LinearResponseVariationalBayes.ExponentialFamilies as ef
import LinearResponseVariationalBayes.Modeling as model

import unittest
import numpy.testing as np_test
import scipy as sp

from numpy.polynomial.hermite import hermgauss


class TestEntropy(unittest.TestCase):
    def test_uvn_entropy(self):
        mean_par = 2.0
        info_par = 1.5
        num_draws = 10000
        norm_dist = sp.stats.norm(loc=mean_par, scale=np.sqrt(1 / info_par))
        self.assertAlmostEqual(
            norm_dist.entropy(), ef.univariate_normal_entropy(info_par))

    def test_mvn_entropy(self):
        mean_par = np.array([1., 2.])
        info_par = np.eye(2) + np.full((2, 2), 0.1)
        norm_dist = sp.stats.multivariate_normal(
            mean=mean_par, cov=np.linalg.inv(info_par))
        self.assertAlmostEqual(
            norm_dist.entropy(), ef.multivariate_normal_entropy(info_par))

    def test_gamma_entropy(self):
        shape = 3.0
        rate = 2.4
        gamma_dist = sp.stats.gamma(a=shape, scale=1 / rate)
        self.assertAlmostEqual(gamma_dist.entropy(),
                               ef.gamma_entropy(shape, rate))

    def test_dirichlet_entropy(self):
        alpha = np.array([23, 4, 5, 6, 7])
        dirichlet_dist = sp.stats.dirichlet(alpha)
        self.assertAlmostEqual\
                (dirichlet_dist.entropy(), ef.dirichlet_entropy(alpha))

        alpha_shape = (5, 2)
        alpha = 10 * np.random.random(alpha_shape)
        ef_entropy = ef.dirichlet_entropy(alpha)
        dirichlet_entropy = \
            [ sp.stats.dirichlet.entropy(alpha[:, k]) for k in range(2) ]
        np_test.assert_array_almost_equal(dirichlet_entropy, ef_entropy)

    def test_wishart_entropy(self):
        df = 4.3
        v = np.eye(2) + np.full((2, 2), 0.1)
        wishart_dist = sp.stats.wishart(df=df, scale=v)
        self.assertAlmostEqual(
            wishart_dist.entropy(), ef.wishart_entropy(df, v))

    def test_beta_entropy(self):
        tau = np.array([[1,2], [3,4], [5,6]])
        test_entropy = np.sum([sp.stats.beta.entropy(tau[i, 0], tau[i, 1])
            for i in range(np.shape(tau)[0])])
        self.assertAlmostEqual(ef.beta_entropy(tau), test_entropy)

class TestMoments(unittest.TestCase):
    def test_wishart_moments(self):
        num_draws = 10000
        df = 4.3
        v = np.diag(np.array([2., 3.])) + np.full((2, 2), 0.1)
        wishart_dist = sp.stats.wishart(df=df, scale=v)
        wishart_draws = wishart_dist.rvs(num_draws)
        log_det_draws = np.linalg.slogdet(wishart_draws)[1]
        moment_tolerance = 3.0 * np.std(log_det_draws) / np.sqrt(num_draws)
        print('Wishart e log det test tolerance: ', moment_tolerance)
        np_test.assert_allclose(
            np.mean(log_det_draws), ef.e_log_det_wishart(df, v),
            atol=moment_tolerance)

        # Test the log inverse diagonals
        wishart_inv_draws = \
            [ np.linalg.inv(wishart_draws[n, :, :]) for n in range(num_draws) ]
        wishart_log_diag = \
            np.log([ np.diag(mat) for mat in wishart_inv_draws ])
        diag_mean = np.mean(wishart_log_diag, axis=0)
        diag_sd = np.std(wishart_log_diag, axis=0)
        moment_tolerance = 3.0 * np.max(diag_sd) / np.sqrt(num_draws)
        print('Wishart e log diag test tolerance: ', moment_tolerance)
        np_test.assert_allclose(
            diag_mean, ef.e_log_inv_wishart_diag(df, v),
            atol=moment_tolerance)

        # Test the LKJ prior
        lkj_param = 5.5
        def get_r_matrix(mat):
            mat_diag = np.diag(1. / np.sqrt(np.diag(mat)))
            return np.matmul(mat_diag, np.matmul(mat, mat_diag))

        wishart_log_det_r_draws = \
            np.array([ np.linalg.slogdet(get_r_matrix(mat))[1] \
                       for mat in wishart_inv_draws ]) * (lkj_param - 1)

        moment_tolerance = \
            3.0 * np.std(wishart_log_det_r_draws) / np.sqrt(num_draws)
        print('Wishart lkj prior test tolerance: ', moment_tolerance)
        np_test.assert_allclose(
            np.mean(wishart_log_det_r_draws),
            ef.expected_ljk_prior(lkj_param, df, v),
            atol=moment_tolerance)

    def test_logitnormal_moments(self):
        # global parameters for computing lognormals
        gh_loc, gh_weights = hermgauss(4)

        # log normal parameters
        lognorm_means = np.random.random((5, 3)) # should work for arrays now
        lognorm_infos = np.random.random((5, 3))**2 + 1
        alpha = 2 # dp parameter

        # draw samples
        num_draws = 10**5
        samples = np.random.normal(lognorm_means,
                        1/np.sqrt(lognorm_infos), size = (num_draws, 5, 3))
        logit_norm_samples = sp.special.expit(samples)

        # test lognormal means
        np_test.assert_allclose(
            np.mean(logit_norm_samples, axis = 0),
            ef.get_e_logitnormal(
                lognorm_means, lognorm_infos, gh_loc, gh_weights),
            atol = 3 * np.std(logit_norm_samples) / np.sqrt(num_draws))

        # test Elog(x) and Elog(1-x)
        log_logistic_norm = np.mean(np.log(logit_norm_samples), axis = 0)
        log_1m_logistic_norm = np.mean(np.log(1 - logit_norm_samples), axis = 0)

        tol1 = 3 * np.std(np.log(logit_norm_samples))/ np.sqrt(num_draws)
        tol2 = 3 * np.std(np.log(1 - logit_norm_samples))/ np.sqrt(num_draws)

        np_test.assert_allclose(
            log_logistic_norm,
            ef.get_e_log_logitnormal(
                lognorm_means, lognorm_infos, gh_loc, gh_weights)[0],
            atol = tol1)

        np_test.assert_allclose(
            log_1m_logistic_norm,
            ef.get_e_log_logitnormal(
                        lognorm_means, lognorm_infos, gh_loc, gh_weights)[1],
            atol = tol2)

        # test prior
        prior_samples = np.mean((alpha - 1) *
                            np.log(1 - logit_norm_samples), axis = 0)
        tol3 = 3 * np.std((alpha - 1) * np.log(1 - logit_norm_samples)) \
                    /np.sqrt(num_draws)
        np_test.assert_allclose(
            prior_samples,
            ef.get_e_dp_prior_logitnorm_approx(
                        alpha, lognorm_means, lognorm_infos, gh_loc, gh_weights),
            atol = tol3)

        x = np.random.normal(0, 1e2, size = 10)
        def e_log_v(x):
            return np.sum(ef.get_e_log_logitnormal(\
                        x[0:5], np.abs(x[5:10]), gh_loc, gh_weights)[0])
        quick_grad_check(e_log_v, x)


class TestModelingFunctions(unittest.TestCase):
    def test_e_logistic_term(self):
        z_dim = (3, 2)
        z_mean = np.random.random(z_dim)
        z_sd = np.exp(z_mean)

        num_std_draws = 100
        std_draws = model.get_standard_draws(num_std_draws)

        # This would normally be a matrix of zeros and ones
        y = np.random.random(z_dim)

        num_draws = 10000
        z_draws = sp.stats.norm(loc=z_mean, scale=z_sd).rvs(
            (num_draws, z_dim[0], z_dim[1]))
        logit_term_draws = \
            np.expand_dims(y, axis=0) * z_draws - \
            np.log1p(np.exp(z_draws))

        # The Monte Carlo error will be dominated by the number of draws used
        # in the get_e_logistic_term approximation.
        test_se = np.max(
            3 * np.std(logit_term_draws, axis=0) / np.sqrt(num_std_draws))
        print('Logistic test moment tolerance: ', test_se)
        np_test.assert_allclose(
            np.sum(np.mean(logit_term_draws, axis=0)),
            model.get_e_logistic_term(y, z_mean, z_sd, std_draws),
            atol=test_se)


class TestNaturalParameterUpdates(unittest.TestCase):
    def test_get_uvn_from_natural_parameters(self):
        true_mean = 1.5
        true_info = 0.4
        true_sd = 1 / np.sqrt(true_info)
        num_draws = 10000

        e_x = true_mean
        e_x2 = true_mean ** 2 + true_sd ** 2

        draws = np.random.normal(0, 1, num_draws)
        def get_log_normal_prob(e_x, e_x2):
            sd = np.sqrt(e_x2 - e_x ** 2)
            draws_shift = sd * draws + e_x
            log_pdf = -0.5 * true_info * (draws_shift - true_mean) ** 2
            return np.mean(log_pdf)

        get_log_normal_prob_grad_1 = \
            grad(get_log_normal_prob, argnum=0)
        get_log_normal_prob_grad_2 = \
            grad(get_log_normal_prob, argnum=1)

        e_x_term = get_log_normal_prob_grad_1(e_x, e_x2)
        e_x2_term = get_log_normal_prob_grad_2(e_x, e_x2)
        mean, info = ef.get_uvn_from_natural_parameters(e_x_term, e_x2_term)
        atol = 3 * true_sd / np.sqrt(num_draws)
        np_test.assert_allclose(true_mean, mean, atol=atol, err_msg='mean')
        np_test.assert_allclose(true_info, info, atol=atol, err_msg='info')


if __name__ == '__main__':
    unittest.main()
