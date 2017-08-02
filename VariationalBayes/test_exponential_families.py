#!/usr/bin/python3

import autograd.numpy as np
from autograd import grad, jacobian, hessian
import VariationalBayes.ExponentialFamilies as ef
import VariationalBayes.Modeling as model
import unittest
import numpy.testing as np_test
import scipy as sp

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

    def test_wishart_entropy(self):
        df = 4.3
        v = np.eye(2) + np.full((2, 2), 0.1)
        wishart_dist = sp.stats.wishart(df=df, scale=v)
        self.assertAlmostEqual(
            wishart_dist.entropy(), ef.wishart_entropy(df, v))

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


if __name__ == '__main__':
    unittest.main()
