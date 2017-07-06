#!/usr/bin/python3

import autograd.numpy as np
from autograd import grad, jacobian, hessian
from VariationalBayes.ExponentialFamilies import \
    univariate_normal_entropy, multivariate_normal_entropy, gamma_entropy, \
    dirichlet_entropy, wishart_entropy
import unittest
import scipy as sp

class TestEntropy(unittest.TestCase):
    def test_uvn_entropy(self):
        mean_par = 2.0
        info_par = 1.5
        num_draws = 10000
        norm_dist = sp.stats.norm(loc=mean_par, scale=np.sqrt(1 / info_par))
        self.assertAlmostEqual(norm_dist.entropy(), univariate_normal_entropy(info_par))

    def test_mvn_entropy(self):
        mean_par = np.array([1., 2.])
        info_par = np.eye(2) + np.full((2, 2), 0.1)
        norm_dist = sp.stats.multivariate_normal(
            mean=mean_par, cov=np.linalg.inv(info_par))
        self.assertAlmostEqual(
            norm_dist.entropy(), multivariate_normal_entropy(info_par))

    def test_gamma_entropy(self):
        shape = 3.0
        rate = 2.4
        gamma_dist = sp.stats.gamma(a=shape, scale=1 / rate)
        self.assertAlmostEqual(gamma_dist.entropy(), gamma_entropy(shape, rate))

    def test_dirichlet_entropy(self):
        alpha = np.array([23, 4, 5, 6, 7])
        dirichlet_dist = sp.stats.dirichlet(alpha)
        self.assertAlmostEqual\
                (dirichlet_dist.entropy(), dirichlet_entropy(alpha))

    def test_wishart_entropy(self):
        df = 4.3
        v = np.eye(2) + np.full((2, 2), 0.1)
        wishart_dist = sp.stats.wishart(df=df, scale=v)
        self.assertAlmostEqual(
            wishart_dist.entropy(), wishart_entropy(df, v))


if __name__ == '__main__':
    unittest.main()
