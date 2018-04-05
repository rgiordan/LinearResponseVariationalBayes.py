# test the regression utilities

import unittest
import numpy.testing as np_test

from LinearResponseVariationalBayes import regression_utils as regression

from copy import deepcopy

import numpy as np
import os
import scipy as sp

import autograd

np.random.seed(43534543)

# some helpful functions:
def symmetrize(array):
    # symmetrize the last two dimensions of an array
    # useful for drawing an array of random symmetric matrices

    assert np.shape(array)[-1] == np.shape(array)[-2]

    return regression.mat_mul_last2dims(array, array)

def regression_objective(beta, y, x, info):
    # this is the objective function (y - X\beta)^T [info] (y - X\beta)
    # y is a n-vector, x is n by r, and beta is an r vector
    # info is n by n

    # we will check the gradient of this function wrt to beta
    # and assert that it is zero.

    assert len(y) == np.shape(x)[0]
    assert np.shape(x)[1] == len(beta)
    assert np.shape(info)[0] == len(y)
    assert np.shape(info)[0] == np.shape(info)[1]

    error = y - np.dot(x, beta)
    return np.dot(np.dot(error, info), error)

def assert_zero_grad_regression_objective(\
            beta, y, x, info, tol = 1e-8, display = False):
    # we check that the gradient is zero at beta for the regression objective

    objective_grad = autograd.grad(regression_objective, argnum = 0)

    if display:
        print(np.abs(objective_grad(beta, y, x, info)))

    assert np.all(np.abs(objective_grad(beta, y, x, info)) < tol)

def regression_objective_sum(beta, y, x, info):
    # in the case that the objective function is the sum over y_n
    # y is N x n_t
    # x is n_t x r

    assert np.shape(x)[0] == np.shape(y)[1]

    n = np.shape(y)[0]
    objective = 0.
    for i in range(n):
        objective += regression_objective(beta, y[i, :], x, info)

    return objective


def get_test_data(y_size, x_size, beta_size):
    # returns a random y, x, and prior parameters
    # and we will regress y on x

    # returns a random y, x, residual info, and prior parameters

    y = np.random.normal(size = y_size)
    x = np.random.normal(size = x_size)

    # get heteroskedastic information of errors
    info_size = x_size[0:-1] + (x_size[-2], )
    info_het = np.random.normal(size = info_size)
    info_het = symmetrize(info_het)

    # homoskedastic info: return a scalar
    info_hom = np.random.rand()

    # priors
    prior_means = np.random.normal(size = beta_size)

    beta_info_size = beta_size + (beta_size[-1], )
    prior_infos = np.random.normal(size = beta_info_size)
    prior_infos = symmetrize(prior_infos)

    return y, x, info_het, info_hom, prior_means, prior_infos


def compute_ridge_regression_update(y, x, info, prior_means, prior_infos):
    # classic ridge regression formula for a vector y
    # and covariate matrix x

    covar = np.linalg.inv(np.dot(x.T, np.dot(info, x)) + prior_infos)

    return \
        np.dot(covar, np.dot(np.dot(info, x).T, y) + \
        np.dot(prior_infos, prior_means)), covar

def inf_norm_diff(x, y):
    return np.max(np.abs(x - y))

class TestMatMulFunctions(unittest.TestCase):
    # test our specialized matrix multiplication functions
    # that are used in getting the regression updates
    def test_matmul(self):
        x1 = np.random.random(size = (4, 5, 3, 2))
        x2 = np.random.random(size = (4, 5, 3, 4))

        out = regression.mat_mul_last2dims(x1, x2)

        for n in range(4):
            for k in range(5):
                assert np.all(np.abs(out[n, k, :, :] - \
                            np.dot(x1[n, k, :, :].T, x2[n, k, :, :])) < 1e-15)

    def test_vecmul(self):
        x = np.random.random(size = (4, 5, 3, 2))
        y1 = np.random.random(size = (4, 5, 3))
        out1 = regression.matvec_mul_last2dims(x, y1)
        for n in range(4):
            for k in range(5):
                assert np.all(np.abs(out1[n, k, :] - \
                                np.dot(x[n, k, :, :].T, y1[n, k, :])) < 1e-15)


        y2 = np.random.random(size = (4, 5, 6, 3))
        out2 = regression.matvec_mul_last2dims(x, y2)

        for n in range(4):
            for k in range(5):
                truth_nk = \
                    np.sum(np.dot(x[n, k, :, :].T, y2[n, k, :, :].T), axis = 1)
                assert np.all(np.abs(out2[n, k, :] - truth_nk) < 1e-14)

class TestRegression(unittest.TestCase):
    # tests the general regression functions
    # should be moved to the LRVB when the functions are moved there

    def test_case1(self):
        # see jupyter notebook for more detailed description of these cases ...
        # this is the simplest case: y \in R^N, x \in R^{N x r}
        # classic regression

        # get data
        N = 100
        r = 5

        y_size = (N)
        x_size = (N, r)
        beta_size = (r, )

        y, x, info_het, info_hom, prior_means, prior_infos =  \
                get_test_data(y_size, x_size, beta_size)

        # check homoskedastic case:
        beta_hom = regression.get_regression_coefficients(y, x, info_hom)
        assert_zero_grad_regression_objective(
            beta_hom, y, x, np.eye(N) * info_hom)

        # check heteroskedastic case:
        beta_het = regression.get_regression_coefficients(y, x, info_het)
        assert_zero_grad_regression_objective(beta_het, y, x, info_het)

        # check ridge regression:
        beta_mean, beta_info = \
            regression.get_posterior_regression_coefficients(\
                    y, x, info_het, prior_means, prior_infos)

        beta_mean_true, beta_covar_true = \
            compute_ridge_regression_update(y, x, info_het, \
                                            prior_means, prior_infos)

        assert inf_norm_diff(beta_mean, beta_mean_true) < 1e-8
        assert inf_norm_diff(beta_info, np.linalg.inv(beta_covar_true)) < 1e-8

    def test_case2(self):
        # this is N independent ridge regressions
        # this is what we need for two stage modeling

        # get data
        N = 20 # number of observations
        r = 5 # dimension of regression coefficients
        n_t = 6 # number of time points, ie. dimension of each y_n

        y_size = (N, n_t)
        x_size = (N, n_t, r)
        beta_size = (N, r)

        y, x, info_het, info_hom, prior_means, prior_infos = \
            get_test_data(y_size, x_size, beta_size)

        # check homoskedastic case
        beta_hom = regression.get_regression_coefficients(y, x, info_hom)

        for i in range(N):
            # check like N independent regressions
            assert_zero_grad_regression_objective(beta_hom[i, :], y[i, :], x[i, :], \
                                                    info_hom * np.eye(n_t))

        # check heteroskedastic case
        beta_het = regression.get_regression_coefficients(y, x, info_het)
        for i in range(N):
            # check like N independent regressions
            assert_zero_grad_regression_objective(beta_het[i, :], y[i, :], x[i, :], \
                                                info_het[i, :, :])

        # check ridge regression case
        beta_mean, beta_info = \
            regression.get_posterior_regression_coefficients(y, x, \
                        info_het, prior_means, prior_infos)

        for i in range(N):
            # this is like N independent ridge regressions
            beta_mean_true, beta_covar_true = \
                compute_ridge_regression_update(\
                        y[i, :], x[i, :, :], info_het[i, :, :],
                        prior_means[i, :], prior_infos[i, :, :])

            assert inf_norm_diff(beta_mean[i, :], beta_mean_true) < 1e-8
            assert inf_norm_diff(beta_info[i, :, :], \
                                np.linalg.inv(beta_covar_true)) < 1e-8

    def test_case3(self):
        # this is N x K independent ridge regressions
        # this is what we need for the shifts

        # get data
        N = 20
        k_approx = 11
        n_t = 8
        r = 5

        y_size = (N, k_approx, n_t)
        x_size = (N, k_approx, n_t, r)
        beta_size = (N, k_approx, r)

        y, x, info_het, info_hom, prior_means, prior_infos = \
            get_test_data(y_size, x_size, beta_size)

        # check homoskedastic case
        beta_hom = regression.get_regression_coefficients(y, x, info_hom)
        for n in range(N):
            for k in range(k_approx):
                # check like N x k_approx independent regressions
                assert_zero_grad_regression_objective(\
                    beta_hom[n, k, :], y[n, k, :], x[n, k, :, :], \
                    np.eye(n_t) * info_hom)

        # check heteroskedastic case
        beta_het = regression.get_regression_coefficients(y, x, info_het)
        for n in range(N):
            for k in range(k_approx):
                # check like N x k_approx independent regressions
                assert_zero_grad_regression_objective(\
                    beta_het[n, k, :], y[n, k, :], \
                    x[n, k, :, :], info_het[n, k, :, :])

        # check ridge regression
        beta_mean, beta_info =\
            regression.get_posterior_regression_coefficients(y, x, info_het, \
                                                    prior_means, prior_infos)

        for i in range(N):
            for k in range(k_approx):
                # this is like Nxk independent ridge regressions

                beta_mean_true, beta_covar_true = \
                    compute_ridge_regression_update(\
                        y[i, k, :], x[i, k, :, :], info_het[i, k, :, :],
                        prior_means[i, k, :], prior_infos[i, k, :, :])

                assert inf_norm_diff(beta_mean[i, k, :], beta_mean_true) < 1e-8
                assert inf_norm_diff(beta_info[i, k, :, :], \
                            np.linalg.inv(beta_covar_true)) < 1e-8


    def test_case4(self):
        # this is like ridge regression, where we have N observations,
        # but each observation is n_t dimensional

        # get data
        N = 20
        n_t = 12
        r = 5

        y_size = (N, n_t)
        x_size = (n_t, r)
        beta_size = (r, )

        y, x, info_het, info_hom, prior_means, prior_infos = \
            get_test_data(y_size, x_size, beta_size)

        objective_grad = autograd.grad(regression_objective_sum, argnum = 0)

        # check homoskedastic case
        beta_hom = regression.get_regression_coefficients(y, x, info_hom)
        assert np.all(np.abs(objective_grad(beta_hom, y, x, info_hom * np.eye(n_t)) \
                        < 1e-8))

        # check heteroskedastic case
        beta_het = regression.get_regression_coefficients(y, x, info_het)
        assert np.all(np.abs(objective_grad(beta_het, y, x, info_het) < 1e-8))

        # check against ridge regression
        y_ = np.sum(y, axis = 0) / np.sqrt(N)
        x_ = x * np.sqrt(N)
        beta_mean_true, beta_covar_true = \
            compute_ridge_regression_update(y_, x_, info_het, \
                                            prior_means, prior_infos)

        beta_mean, beta_info = \
            regression.get_posterior_regression_coefficients(y, x, info_het, \
                                                    prior_means, prior_infos)

        assert inf_norm_diff(beta_mean, beta_mean_true) < 1e-8
        assert inf_norm_diff(beta_info, np.linalg.inv(beta_covar_true)) < 1e-8

    def test_case5(self):
        # this is an example of the most general case
        # where we have L x M x N x K observations each of dimension y
        # The L x M regressions are independent
        # for each L x M group there are N x K observations sharing a
        # design matrix

        L = 5
        M = 3
        N = 7
        K = 4
        n_t = 6

        # dimension of regression coeffs
        r = 5

        # data and design matrix
        y_size = (L, M, N, K, n_t)
        x_size = (L, M, n_t, r)
        beta_size = (L, M, r)

        y, x, info_het, info_hom, prior_means, prior_infos = \
            get_test_data(y_size, x_size, beta_size)

        # check homoskedastic case
        beta_hom = regression.get_regression_coefficients(y, x, info_hom)

        for l in range(L):
            for m in range(M):
                # stack last two dimensions
                y_flatten = y[l, m, :, :, :].reshape(N * K, n_t)
                x_flatten = x[l, m, :, :]

                objective_grad = autograd.grad(regression_objective_sum, argnum = 0)

                assert np.all(np.abs(objective_grad(beta_hom[l, m, :], y_flatten,\
                                x_flatten, info_hom * np.eye(n_t)) < 1e-8))

        # check heteroskedastic case
        beta_het = regression.get_regression_coefficients(y, x, info_het)

        for l in range(L):
            for m in range(M):
                # stack last two dimensions
                y_flatten = y[l, m, :, :, :].reshape(N * K, n_t)
                x_flatten = x[l, m, :, :]

                objective_grad = autograd.grad(regression_objective_sum, argnum = 0)

                assert np.all(np.abs(objective_grad(beta_het[l, m, :], y_flatten, \
                                    x_flatten, info_het[l, m, :, :]) < 1e-8))

        # check ridge regression
        beta_mean, beta_info = \
        regression.get_posterior_regression_coefficients(y, x, info_het, \
                    prior_means, prior_infos)

        for l in range(L):
            for m in range(M):
                # stack last two dimensions
                y_flatten = y[l, m, :, :, :].reshape(N * K, n_t)
                x_flatten = x[l, m, :, :]

                # scale for the correct ridge regression formula
                y_ = np.sum(y_flatten, axis = 0) / np.sqrt(N * K)
                x_ = x_flatten * np.sqrt(N * K)

                beta_mean_true, beta_covar_true =\
                    compute_ridge_regression_update(y_, x_, info_het[l, m], \
                                prior_means[l, m, :], prior_infos[l, m, :, :])

                assert inf_norm_diff(beta_mean[l, m, :], beta_mean_true) < 1e-10
                assert inf_norm_diff(np.linalg.inv(beta_info[l, m, :, :]), \
                                        beta_covar_true) < 1e-10



if __name__ == '__main__':
    unittest.main()
