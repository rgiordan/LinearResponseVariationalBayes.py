# library for the regression functions

import autograd
import autograd.numpy as np
import autograd.scipy as sp


def mat_mul_last2dims(x1, x2):
    # multiply the last two dimensions of two arrays

    # x1 is m1 x m2 ....x m_d x s x r
    # x2 is m1 x m2 ... x m_d x s x t

    # x1 and x2 are an (m1 x m2 ....x m_d) array of matrices,
    # whose last two dimensions specify a matrix.

    # We return 'x1^T x2', this matrix multiplication done along
    # the last two dimensions

    assert len(np.shape(x1)) == (len(np.shape(x2)))
    assert np.shape(x2)[-2] == np.shape(x1)[-2]

    d = len(np.shape(x1))
    einsum_indx2 = list(range(d - 1))
    einsum_indx2.append(d)
    einsum_indx_out = list(range(d - 2))
    einsum_indx_out.append(d-1)
    einsum_indx_out.append(d)

    return np.einsum(x1, list(range(d)),
                      x2, einsum_indx2,
                      einsum_indx_out)

def matvec_mul_last2dims(x, y):
    # x is m1 x m2 ....x m_d2 x s x r
    # y is m1 x m2 ... x m_d1 x s

    # we do x^T y, along the last two dimension of x
    # the y can have more dimension than x (d1 >= d2),
    # in which case we sum y over the extra dimensions

    assert np.shape(x)[-2] == np.shape(y)[-1]

    d1 = len(np.shape(y)) - 1
    d2 = len(np.shape(x)) - 2

    assert d2 <= d1

    einsum_indx1 = list(range(d2))
    einsum_indx1.append(d1)
    einsum_indx1.append(d1 + 1)
    einsum_indx2 = list(range(d1 + 1))
    einsum_indx_out = list(range(d2))
    einsum_indx_out.append(d1 + 1)

    return np.einsum(x, einsum_indx1,
                     y, einsum_indx2, einsum_indx_out)

def get_nat_params_from_likelihood(y, x, info):
    # get the natural parameters of beta from the likelihood
    # ie. get the coefficeints of beta and beta.T x beta

    assert np.shape(y)[-1] == np.shape(x)[-2]

    d1 = len(np.shape(y)) - 1
    d2 = len(np.shape(x)) - 2
    assert d1 >= d2

    # extra dims
    n_per_dim = np.prod(np.shape(y)[d2:d1])

    homosked = np.isscalar(info) # whether the errors are homoskedastic
    if homosked:
        # info is just a scalar
        info_x = x * info
    else:
        # info is n_t x n_t
        assert np.shape(info)[-1] == np.shape(info)[-2]
        assert np.shape(info)[-1] == np.shape(x)[-2]
        assert len(np.shape(info)) == len(np.shape(x))

        info_x = mat_mul_last2dims(info, x)


    nat_param1 = matvec_mul_last2dims(info_x, y)
    nat_param2 = -0.5 * mat_mul_last2dims(x, info_x) * n_per_dim

    return nat_param1, nat_param2


def get_nat_params_from_prior(prior_means, prior_infos):
    # get the natural parameters of beta from the prior
    # ie. get the coefficeints of beta and beta.T x beta

    # this is a special case of get_nat_params_from_likelihood above

    assert np.shape(prior_means)[-1] == np.shape(prior_infos)[-1]
    assert np.shape(prior_infos)[-2] == np.shape(prior_infos)[-1]
    assert len(np.shape(prior_means)) == (len(np.shape(prior_infos)) - 1)

    r = np.shape(prior_means)[-1] # dimension of beta
    d2 = len(np.shape(prior_means))-1 # number of beta's

    x_shape = np.shape(prior_means)[0:-1] + (1, 1)
    x = np.tile(np.eye(r), x_shape)

    return get_nat_params_from_likelihood(prior_means, x, prior_infos)


def get_mvn_from_nat_params(nat_param1, nat_param2):
    # coverts the natural parameters to mean and info parameters

    info = -2 * nat_param2
    mean = matvec_mul_last2dims(np.linalg.inv(info), nat_param1)

    return mean, info

def get_regression_coefficients(y, x, info):
    nat_param1, nat_param2 = get_nat_params_from_likelihood(y, x, info)
    mean, _ = get_mvn_from_nat_params(nat_param1, nat_param2)

    return mean

def get_posterior_regression_coefficients(y, x, info, prior_means, prior_infos):
    # returns the posterior mean and info
    # of the regression coefficients

    nat_param1, nat_param2 = get_nat_params_from_likelihood(y, x, info)
    prior_nat_param1, prior_nat_param2 = \
        get_nat_params_from_prior(prior_means, prior_infos)

    return get_mvn_from_nat_params(nat_param1 + prior_nat_param1,
                                  nat_param2 + prior_nat_param2)
