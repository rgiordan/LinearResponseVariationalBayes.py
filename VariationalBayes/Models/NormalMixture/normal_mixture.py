import autograd
import autograd.numpy as np
import autograd.scipy as sp
import scipy as osp
from scipy import optimize
from scipy import stats
from copy import deepcopy

from scipy.sparse import csr_matrix

def generate_data(num_obs, true_mu, true_sigma, true_pi):
    true_z = np.random.multinomial(1, true_pi, num_obs)
    true_z_ind = np.full(num_obs, -1)
    for row in np.argwhere(true_z):
        true_z_ind[row[0]] = row[1]

    x = np.array([ np.random.multivariate_normal(
                    true_mu[true_z_ind[n]], true_sigma[true_z_ind[n]]) \
                   for n in range(num_obs) ])

    return x, true_z, true_z_ind

def get_info_logdet_array(info):
    return np.array([ np.linalg.slogdet(info[k, :, :])[1] \
                      for k in range(info.shape[0]) ])

# This is the log probability of each observation for each component.
def loglik_obs_by_k(mu, info, pi, x):
    log_lik = \
        -0.5 * np.einsum('ni, kij, nj -> nk', x, info, x) + \
               np.einsum('ni, kij, kj -> nk', x, info, mu) + \
        -0.5 * np.expand_dims(np.einsum('ki, kij, kj -> k', mu, info, mu), axis=0)

    logdet_array = np.expand_dims(get_info_logdet_array(info), axis=0)
    log_pi = np.log(pi)

    log_lik += 0.5 * logdet_array + log_pi

    return log_lik

def mu_prior(mu, mu_prior_mean, mu_prior_info):
    k_num = mu.shape[0]
    d_num = len(mu_prior_mean)
    assert mu.shape[1] == d_num
    assert mu_prior_info.shape[0] == d_num
    assert mu_prior_info.shape[1] == d_num
    mu_prior_val = 0.0
    for k in range(k_num):
        mu_centered = mu[k, :] - mu_prior_mean
        mu_prior_val += -0.5 * np.matmul(
            np.matmul(mu_centered, mu_prior_info), mu_centered)
    return mu_prior_val

def pi_prior(pi, alpha):
    return np.sum(alpha * np.log(pi))

def info_prior(info, dof):
    k_num = info.shape[0]
    d_num = info.shape[1]
    assert d_num == info.shape[2]
    assert dof > d_num - 1
    # Not a complete Wishart prior
    # TODO: cache the log determinants.
    info_prior_val = 0.0
    for k in range(k_num):
        sign, logdet = np.linalg.slogdet(info[k, :, :])
        info_prior_val += 0.5 * (dof - d_num - 1) * logdet
    return info_prior_val

# TODO: put this in a library
def multinoulli_entropy(e_z):
    return -1 * np.sum(e_z * np.log(e_z))

def get_sparse_multinoulli_entropy_hessian(e_z_vec):
    k = len(e_z_vec)
    vals = -1. / e_z_vec
    return csr_matrix((vals, ((range(k)), (range(k)))), (k, k))


class Model(object):
    def __init__(self, x, params, prior_params):
        self.x = x
        self.params = deepcopy(params)
        self.prior_params = deepcopy(prior_params)
        self.weights = np.full((x.shape[0], 1), 1.0)
        self.get_moment_jacobian = \
            autograd.jacobian(self.get_interesting_moments)

    def loglik_obs_by_k(self):
        info = self.params['global']['info'].get()
        mu = self.params['global']['mu'].get()
        pi = self.params['global']['pi'].get()
        return loglik_obs_by_k(mu, info, pi, self.x)

    # This needs to be defined so we can differentiate it for CAVI.
    def loglik_e_z(self, e_z):
        return np.sum(e_z * self.loglik_obs_by_k())

    def loglik(self):
        e_z = self.params['local']['e_z'].get()
        return self.loglik_e_z(e_z)

    def loglik_obs(self):
        e_z = self.params['local']['e_z'].get()
        log_lik_array = self.loglik_obs_by_k()
        return np.sum(log_lik_array * e_z, axis=1)

    def kl_optimal_z(self):
        self.optimize_z()
        return -1. * np.sum(self.loglik_obs())

    def prior(self):
        info = self.params['global']['info'].get()
        mu = self.params['global']['mu'].get()
        pi = self.params['global']['pi'].get()
        mu_prior_mean = self.prior_params['mu_prior_mean'].get()
        mu_prior_info = self.prior_params['mu_prior_info'].get()
        prior = 0.
        prior += mu_prior(mu, mu_prior_mean, mu_prior_info)
        prior += pi_prior(pi, self.prior_params['alpha'].get())
        prior += info_prior(info, self.prior_params['dof'].get())
        return prior

    def optimize_z(self):
        # Take a CAVI step on Z.
        info = self.params['global']['info'].get()
        mu = self.params['global']['mu'].get()
        pi = self.params['global']['pi'].get()
        e_z = self.params['local']['e_z'].get()

        natural_parameters = self.loglik_obs_by_k()
        z_logsumexp = np.expand_dims(
            sp.misc.logsumexp(natural_parameters, 1), axis=1)
        e_z = np.exp(natural_parameters - z_logsumexp)
        self.params['local']['e_z'].set(e_z)

    def kl(self, include_local_entropy=True):
        elbo = self.prior() + self.loglik()

        if include_local_entropy:
            e_z = self.params['local']['e_z'].get()
            elbo += multinoulli_entropy(e_z)

        return -1 * elbo


    #######################
    # Moments for sensitivity

    def get_interesting_moments(self, global_free_params):
        self.params['global'].set_free(global_free_params)
        self.optimize_z()
        return self.params['global']['mu'].get_vector()

    ######################################
    # Compute sparse hessians by hand.

    # Log likelihood by data point.

    # The rows are the z vector indices and the columns are the data points.
    def loglik_vector_local_weight_hessian_sparse(self):
        log_lik_array = self.loglik_obs_by_k()

        hess_vals = [] # These will be the entries of dkl / dz dweight^T
        hess_rows = [] # These will be the z indices
        hess_cols = [] # These will be the data indices
        # This is the Hessian of the negative entropy, which enters the KL
        # divergence.
        e_z_param = self.params['local']['e_z']
        e_z = e_z_param.get()
        for row in range(e_z.shape[0]):
            z_row_inds = e_z_param.get_vector_indices(row)
            for col in range(e_z.shape[1]):
                hess_vals.append(log_lik_array[row, col])
                hess_rows.append(z_row_inds[col])
                hess_cols.append(row)

        local_size = e_z_param.vector_size()
        return csr_matrix((hess_vals, (hess_rows, hess_cols)),
                                     (local_size, self.x.shape[0]))

    # KL
    def kl_vector_local_hessian_sparse(self, global_vec, local_vec):
        self.params['global'].set_vector(global_vec)
        self.params['local'].set_vector(local_vec)
        hess_vals = []
        hess_rows = []
        # This is the Hessian of the negative entropy, which enters the KL
        # divergence.
        e_z = self.params['local']['e_z'].get()
        for row in range(e_z.shape[0]):
            # Note that we are relying on the fact that the local parameters
            # only contain e_z, so the vector index in e_z is the vector index
            # in the local parameters.
            row_inds = self.params['local']['e_z'].get_vector_indices(row)
            for col in range(e_z.shape[1]):
                hess_vals.append(1. / e_z[row, col])
                hess_rows.append(row_inds[col])
        local_size = self.params['local']['e_z'].vector_size()
        return csr_matrix((hess_vals, (hess_rows, hess_rows)),
                                    (local_size, local_size))

    ######################
    # Everything below here should be boilerplate.

    # The SparseObjectives module still needs to support sparse Jacobians.
    def loglik_free_local_weight_hessian_sparse(self):
        free_par_local = self.params['local'].get_free()
        free_to_vec_jac = \
            self.params['local'].free_to_vector_jac(free_par_local)
        return free_to_vec_jac .T * \
               self.loglik_vector_local_weight_hessian_sparse()

    def loglik_free_weight_hessian_sparse(self):
        get_loglik_obs_free_global_jac = \
            autograd.jacobian(self.loglik_obs_free_global_local, argnum=0)
        loglik_obs_free_global_jac = \
            get_loglik_obs_free_global_jac(self.params['global'].get_free(),
                                           self.params['local'].get_free()).T
        loglik_obs_free_local_jac = \
            self.loglik_free_local_weight_hessian_sparse()

        return osp.sparse.vstack(
            [ loglik_obs_free_global_jac, loglik_obs_free_local_jac ])

    def loglik_obs_free_global_local(self, free_params_global, free_params_local):
        self.params['global'].set_free(free_params_global)
        self.params['local'].set_free(free_params_local)
        return self.loglik_obs()
