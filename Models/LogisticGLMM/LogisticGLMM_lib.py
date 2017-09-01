
import VariationalBayes as vb
import VariationalBayes.ExponentialFamilies as ef
import VariationalBayes.Modeling as modeling
from VariationalBayes.SparseObjectives import get_sparse_hessian
from VariationalBayes.Parameters import convert_vector_to_free_hessian

import autograd
import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy as sp
import scipy as osp
import numpy as onp

import copy

def get_glmm_parameters(
    K, NG,
    mu_info_min=0.0, tau_alpha_min=0.0, tau_beta_min=0.0,
    beta_diag_min=0.0, u_info_min=0.0):

    glmm_par = vb.ModelParamsDict('GLMM Parameters')
    glmm_par.push_param(vb.UVNParam('mu', min_info=mu_info_min))
    glmm_par.push_param(
        vb.GammaParam('tau', min_shape=tau_alpha_min, min_rate=tau_beta_min))
    #glmm_par.push_param(vb.MVNParam('beta', K, min_info=beta_diag_min))
    glmm_par.push_param(vb.UVNParamVector('beta', K, min_info=beta_diag_min))
    glmm_par.push_param(vb.UVNParamVector('u', NG, min_info=u_info_min))

    return glmm_par


def simulate_data(N, NG, true_beta, true_mu, true_tau):
    def Logistic(u):
        return np.exp(u) / (1 + np.exp(u))

    K = len(true_beta)
    NObs = NG * N
    true_u = np.random.normal(true_mu, 1 / np.sqrt(true_tau), NG)

    x_mat = np.random.random(K * NObs).reshape(NObs, K) - 0.5
    y_g_vec = [ g for g in range(NG) for n in range(N) ]
    true_rho = Logistic(np.matmul(x_mat, true_beta) + true_u[y_g_vec])
    y_vec = np.random.random(NObs) < true_rho

    return x_mat, y_g_vec, y_vec, true_rho, true_u


def get_e_logistic_term_guass_hermite(
    z_mean, z_sd, gh_x, gh_w, aggregate_all=True):

    assert z_mean.shape == z_sd.shape
    draws_axis = z_sd.ndim
    z_vals = \
        np.sqrt(2) * np.expand_dims(z_sd, axis=draws_axis) * gh_x + \
        np.expand_dims(z_mean, axis=draws_axis)

    # By dividing by the number of standard draws after summing,
    # we add the sample means for all the observations.
    # Note that
    # log(1 - p) = log(1 / (1 + exp(z))) = -log(1 + exp(z))
    logit_term = gh_w * np.log1p(np.exp(z_vals)) / np.sqrt(np.pi)
    if aggregate_all:
        return np.sum(logit_term)
    else:
        return np.sum(logit_term, axis=draws_axis)


def get_default_prior_params(K):
    prior_par = vb.ModelParamsDict('Prior Parameters')
    prior_par.push_param(
        vb.VectorParam('beta_prior_mean', K, val=np.zeros(K)))
    prior_par.push_param(
        vb.PosDefMatrixParam('beta_prior_info', K, val=0.01 * np.eye(K)))

    prior_par.push_param(vb.ScalarParam('mu_prior_mean', val=0))
    prior_par.push_param(vb.ScalarParam('mu_prior_info', val=0.5))

    prior_par.push_param(vb.ScalarParam('tau_prior_alpha', val=3.0))
    prior_par.push_param(vb.ScalarParam('tau_prior_beta', val=10.0))

    return prior_par


####### Modeling functions
def get_data_log_lik_terms(e_beta, var_beta, e_u, var_u,
                           x_mat, y_vec, y_g_vec, gh_x, gh_w):
    # Log likelihood from data.
    z_mean = e_u[y_g_vec] + np.matmul(x_mat, e_beta)
    z_sd = np.sqrt(
        var_u[y_g_vec] + np.einsum('nk,k,nk->n',
                          x_mat, var_beta, x_mat))
    return \
        y_vec * z_mean - \
        get_e_logistic_term_guass_hermite(
            z_mean, z_sd, gh_x, gh_w, aggregate_all=False)


def get_re_log_lik(e_mu, var_mu, e_tau, e_log_tau, e_u, var_u):
    return -0.5 * e_tau * np.sum(
        ((e_mu - e_u) ** 2) + var_mu + var_u) + 0.5 * e_log_tau * len(e_u)




class LogisticGLMM(object):
    def __init__(
        self, glmm_par, prior_par, x_mat, y_vec, y_g_vec, num_gh_points):

        self.glmm_par = copy.deepcopy(glmm_par)
        self.prior_par = copy.deepcopy(prior_par)
        self.x_mat = x_mat
        self.y_vec = y_vec
        self.y_g_vec = y_g_vec
        self.set_gh_points(num_gh_points)

        self.use_weights = False
        self.weights = np.full(self.x_mat.shape[0], 1.0)

        assert np.min(y_g_vec) == 0
        assert np.max(y_g_vec) == self.glmm_par['u'].size() - 1

    def set_gh_points(self, num_gh_points):
        self.gh_x, self.gh_w = onp.polynomial.hermite.hermgauss(num_gh_points)

    def get_e_log_prior(self):
        e_beta = self.glmm_par['beta'].mean.get()
        info_beta = self.glmm_par['beta'].info.get()
        #cov_beta = np.linalg.inv(info_beta)
        cov_beta = np.diag(1. / info_beta)
        beta_prior_info = self.prior_par['beta_prior_info'].get()
        beta_prior_mean = self.prior_par['beta_prior_mean'].get()
        e_mu = self.glmm_par['mu'].mean.get()
        info_mu = self.glmm_par['mu'].info.get()
        var_mu = 1 / info_mu
        e_tau = self.glmm_par['tau'].e()
        e_log_tau = self.glmm_par['tau'].e_log()

        e_log_p_beta = ef.mvn_prior(
            prior_mean = self.prior_par['beta_prior_mean'].get(),
            prior_info = self.prior_par['beta_prior_info'].get(),
            e_obs = e_beta,
            cov_obs = cov_beta)

        e_log_p_mu = ef.uvn_prior(
            prior_mean = self.prior_par['mu_prior_mean'].get(),
            prior_info = self.prior_par['mu_prior_info'].get(),
            e_obs = e_mu,
            var_obs = var_mu)

        e_log_p_tau = ef.gamma_prior(
            prior_shape = self.prior_par['tau_prior_alpha'].get(),
            prior_rate = self.prior_par['tau_prior_beta'].get(),
            e_obs = e_tau,
            e_log_obs = e_log_tau)

        return e_log_p_beta + e_log_p_mu + e_log_p_tau

    def get_data_log_lik_terms(self):
        e_beta = self.glmm_par['beta'].e()
        var_beta = self.glmm_par['beta'].var()
        e_u = self.glmm_par['u'].e()
        var_u = self.glmm_par['u'].var()

        return get_data_log_lik_terms(
            e_beta = e_beta,
            var_beta = var_beta,
            e_u = e_u,
            var_u = var_u,
            x_mat = self.x_mat,
            y_vec = self.y_vec,
            y_g_vec = self.y_g_vec,
            gh_x = self.gh_x,
            gh_w = self.gh_w)

    def get_log_lik(self):
        if self.use_weights:
            log_lik = np.sum(self.weights * self.get_data_log_lik_terms())
        else:
            log_lik = np.sum(self.get_data_log_lik_terms())

        # Log likelihood from random effect terms.
        e_mu = self.glmm_par['mu'].e()
        var_mu = self.glmm_par['mu'].var()
        e_tau = self.glmm_par['tau'].e()
        e_log_tau = self.glmm_par['tau'].e_log()
        e_u = self.glmm_par['u'].e()
        var_u = self.glmm_par['u'].var()

        log_lik += get_re_log_lik(
            e_mu = e_mu,
            var_mu = var_mu,
            e_tau = e_tau,
            e_log_tau = e_log_tau,
            e_u = e_u,
            var_u = var_u)

        return log_lik

    def get_entropy(self):
        info_mu = self.glmm_par['mu'].info.get()
        info_beta = self.glmm_par['beta'].info.get()
        info_u = self.glmm_par['u'].info.get()
        tau_shape = self.glmm_par['tau'].shape.get()
        tau_rate = self.glmm_par['tau'].rate.get()

        return \
            ef.univariate_normal_entropy(info_mu) + \
            ef.univariate_normal_entropy(info_beta) + \
            ef.univariate_normal_entropy(info_u) + \
            ef.gamma_entropy(tau_shape, tau_rate)

    def get_elbo(self):
        return np.squeeze(
            self.get_log_lik() + self.get_entropy() + self.get_e_log_prior())

    def get_kl(self):
        return -1 * self.get_elbo()


class MomentWrapper(object):
    def __init__(self, glmm_par):
        self.glmm_par = copy.deepcopy(glmm_par)
        K = glmm_par['beta'].mean.size()
        NG =  glmm_par['u'].mean.size()
        self.moment_par = vb.ModelParamsDict('Moment Parameters')
        self.moment_par.push_param(vb.VectorParam('e_beta', K))
        self.moment_par.push_param(vb.ScalarParam('e_mu'))
        self.moment_par.push_param(vb.ScalarParam('e_tau'))
        self.moment_par.push_param(vb.ScalarParam('e_log_tau'))
        self.moment_par.push_param(vb.VectorParam('e_u', NG))

    def __str__(self):
        return str(self.moment_par)

    def set_moments(self, free_par_vec):
        self.glmm_par.set_free(free_par_vec)
        self.moment_par['e_beta'].set(self.glmm_par['beta'].e())
        self.moment_par['e_mu'].set(self.glmm_par['mu'].e())
        self.moment_par['e_tau'].set(self.glmm_par['tau'].e())
        self.moment_par['e_log_tau'].set(self.glmm_par['tau'].e_log())
        self.moment_par['e_u'].set(self.glmm_par['u'].e())

    # Return a posterior moment of interest as a function of unconstrained parameters.
    def get_moment_vector(self, free_par_vec):
        self.set_moments(free_par_vec)
        return self.moment_par.get_vector()


#####################################
# A sparse version of the objective to construct sparse Hessians

# Since we never use the free version of the observation parameters,
# we don't need to set the minimum allowable values.
def get_group_parameters(K):
    group_par = vb.ModelParamsDict('Single group GLMM parameters')
    group_par.push_param(vb.UVNParam('mu'))
    group_par.push_param(vb.GammaParam('tau'))
    group_par.push_param(vb.UVNParamVector('beta', K))
    group_par.push_param(vb.UVNParamVector('u', 1))
    return group_par

# Since we never use the free version of the global parameters, we don't need to
# set the minimum allowable values.
def get_global_parameters(K):
    global_par = vb.ModelParamsDict('Global GLMM parameters')
    global_par.push_param(vb.UVNParam('mu'))
    global_par.push_param(vb.GammaParam('tau'))
    global_par.push_param(vb.UVNParamVector('beta', K))
    return global_par

def set_group_parameters(glmm_par, group_par, group):
    group_par['beta'].set_vector(glmm_par['beta'].get_vector())
    group_par['mu'].set_vector(glmm_par['mu'].get_vector())
    group_par['tau'].set_vector(glmm_par['tau'].get_vector())

    group_par['u'].mean.set(glmm_par['u'].mean.get()[group])
    group_par['u'].info.set(glmm_par['u'].info.get()[group])

def set_global_parameters(glmm_par, global_par):
    global_par['beta'].set_vector(glmm_par['beta'].get_vector())
    global_par['mu'].set_vector(glmm_par['mu'].get_vector())
    global_par['tau'].set_vector(glmm_par['tau'].get_vector())


class SparseModelObjective(LogisticGLMM):
    def __init__(self, glmm_par, prior_par, x_mat, y_vec, y_g_vec, num_gh_points):
        super().__init__(glmm_par, prior_par, x_mat, y_vec, y_g_vec, num_gh_points)

        self.glmm_indices = copy.deepcopy(self.glmm_par)
        self.glmm_indices.set_vector(np.arange(0, self.glmm_indices.vector_size()))

        # Parameters for a single observation.
        K = glmm_par['beta'].size()
        self.group_par = get_group_parameters(K)
        self.group_indices = get_group_parameters(K)
        self.group_indices.set_vector(np.arange(0, self.group_indices.vector_size()))

        self.group_rows = [ self.y_g_vec == g \
                            for g in range(np.max(self.y_g_vec) + 1)]

        # Parameters that are shared across all observations.
        self.global_par = get_global_parameters(K)
        self.global_indices = get_global_parameters(K)
        self.global_indices.set_vector(np.arange(0, self.global_indices.vector_size()))

        # Hessians with respect to the vectorized versions of the observation
        # and global parameter vectors.
        self.get_group_vector_hessian = \
            autograd.hessian(self.get_group_elbo_from_vec)
        self.get_global_vector_hessian = \
            autograd.hessian(self.get_global_elbo_from_vec)
        self.get_vector_grad = \
            autograd.grad(self.get_elbo_from_vec)
        self.get_vector_hessian = \
            autograd.hessian(self.get_elbo_from_vec)

    # Set the group parameters from the global parameters and
    # return a vector of the indices within the full model.
    def set_group_parameters(self, group):
        set_group_parameters(self.glmm_par, self.group_par, group)
        set_group_parameters(self.glmm_indices, self.group_indices, group)
        return self.group_par.get_vector(), self.group_indices.get_vector()

    def set_global_parameters(self, unused_group=-1):
        set_global_parameters(self.glmm_par, self.global_par)
        set_global_parameters(self.glmm_indices, self.global_indices)
        return self.global_par.get_vector(), self.global_indices.get_vector()

    # Likelihood functions:
    def get_group_elbo(self, group):
        e_beta = self.group_par['beta'].e()
        var_beta = self.group_par['beta'].var()
        e_u = np.array([self.group_par['u'].e()])
        var_u = np.array([self.group_par['u'].var()])
        info_u = np.array([self.group_par['u'].info.get()])
        e_tau = self.group_par['tau'].e()
        e_log_tau = self.group_par['tau'].e_log()
        e_mu = self.group_par['mu'].e()
        var_mu = self.group_par['mu'].var()

        assert(len(e_u) == 1)
        assert(len(var_u) == 1)
        assert(len(info_u) == 1)

        return \
            np.sum(get_data_log_lik_terms(
                e_beta=e_beta,
                var_beta=var_beta,
                e_u=e_u,
                var_u=var_u,
                y_g_vec=[0],
                x_mat=self.x_mat[self.group_rows[group], :],
                y_vec=self.y_vec[self.group_rows[group]],
                gh_x=self.gh_x,
                gh_w=self.gh_w)) + \
            get_re_log_lik(
                e_mu=e_mu,
                var_mu=var_mu,
                e_tau=e_tau,
                e_log_tau=e_log_tau,
                e_u=np.array([e_u]),
                var_u=np.array([var_u])) + \
            ef.univariate_normal_entropy(info_u)

    def get_global_elbo(self):
        e_beta = self.global_par['beta'].mean.get()
        info_beta = self.global_par['beta'].info.get()
        cov_beta = np.diag(1. / info_beta)

        e_mu = self.global_par['mu'].mean.get()
        info_mu = self.global_par['mu'].info.get()
        var_mu = 1 / info_mu

        tau_shape = self.global_par['tau'].shape.get()
        tau_rate = self.global_par['tau'].rate.get()
        e_tau = self.global_par['tau'].e()
        e_log_tau = self.global_par['tau'].e_log()

        e_log_p_beta = ef.mvn_prior(
            prior_mean=self.prior_par['beta_prior_mean'].get(),
            prior_info=self.prior_par['beta_prior_info'].get(),
            e_obs=e_beta, cov_obs=cov_beta)

        e_log_p_mu = ef.uvn_prior(
            prior_mean=self.prior_par['mu_prior_mean'].get(),
            prior_info=self.prior_par['mu_prior_info'].get(),
            e_obs=e_mu, var_obs=var_mu)

        e_log_p_tau = ef.gamma_prior(
            prior_shape=self.prior_par['tau_prior_alpha'].get(),
            prior_rate=self.prior_par['tau_prior_beta'].get(),
            e_obs=e_tau, e_log_obs=e_log_tau)

        return \
            ef.univariate_normal_entropy(info_mu) + \
            ef.univariate_normal_entropy(info_beta) + \
            ef.gamma_entropy(tau_shape, tau_rate) + \
            e_log_p_beta + e_log_p_mu + e_log_p_tau

    def get_group_elbo_from_vec(self, group_par_vec, group):
        self.group_par.set_vector(group_par_vec)
        return self.get_group_elbo(group)

    def get_global_elbo_from_vec(self, global_par_vec, group):
        self.global_par.set_vector(global_par_vec)
        return self.get_global_elbo()

    def get_elbo_from_vec(self, par_vec):
        self.glmm_par.set_vector(par_vec)
        return self.get_elbo()

    def get_sparse_vector_hessian(self, print_every=20):
        print('Calculating global hessian:')
        sparse_global_hess = get_sparse_hessian(
            set_parameters_fun = self.set_global_parameters,
            get_group_hessian = self.get_global_vector_hessian,
            group_range = range(1),
            full_hess_dim = self.glmm_par.vector_size(),
            print_every = 1)

        print('Calculating local hessian:')
        NG = np.max(self.y_g_vec) + 1
        sparse_group_hess = get_sparse_hessian(
            set_parameters_fun = self.set_group_parameters,
            get_group_hessian = self.get_group_vector_hessian,
            group_range = range(NG),
            full_hess_dim = self.glmm_par.vector_size(),
            print_every = print_every)

        return sparse_group_hess + sparse_global_hess

    def get_free_hessian(self, vector_hess):
        vector_grad = self.get_vector_grad(self.glmm_par.get_vector())
        return convert_vector_to_free_hessian(
            self.glmm_par,
            self.glmm_par.get_free(),
            vector_grad,
            vector_hess)
