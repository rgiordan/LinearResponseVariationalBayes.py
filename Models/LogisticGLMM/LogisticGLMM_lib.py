
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

import json
import copy

def load_json_data(json_filename):
    json_file = open(json_filename, 'r')
    json_dat = json.load(json_file)
    json_file.close()

    stan_dat = json_dat['stan_dat']

    print(stan_dat.keys())
    K = stan_dat['K'][0]
    NObs = stan_dat['N'][0]
    NG = stan_dat['NG'][0]

    y_g_vec = np.array(stan_dat['y_group'])
    y_vec = np.array(stan_dat['y'])
    x_mat = np.array(stan_dat['x'])

    glmm_par = get_glmm_parameters(K=K, NG=NG)

    # Define a class to contain prior parameters.
    prior_par = get_default_prior_params(K)
    prior_par['beta_prior_mean'].set(np.array(stan_dat['beta_prior_mean']))

    prior_par['beta_prior_info'].set(np.array(stan_dat['beta_prior_info']))

    prior_par['mu_prior_mean'].set(stan_dat['mu_prior_mean'][0])
    prior_par['mu_prior_info'].set(stan_dat['mu_prior_info'][0])

    prior_par['tau_prior_alpha'].set(stan_dat['tau_prior_alpha'][0])
    prior_par['tau_prior_beta'].set(stan_dat['tau_prior_beta'][0])

    # An index set to make sure jacobians match the order expected by R.
    prior_par_indices = copy.deepcopy(prior_par)
    prior_par_indices.set_name('Prior Indices')
    prior_par_indices.set_vector(np.array(range(prior_par_indices.vector_size())))

    return y_g_vec, y_vec, x_mat, glmm_par, prior_par


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

    return np.array(x_mat), np.array(y_g_vec), np.array(y_vec), true_rho, true_u


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
def get_data_log_lik_terms(glmm_par, x_mat, y_vec, y_g_vec, gh_x, gh_w):
    e_beta = glmm_par['beta'].e()
    var_beta = glmm_par['beta'].var()

    # atleast_1d is necessary for indexing by y_g_vec to work right.
    e_u = np.atleast_1d(glmm_par['u'].e())
    var_u = np.atleast_1d(glmm_par['u'].var())

    # Log likelihood from data.
    z_mean = e_u[y_g_vec] + np.squeeze(np.matmul(x_mat, e_beta))
    z_sd = np.sqrt(
        var_u[y_g_vec] +
        np.squeeze(np.einsum('nk,k,nk->n', x_mat, var_beta, x_mat)))
    return \
        y_vec * z_mean - \
        get_e_logistic_term_guass_hermite(
            z_mean, z_sd, gh_x, gh_w, aggregate_all=False)


def get_re_log_lik(glmm_par):
    e_mu = glmm_par['mu'].e()
    var_mu = glmm_par['mu'].var()
    e_tau = glmm_par['tau'].e()
    e_log_tau = glmm_par['tau'].e_log()
    e_u = glmm_par['u'].e()
    var_u = glmm_par['u'].var()

    return -0.5 * e_tau * np.sum(
        ((e_mu - e_u) ** 2) + var_mu + var_u) + \
        0.5 * e_log_tau * glmm_par['u'].size()

def get_global_entropy(glmm_par):
    info_mu = glmm_par['mu'].info.get()
    info_beta = glmm_par['beta'].info.get()
    tau_shape = glmm_par['tau'].shape.get()
    tau_rate = glmm_par['tau'].rate.get()

    return \
        ef.univariate_normal_entropy(info_mu) + \
        ef.univariate_normal_entropy(info_beta) + \
        ef.gamma_entropy(tau_shape, tau_rate)

def get_local_entropy(glmm_par):
    info_u = glmm_par['u'].info.get()
    return ef.univariate_normal_entropy(info_u)


def get_e_log_prior(glmm_par, prior_par):
    e_beta = glmm_par['beta'].mean.get()
    info_beta = glmm_par['beta'].info.get()
    #cov_beta = np.linalg.inv(info_beta)
    cov_beta = np.diag(1. / info_beta)
    beta_prior_info = prior_par['beta_prior_info'].get()
    beta_prior_mean = prior_par['beta_prior_mean'].get()
    e_mu = glmm_par['mu'].mean.get()
    info_mu = glmm_par['mu'].info.get()
    var_mu = 1 / info_mu
    e_tau = glmm_par['tau'].e()
    e_log_tau = glmm_par['tau'].e_log()

    e_log_p_beta = ef.mvn_prior(
        prior_mean = prior_par['beta_prior_mean'].get(),
        prior_info = prior_par['beta_prior_info'].get(),
        e_obs = e_beta,
        cov_obs = cov_beta)

    e_log_p_mu = ef.uvn_prior(
        prior_mean = prior_par['mu_prior_mean'].get(),
        prior_info = prior_par['mu_prior_info'].get(),
        e_obs = e_mu,
        var_obs = var_mu)

    e_log_p_tau = ef.gamma_prior(
        prior_shape = prior_par['tau_prior_alpha'].get(),
        prior_rate = prior_par['tau_prior_beta'].get(),
        e_obs = e_tau,
        e_log_obs = e_log_tau)

    return e_log_p_beta + e_log_p_mu + e_log_p_tau


class LogisticGLMM(object):
    def __init__(
        self, glmm_par, prior_par, x_mat, y_vec, y_g_vec, num_gh_points):

        self.glmm_par = copy.deepcopy(glmm_par)
        self.prior_par = copy.deepcopy(prior_par)
        self.x_mat = np.array(x_mat)
        self.y_vec = np.array(y_vec)
        self.y_g_vec = np.array(y_g_vec)
        self.set_gh_points(num_gh_points)

        self.use_weights = False
        self.weights = np.full(self.x_mat.shape[0], 1.0)

        assert np.min(y_g_vec) == 0
        assert np.max(y_g_vec) == self.glmm_par['u'].size() - 1

    def set_gh_points(self, num_gh_points):
        self.gh_x, self.gh_w = onp.polynomial.hermite.hermgauss(num_gh_points)

    def get_e_log_prior(self):
        return get_e_log_prior(self.glmm_par, self.prior_par)

    def get_data_log_lik_terms(self):
        return get_data_log_lik_terms(
            glmm_par = self.glmm_par,
            x_mat = self.x_mat,
            y_vec = self.y_vec,
            y_g_vec = self.y_g_vec,
            gh_x = self.gh_x,
            gh_w = self.gh_w)

    def get_log_lik(self):
        if self.use_weights:
            data_log_lik = np.sum(self.weights * self.get_data_log_lik_terms())
        else:
            data_log_lik = np.sum(self.get_data_log_lik_terms())

        # Log likelihood from random effect terms.
        re_log_lik = get_re_log_lik(self.glmm_par)

        return data_log_lik + re_log_lik

    def get_entropy(self):
        return get_global_entropy(self.glmm_par) + \
               get_local_entropy(self.glmm_par)

    def get_elbo(self):
        log_lik = self.get_log_lik()
        entropy = self.get_entropy()
        e_log_prior = self.get_e_log_prior()
        return np.squeeze(log_lik + entropy + e_log_prior)

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
def get_group_parameters(K, num_groups=1):
    group_par = vb.ModelParamsDict('Single group GLMM parameters')
    group_par.push_param(vb.UVNParam('mu'))
    group_par.push_param(vb.GammaParam('tau'))
    group_par.push_param(vb.UVNParamVector('beta', K))
    group_par.push_param(vb.UVNParamVector('u', num_groups))
    return group_par

# Since we never use the free version of the global parameters, we don't need to
# set the minimum allowable values.
def get_global_parameters(K):
    global_par = vb.ModelParamsDict('Global GLMM parameters')
    global_par.push_param(vb.UVNParam('mu'))
    global_par.push_param(vb.GammaParam('tau'))
    global_par.push_param(vb.UVNParamVector('beta', K))
    return global_par

def set_group_parameters(glmm_par, group_par, groups):
    # Stupid assert fails with integers.  Why can't an integer just have
    # len() == 1?
    assert(len(groups) == group_par['u'].size())
    group_par['beta'].set_vector(glmm_par['beta'].get_vector())
    group_par['mu'].set_vector(glmm_par['mu'].get_vector())
    group_par['tau'].set_vector(glmm_par['tau'].get_vector())

    group_par['u'].mean.set(glmm_par['u'].mean.get()[groups])
    group_par['u'].info.set(glmm_par['u'].info.get()[groups])

def set_global_parameters(glmm_par, global_par):
    global_par['beta'].set_vector(glmm_par['beta'].get_vector())
    global_par['mu'].set_vector(glmm_par['mu'].get_vector())
    global_par['tau'].set_vector(glmm_par['tau'].get_vector())


class SparseModelObjective(LogisticGLMM):
    def __init__(self, glmm_par, prior_par, x_mat, y_vec, y_g_vec,
                 num_gh_points, num_groups=1):
        super().__init__(
            glmm_par, prior_par,
            np.array(x_mat),
            np.array(y_vec),
            np.array(y_g_vec), num_gh_points)

        self.glmm_indices = copy.deepcopy(self.glmm_par)
        self.glmm_indices.set_vector(np.arange(0, self.glmm_indices.vector_size()))

        # Parameters for a single observation.
        K = glmm_par['beta'].size()
        self.group_par = get_group_parameters(K, num_groups)
        self.group_indices = get_group_parameters(K, num_groups)
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

    # For a vector of groups, return the subsest of the data needed to evaluate
    # the model at those groups, including a y_g vector appropriate to a set
    # of parameters that only contains parameters for these groups.
    def get_data_for_groups(self, groups):
        # Rows in the dataset corresponding to these groups:
        all_group_rows = onp.logical_or.reduce([self.group_rows[g] for g in groups])

        # Which indices within the groups vector correspond to these rows:
        y_g_sub = np.hstack([ np.full(np.sum(self.group_rows[groups[ig]]), ig) \
                              for ig in range(len(groups))])
        return all_group_rows, y_g_sub

    # Likelihood functions:
    def get_group_elbo(self, groups):
        all_group_rows, y_g_sub = self.get_data_for_groups(groups)

        data_log_lik = np.sum(get_data_log_lik_terms(
            glmm_par = self.group_par,
            y_g_vec = y_g_sub,
            x_mat = self.x_mat[all_group_rows, :],
            y_vec = self.y_vec[all_group_rows],
            gh_x = self.gh_x,
            gh_w = self.gh_w))
        re_log_lik = get_re_log_lik(self.group_par)
        u_entropy = get_local_entropy(self.group_par)

        return np.squeeze(data_log_lik + re_log_lik + u_entropy)

    def get_global_elbo(self):
        return np.squeeze(
            get_global_entropy(self.global_par) + \
            get_e_log_prior(self.global_par, self.prior_par))

    def get_group_elbo_from_vec(self, group_par_vec, group):
        self.group_par.set_vector(group_par_vec)
        return self.get_group_elbo(group)

    def get_global_elbo_from_vec(self, global_par_vec, group):
        self.global_par.set_vector(global_par_vec)
        return self.get_global_elbo()

    def get_elbo_from_vec(self, par_vec):
        self.glmm_par.set_vector(par_vec)
        return self.get_elbo()

    def get_sparse_vector_hessian(self, print_every_n):
        print('Calculating global hessian:')
        sparse_global_hess = get_sparse_hessian(
            set_parameters_fun = self.set_global_parameters,
            get_group_hessian = self.get_global_vector_hessian,
            group_range = [0],
            full_hess_dim = self.glmm_par.vector_size(),
            print_every = 1)

        print('Calculating local hessian:')
        NG = np.max(self.y_g_vec) + 1
        sparse_group_hess = get_sparse_hessian(
            set_parameters_fun = self.set_group_parameters,
            get_group_hessian = self.get_group_vector_hessian,
            group_range = [[g] for g in range(NG)],
            full_hess_dim = self.glmm_par.vector_size(),
            print_every = print_every_n)

        return sparse_group_hess + sparse_global_hess

    def get_free_hessian(self, vector_hess):
        vector_grad = self.get_vector_grad(self.glmm_par.get_vector())
        return convert_vector_to_free_hessian(
            self.glmm_par,
            self.glmm_par.get_free(),
            vector_grad,
            vector_hess)


###################################
# MLE (MAP) estimators


def get_mle_parameters(K, NG):
    mle_par = vb.ModelParamsDict('GLMER Parameters')
    mle_par.push_param(vb.VectorParam('mu'))
    mle_par.push_param(vb.VectorParam('tau'))
    mle_par.push_param(vb.VectorParam('beta', K))
    mle_par.push_param(vb.VectorParam('u', NG))

    return mle_par

def get_mle_data_log_lik_terms(mle_par, x_mat, y_vec, y_g_vec):
    beta = mle_par['beta'].get()

    # atleast_1d is necessary for indexing by y_g_vec to work right.
    e_u = np.atleast_1d(mle_par['u'].get())

    # Log likelihood from data.
    z = e_u[y_g_vec] + np.squeeze(np.matmul(x_mat, beta))

    return y_vec * z - np.log1p(np.exp(z))

def get_mle_re_log_lik(mle_par):
    mu = mle_par['mu'].get()
    tau = mle_par['tau'].get()
    u = mle_par['u'].get()

    return -0.5 * tau * np.sum(
        ((mu - u) ** 2)) +  0.5 * np.log(tau) * mle_par['u'].size()

def get_mle_log_prior(mle_par, prior_par):
    beta = mle_par['beta'].get()
    mu = mle_par['mu'].get()
    tau = mle_par['tau'].get()

    K = len(beta)
    log_p_beta = ef.mvn_prior(
        prior_mean = prior_par['beta_prior_mean'].get(),
        prior_info = prior_par['beta_prior_info'].get(),
        e_obs = beta,
        cov_obs = np.zeros((K, K)))

    log_p_mu = ef.uvn_prior(
        prior_mean = prior_par['mu_prior_mean'].get(),
        prior_info = prior_par['mu_prior_info'].get(),
        e_obs = mu,
        var_obs = 0.0)

    log_p_tau = ef.gamma_prior(
        prior_shape = prior_par['tau_prior_alpha'].get(),
        prior_rate = prior_par['tau_prior_beta'].get(),
        e_obs = tau,
        e_log_obs = np.log(tau))

    return log_p_beta + log_p_mu + log_p_tau


def set_moment_par_from_mle(moment_par, mle_par):
    moment_par.set_vector(np.full(moment_par.vector_size(), np.nan))
    moment_par['e_beta'].set(mle_par['beta'].get())
    moment_par['e_mu'].set(mle_par['mu'].get())
    moment_par['e_tau'].set(mle_par['tau'].get())
    moment_par['e_log_tau'].set(np.log(mle_par['tau'].get()))
    moment_par['e_u'].set(mle_par['u'].get())

    return moment_par


class LogisticGLMMMaximumLikelihood(object):
    def __init__(self, mle_par, prior_par, x_mat, y_vec, y_g_vec):

        self.mle_par = copy.deepcopy(mle_par)
        self.prior_par = copy.deepcopy(prior_par)
        self.x_mat = np.array(x_mat)
        self.y_vec = np.array(y_vec)
        self.y_g_vec = np.array(y_g_vec)

        assert np.min(y_g_vec) == 0
        assert np.max(y_g_vec) == self.mle_par['u'].size() - 1


    def get_log_lik(self):
        data_log_lik = np.sum(get_mle_data_log_lik_terms(
            mle_par = self.mle_par,
            x_mat = self.x_mat,
            y_vec = self.y_vec,
            y_g_vec = self.y_g_vec))
        re_log_lik = get_mle_re_log_lik(self.mle_par)
        log_prior = get_mle_log_prior(self.mle_par, self.prior_par)
        return np.squeeze(data_log_lik + re_log_lik + log_prior)

    def get_log_loss(self):
        return -1 * self.get_log_lik()
