
import VariationalBayes as vb
import VariationalBayes.ExponentialFamilies as ef
import VariationalBayes.Modeling as modeling

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
        e_log_p_beta = ef.mvn_prior(
            beta_prior_mean, beta_prior_info, e_beta, cov_beta)

        e_mu = self.glmm_par['mu'].mean.get()
        info_mu = self.glmm_par['mu'].info.get()
        var_mu = 1 / info_mu
        e_log_p_mu = ef.uvn_prior(
            self.prior_par['mu_prior_mean'].get(),
            self.prior_par['mu_prior_info'].get(), e_mu, var_mu)

        e_tau = self.glmm_par['tau'].e()
        e_log_tau = self.glmm_par['tau'].e_log()
        tau_prior_shape = self.prior_par['tau_prior_alpha'].get()
        tau_prior_rate = self.prior_par['tau_prior_beta'].get()
        e_log_p_tau = ef.gamma_prior(
            tau_prior_shape, tau_prior_rate, e_tau, e_log_tau)

        return e_log_p_beta + e_log_p_mu + e_log_p_tau

    def get_data_log_lik_terms(self):
        e_beta = self.glmm_par['beta'].e()
        #cov_beta = self.glmm_par['beta'].cov()
        cov_beta = np.diag(self.glmm_par['beta'].var())
        e_u = self.glmm_par['u'].e()
        var_u = self.glmm_par['u'].var()

        # Log likelihood from data.
        z_mean = e_u[self.y_g_vec] + np.matmul(self.x_mat, e_beta)
        z_sd = np.sqrt(
            var_u[self.y_g_vec] + np.einsum('nk,kj,nj->n',
                              self.x_mat, cov_beta, self.x_mat))

        return \
            self.y_vec * z_mean - \
            get_e_logistic_term_guass_hermite(
                z_mean, z_sd, self.gh_x, self.gh_w, aggregate_all=False)

    def get_log_lik(self):
        if self.use_weights:
            log_lik = np.sum(self.weights * self.get_data_log_lik_terms())
        else:
            log_lik = np.sum(self.get_data_log_lik_terms())

        # Log likelihood from random effect terms.
        e_u = self.glmm_par['u'].e()
        var_u = self.glmm_par['u'].var()

        e_mu = self.glmm_par['mu'].e()
        var_mu = self.glmm_par['mu'].var()
        e_tau = self.glmm_par['tau'].e()
        e_log_tau = self.glmm_par['tau'].e_log()

        log_lik += -0.5 * e_tau * np.sum(
            ((e_mu - e_u) ** 2) + var_mu + var_u) + \
            0.5 * e_log_tau * len(e_u)

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

        # return \
        #     ef.univariate_normal_entropy(info_mu) + \
        #     ef.multivariate_normal_entropy(info_beta) + \
        #     ef.univariate_normal_entropy(info_u) + \
        #     ef.gamma_entropy(tau_shape, tau_rate)

    def get_kl(self):
        return -1 * np.squeeze(
            self.get_log_lik() + self.get_entropy() + self.get_e_log_prior())


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

        #self.moment_par.push_param(
        #    vb.PosDefMatrixParam('e_beta_outer', K))
        #self.moment_par.push_param(vb.ScalarParam('e_mu2'))
        #self.moment_par.push_param(vb.VectorParam('e_u2', NG))

    def __str__(self):
        return str(self.moment_par)

    def set_moments(self, free_par_vec):
        self.glmm_par.set_free(free_par_vec)
        self.moment_par['e_beta'].set(self.glmm_par['beta'].e())
        self.moment_par['e_mu'].set(self.glmm_par['mu'].e())
        self.moment_par['e_tau'].set(self.glmm_par['tau'].e())
        self.moment_par['e_log_tau'].set(self.glmm_par['tau'].e_log())
        self.moment_par['e_u'].set(self.glmm_par['u'].e())

        #self.moment_par['e_beta_outer'].set(self.glmm_par['beta'].e_outer())
        #self.moment_par['e_mu2'].set(self.glmm_par['mu'].e_outer())
        #self.moment_par['e_u2'].set((self.glmm_par['u'].e_outer()))

    # Return a posterior moment of interest as a function of unconstrained parameters.
    def get_moment_vector(self, free_par_vec):
        self.set_moments(free_par_vec)
        return self.moment_par.get_vector()



    # def ExpectedLogPrior(self, free_par_vec, prior_par_vec):
    #     # Encode the glmm parameters first and the prior second.
    #     self.__glmm_par_ad.set_free(free_par_vec)
    #     self.__prior_par_ad.set_vector(prior_par_vec)
    #     e_log_prior = ELogPrior(self.__prior_par_ad, self.__glmm_par_ad)
    #     return e_log_prior[0]
