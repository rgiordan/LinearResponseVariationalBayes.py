
import VariationalBayes as vb
import VariationalBayes.ExponentialFamilies as ef
import VariationalBayes.Modeling as modeling

import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy as sp
import scipy as osp

import copy


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


class LogisticGLMM(object):
    def __init__(self, glmm_par, prior_par, x_mat, y_vec, y_g_vec, num_draws):
        self.glmm_par = copy.deepcopy(glmm_par)
        self.prior_par = copy.deepcopy(prior_par)
        self.x_mat = x_mat
        self.y_vec = y_vec
        self.y_g_vec = y_g_vec
        self.set_draws(num_draws)

    def set_draws(self, num_draws):
        self.std_draws = modeling.get_standard_draws(num_draws)

    def get_e_log_prior(self):
        e_beta = self.glmm_par['beta'].mean.get()
        info_beta = self.glmm_par['beta'].info.get()
        cov_beta = np.linalg.inv(info_beta)
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

        return  e_log_p_beta + e_log_p_mu + e_log_p_tau


    def get_log_lik(self):
        e_beta = self.glmm_par['beta'].e()
        cov_beta = self.glmm_par['beta'].cov()
        e_u = self.glmm_par['u'].e()[self.y_g_vec]
        var_u = self.glmm_par['u'].var()[self.y_g_vec]

        log_lik = 0.

        # Log likelihood from data.
        z_mean = e_u + np.matmul(self.x_mat, e_beta)
        z_sd = np.sqrt(
            var_u + np.einsum('nk,kj,nj->n',
                              self.x_mat, cov_beta, self.x_mat))

        log_lik += modeling.get_e_logistic_term(
            self.y_vec, z_mean, z_sd, self.std_draws)

        # Log likelihood from random effect terms.
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
            ef.multivariate_normal_entropy(info_beta) + \
            ef.univariate_normal_entropy(info_u) + \
            ef.gamma_entropy(tau_shape, tau_rate)

    def get_kl(self):
        return -1 * (self.get_log_lik() + self.get_entropy())



    # def ExpectedLogPrior(self, free_par_vec, prior_par_vec):
    #     # Encode the glmm parameters first and the prior second.
    #     self.__glmm_par_ad.set_free(free_par_vec)
    #     self.__prior_par_ad.set_vector(prior_par_vec)
    #     e_log_prior = ELogPrior(self.__prior_par_ad, self.__glmm_par_ad)
    #     return e_log_prior[0]
