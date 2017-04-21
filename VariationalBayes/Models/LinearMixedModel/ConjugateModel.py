from VariationalBayes import ScalarParam, ModelParamsDict, VectorParam, PosDefMatrixParam
from VariationalBayes.NormalParams import MVNParam, UVNParam, UVNParamVector
from VariationalBayes.GammaParams import GammaParam
from VariationalBayes.ExponentialFamilies import \
    UnivariateNormalEntropy, MultivariateNormalEntropy, \
    GammaEntropy, MVNPrior, UVNPrior, GammaPrior

from autograd import grad, hessian, jacobian, hessian_vector_product
import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy as asp
import scipy as sp

import copy

def get_base_prior_parameters(K):
    prior_par = ModelParamsDict('Prior Parameters')
    prior_par.push_param(VectorParam('beta_mean', K, val=np.full(K, 0.0)))
    prior_par.push_param(PosDefMatrixParam('beta_info', K, val=0.01 * np.eye(K)))

    prior_par.push_param(ScalarParam('mu_mean', val=0.))
    prior_par.push_param(ScalarParam('mu_info', val=0.5))

    prior_par.push_param(ScalarParam('mu_info_shape', val=0.2))
    prior_par.push_param(ScalarParam('mu_info_rate', val=0.2))

    prior_par.push_param(ScalarParam('y_info_shape', val=0.1))
    prior_par.push_param(ScalarParam('y_info_rate', val=0.1))

    return prior_par


def get_base_sufficient_parameters(K, NG):
    lmm_par = ModelParamsDict('LMM Parameters')

    lmm_par.push_param(MVNParam('beta', K))
    lmm_par.push_param(UVNParam('mu'))
    lmm_par.push_param(GammaParam('mu_info'))
    lmm_par.push_param(GammaParam('y_info'))
    lmm_par.push_param(UVNParamVector('u', NG))

    lmm_par['beta'].mean.set(np.full(K, 0.0))
    lmm_par['beta'].info.set(np.eye(K))

    lmm_par['mu'].mean.set(0.)
    lmm_par['mu'].info.set(1.)

    lmm_par['mu_info'].shape.set(1.)
    lmm_par['mu_info'].rate.set(1.)

    lmm_par['y_info'].shape.set(1.)
    lmm_par['y_info'].rate.set(1.)

    lmm_par['u'].mean.set(np.full(NG, 0.0))
    lmm_par['u'].info.set(np.full(NG, 1.0))

    return lmm_par


def get_base_moment_parameters(K, NG):
    moment_par = ModelParamsDict('Moment Parameters')
    moment_par.push_param(VectorParam('e_beta', K))
    moment_par.push_param(PosDefMatrixParam('e_beta_outer', K))
    moment_par.push_param(ScalarParam('e_mu'))
    moment_par.push_param(ScalarParam('e_mu2'))
    moment_par.push_param(ScalarParam('e_mu_info'))
    moment_par.push_param(ScalarParam('e_log_mu_info'))
    moment_par.push_param(ScalarParam('e_y_info'))
    moment_par.push_param(ScalarParam('e_log_y_info'))
    moment_par.push_param(VectorParam('e_u', NG))
    moment_par.push_param(VectorParam('e_u2', NG))
    return moment_par


def set_moments(lmm_par, moment_par):
    moment_par['e_beta'].set(lmm_par['beta'].e())
    moment_par['e_beta_outer'].set(lmm_par['beta'].e_outer())
    moment_par['e_mu'].set(lmm_par['mu'].e())
    moment_par['e_mu2'].set(lmm_par['mu'].e_outer())
    moment_par['e_u'].set(lmm_par['u'].e())
    moment_par['e_u2'].set(lmm_par['u'].e_outer())
    moment_par['e_mu_info'].set(lmm_par['mu_info'].e())
    moment_par['e_log_mu_info'].set(lmm_par['mu_info'].e_log())
    moment_par['e_y_info'].set(lmm_par['y_info'].e())
    moment_par['e_log_y_info'].set(lmm_par['y_info'].e_log())


class LMMDataCache(object):
    def __init__(self, x_mat, y_vec, y_g_vec):
        self.x_mat = x_mat
        self.y_vec = y_vec
        self.y_g_vec = y_g_vec
        self.y_t_y = np.dot(np.transpose(y_vec), y_vec)
        self.y_t_x = np.dot(np.transpose(y_vec), x_mat)
        self.x_t_x = np.dot(np.transpose(x_mat), x_mat)

        num_g = np.max(y_g_vec)
        k = x_mat.shape[1]
        num_g = np.max(y_g_vec) + 1
        self.num_g = num_g
        self.n_g = np.full(num_g, 0.0)
        self.y_sum_g = np.full(num_g, 0.0)
        self.x_sum_g = np.full((num_g, k), 0.0)
        for g in range(num_g):
            g_rows = np.array(y_g_vec) == g
            self.n_g[g] = np.sum(g_rows)
            self.y_sum_g[g] = np.sum(y_vec[g_rows])
            self.x_sum_g[g, :] = np.sum(x_mat[g_rows, :], 0)


def get_e_log_prior(moment_par, prior_par):
    e_beta = moment_par['e_beta'].get()
    e_beta_outer = moment_par['e_beta_outer'].get()

    e_mu = moment_par['e_mu'].get()
    e_mu2 = moment_par['e_mu2'].get()

    e_mu_info = moment_par['e_mu_info'].get()
    e_log_mu_info = moment_par['e_log_mu_info'].get()

    e_y_info = moment_par['e_y_info'].get()
    e_log_y_info = moment_par['e_log_y_info'].get()

    cov_beta = e_beta_outer - np.matmul(e_beta, e_beta.transpose())
    var_mu = e_mu2 - e_mu**2

    return \
        MVNPrior(prior_mean=prior_par['beta_mean'].get(),
                 prior_info=prior_par['beta_info'].get(),
                 e_obs=e_beta,
                 cov_obs=cov_beta) + \
        UVNPrior(prior_mean=prior_par['mu_mean'].get(),
                 prior_info=prior_par['mu_info'].get(),
                 e_obs=e_mu,
                 var_obs=var_mu) + \
        GammaPrior(prior_shape=prior_par['mu_info_shape'].get(),
                   prior_rate=prior_par['mu_info_rate'].get(),
                   e_obs=e_mu_info,
                   e_log_obs=e_log_mu_info) + \
        GammaPrior(prior_shape=prior_par['y_info_shape'].get(),
                   prior_rate=prior_par['y_info_rate'].get(),
                   e_obs=e_y_info,
                   e_log_obs=e_log_y_info)


def get_e_random_effect_log_lik(moment_par):
    e_mu = moment_par['e_mu'].get()
    e_mu2 = moment_par['e_mu2'].get()

    e_u = moment_par['e_u'].get()
    e_u2 = moment_par['e_u2'].get()

    e_mu_info = moment_par['e_mu_info'].get()
    e_log_mu_info = moment_par['e_log_mu_info'].get()

    return -0.5 * e_mu_info * np.sum(e_u2 - 2 * e_u * e_mu + e_mu2) + \
           0.5 * len(e_u) * e_log_mu_info


def e_data_log_lik(data_cache, moment_par):
    e_beta = moment_par['e_beta'].get()
    e_beta_outer = moment_par['e_beta_outer'].get()

    e_u = moment_par['e_u'].get()
    e_u2 = moment_par['e_u2'].get()

    e_y_info = moment_par['e_y_info'].get()
    e_log_y_info = moment_par['e_log_y_info'].get()

    ll_global_term = \
        data_cache.y_t_y + \
        -2 * np.matmul(data_cache.y_t_x, e_beta) + \
        np.trace(np.matmul(data_cache.x_t_x, e_beta_outer))

    ll_group_term = np.sum(
        e_u2 * data_cache.n_g + \
        -2 * e_u * data_cache.y_sum_g + \
        2 * e_u * np.matmul(data_cache.x_sum_g, e_beta))

    return -0.5 * e_y_info * (ll_global_term + ll_group_term) + \
           0.5 * len(data_cache.y_vec) * e_log_y_info


def get_elbo_model_term(data_cache, moment_par, prior_par):
    ll_data = e_data_log_lik(data_cache, moment_par)
    if np.isnan(ll_data):
        print 'bad data log likelihood'
        return -np.inf

    ll_rf = get_e_random_effect_log_lik(moment_par)
    if np.isnan(ll_rf):
        print 'bad random effect log likelihood'
        return -np.inf

    e_log_prior = get_e_log_prior(moment_par, prior_par)
    if np.isnan(e_log_prior):
        print 'bad prior'
        return -np.inf

    return ll_data + ll_rf + e_log_prior


def get_entropy(lmm_par):
    return MultivariateNormalEntropy(lmm_par['beta'].info.get()) + \
           UnivariateNormalEntropy(lmm_par['mu'].info.get()) + \
           UnivariateNormalEntropy(lmm_par['u'].info.get()) + \
           GammaEntropy(shape=lmm_par['y_info'].shape.get(),
                        rate=lmm_par['y_info'].rate.get()) + \
           GammaEntropy(shape=lmm_par['mu_info'].shape.get(),
                        rate=lmm_par['mu_info'].rate.get())


def get_elbo(data_cache, lmm_par, moment_par, prior_par):
    return get_elbo_model_term(data_cache, moment_par, prior_par) + \
           get_entropy(lmm_par)


###########################
# Stuff for optimizing.

class CoordinateAscentUpdater(object):
    def __init__(self, moment_par, data_cache, prior_par):
        self.moment_par = copy.deepcopy(moment_par)
        self.__data_cache = copy.deepcopy(data_cache)
        self.__prior_par = copy.deepcopy(prior_par)

        # Coefficient functions
        self.get_e_beta_coeff = grad(self.beta_e_log_data, 0)
        self.get_e_beta_outer_coeff = grad(self.beta_e_log_data, 1)
        self.get_e_mu_coeff = grad(self.mu_e_log_data, 0)
        self.get_e_mu2_coeff = grad(self.mu_e_log_data, 1)
        self.get_e_u_coeff = grad(self.u_e_log_data, 0)
        self.get_e_u2_coeff = grad(self.u_e_log_data, 1)
        self.get_e_mu_info_coeff = grad(self.mu_info_e_log_data, 0)
        self.get_e_log_mu_info_coeff = grad(self.mu_info_e_log_data, 1)
        self.get_e_y_info_coeff = grad(self.y_info_e_log_data, 0)
        self.get_e_log_y_info_coeff = grad(self.y_info_e_log_data, 1)

    # beta updates
    def beta_e_log_data(self, e_beta, e_beta_outer):
        self.moment_par['e_beta'].set(e_beta)
        self.moment_par['e_beta_outer'].set(e_beta_outer)
        return get_elbo_model_term(self.__data_cache, self.moment_par, self.__prior_par)

    def update_beta(self):
        e_beta = self.moment_par['e_beta'].get()
        e_beta_outer = self.moment_par['e_beta_outer'].get()
        e_beta_coeff = self.get_e_beta_coeff(e_beta, e_beta_outer)
        e_beta_outer_coeff = self.get_e_beta_outer_coeff(e_beta, e_beta_outer)
        e_beta_outer_coeff = 0.5 * (e_beta_outer_coeff + e_beta_outer_coeff.transpose())
        new_cov_beta = -0.5 * np.linalg.inv(e_beta_outer_coeff)
        new_e_beta = np.matmul(new_cov_beta, e_beta_coeff)
        self.moment_par['e_beta'].set(new_e_beta)
        new_e_beta_outer = new_cov_beta + np.outer(new_e_beta, new_e_beta)
        new_e_beta_outer = 0.5 * (new_e_beta_outer + new_e_beta_outer.transpose())
        self.moment_par['e_beta_outer'].set(new_e_beta_outer)

    # mu updates
    def mu_e_log_data(self, e_mu, e_mu2):
        self.moment_par['e_mu'].set(e_mu)
        self.moment_par['e_mu2'].set(e_mu2)
        return get_elbo_model_term(self.__data_cache, self.moment_par, self.__prior_par)

    def update_mu(self):
        e_mu = self.moment_par['e_mu'].get()
        e_mu2 = self.moment_par['e_mu2'].get()
        e_mu_coeff = self.get_e_mu_coeff(e_mu, e_mu2)
        e_mu2_coeff = self.get_e_mu2_coeff(e_mu, e_mu2)
        new_var_mu = -0.5 / e_mu2_coeff
        new_e_mu = new_var_mu * e_mu_coeff
        self.moment_par['e_mu'].set(new_e_mu)
        self.moment_par['e_mu2'].set(new_var_mu + new_e_mu**2)

    # u
    def u_e_log_data(self, e_u, e_u2):
        self.moment_par['e_u'].set(e_u)
        self.moment_par['e_u2'].set(e_u2)
        return get_elbo_model_term(self.__data_cache, self.moment_par, self.__prior_par)

    def update_u(self):
        e_u = self.moment_par['e_u'].get()
        e_u2 = self.moment_par['e_u2'].get()
        e_u_coeff = self.get_e_u_coeff(e_u, e_u2)
        e_u2_coeff = self.get_e_u2_coeff(e_u, e_u2)
        new_var_u = -0.5 / e_u2_coeff
        new_e_u = new_var_u * e_u_coeff
        self.moment_par['e_u'].set(new_e_u)
        self.moment_par['e_u2'].set(new_var_u + new_e_u**2)

    # mu_info
    def mu_info_e_log_data(self, e_mu_info, e_log_mu_info):
        self.moment_par['e_mu_info'].set(e_mu_info)
        self.moment_par['e_log_mu_info'].set(e_log_mu_info)
        return get_elbo_model_term(self.__data_cache, self.moment_par, self.__prior_par)

    def update_mu_info(self):
        e_mu_info = self.moment_par['e_mu_info'].get()
        e_log_mu_info = self.moment_par['e_log_mu_info'].get()
        new_rate = -1 * self.get_e_mu_info_coeff(e_mu_info, e_log_mu_info)
        new_shape = self.get_e_log_mu_info_coeff(e_mu_info, e_log_mu_info) + 1
        self.moment_par['e_mu_info'].set(new_shape / new_rate)
        self.moment_par['e_log_mu_info'].set(asp.special.digamma(new_shape) - np.log(new_rate))

    # y_info
    def y_info_e_log_data(self, e_y_info, e_log_y_info):
        self.moment_par['e_y_info'].set(e_y_info)
        self.moment_par['e_log_y_info'].set(e_log_y_info)
        return get_elbo_model_term(self.__data_cache, self.moment_par, self.__prior_par)

    def update_y_info(self):
        e_y_info = self.moment_par['e_y_info'].get()
        e_log_y_info = self.moment_par['e_log_y_info'].get()
        new_rate = -1 * self.get_e_y_info_coeff(e_y_info, e_log_y_info)
        new_shape = self.get_e_log_y_info_coeff(e_y_info, e_log_y_info) + 1
        self.moment_par['e_y_info'].set(new_shape / new_rate)
        self.moment_par['e_log_y_info'].set(asp.special.digamma(new_shape) - np.log(new_rate))

    # Update and return the sum of absolute differences.
    def update(self):
        initial_moment_vec = self.moment_par.get_vector()
        self.update_beta()
        self.update_mu()
        self.update_mu_info()
        self.update_y_info()
        self.update_u()
        return np.sum(np.abs(initial_moment_vec - self.moment_par.get_vector()))


class KLWrapper(object):
    # Optimize with KL because python optimization seems to prefer minimizing.
    def __init__(self, lmm_par, moment_par, prior_par, x_mat, y_vec, y_g_vec, num_draws):
        self.__lmm_par_ad = copy.deepcopy(lmm_par)
        self.__prior_par_ad = copy.deepcopy(prior_par)
        self.__moment_par_ad = copy.deepcopy(moment_par)
        self.__data_cache = LMMDataCache(x_mat, y_vec, y_g_vec)
        self.kl_grad = grad(self.kl)
        self.kl_hess = hessian(self.kl)
        self.kl_hvp = hessian_vector_product(self.kl)

    def kl(self, free_par_vec, verbose=False):
        self.__lmm_par_ad.set_free(free_par_vec)
        set_moments(self.__lmm_par_ad, self.__moment_par_ad)

        #print self.__lmm_par_ad
        kl = -get_elbo(self.__data_cache,
                       self.__lmm_par_ad,
                       self.__moment_par_ad,
                       self.__prior_par_ad)[0]
        if verbose: print kl

        return kl


#################################
# Stuff for LRVB

class MomentWrapper(object):
    def __init__(self, lmm_par, moment_par):
        self.__lmm_par_ad = copy.deepcopy(lmm_par)
        self.__moment_par = copy.deepcopy(moment_par)
        self.moment_jacobian = jacobian(self.get_moments)

    # Return a posterior moment of interest as a function of unconstrained parameters.
    def get_moments(self, free_par_vec):
        self.__lmm_par_ad.set_free(free_par_vec)
        set_moments(self.__lmm_par_ad, self.__moment_par)
        return self.__moment_par.get_vector()

    def get_moment_parameters(self, free_par_vec):
        self.__glmm_par_ad.set_free(free_par_vec)
        set_moments(self.__glmm_par_ad, self.__moment_par)
        return self.__moment_par






# TODO: put these in tests.

# Make sure it's doing something, or not, depending on whether you use
# the initial point or optimum.

# moment_par_test = copy.deepcopy(moment_par_opt)
# moment_par_test = copy.deepcopy(moment_par)

# ca_updater = CoordinateAscentUpdater(moment_par_test, data_cache, prior_par)
# print ca_updater.moment_par['e_beta']
# print ca_updater.moment_par['e_beta_outer']
# ca_updater.update_beta()
# print ca_updater.moment_par['e_beta']
# print ca_updater.moment_par['e_beta_outer']

# ca_updater = CoordinateAscentUpdater(moment_par_test, data_cache, prior_par)
# print ca_updater.moment_par['e_mu']
# print ca_updater.moment_par['e_mu2']
# ca_updater.update_mu()
# print ca_updater.moment_par['e_mu']
# print ca_updater.moment_par['e_mu2']

# ca_updater = CoordinateAscentUpdater(moment_par_test, data_cache, prior_par)
# print ca_updater.moment_par['e_u']
# print ca_updater.moment_par['e_u2']
# ca_updater.update_u()
# print ca_updater.moment_par['e_u']
# print ca_updater.moment_par['e_u2']

# ca_updater = CoordinateAscentUpdater(moment_par_test, data_cache, prior_par)
# print ca_updater.moment_par['e_mu_info']
# print ca_updater.moment_par['e_log_mu_info']
# ca_updater.update_mu_info()
# print ca_updater.moment_par['e_mu_info']
# print ca_updater.moment_par['e_log_mu_info']

# ca_updater = CoordinateAscentUpdater(moment_par_test, data_cache, prior_par)
# print ca_updater.moment_par['e_y_info']
# print ca_updater.moment_par['e_log_y_info']
# ca_updater.update_y_info()
# print ca_updater.moment_par['e_y_info']
# print ca_updater.moment_par['e_log_y_info']
