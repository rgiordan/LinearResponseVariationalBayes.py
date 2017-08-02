import VariationalBayes as vb
import VariationalBayes.ExponentialFamilies as ef
import VariationalBayes.Modeling as modeling

import autograd.numpy as np
import autograd.scipy as sp
import numpy as onp

from copy import deepcopy
import scipy as osp
from scipy import stats


# y is defined as a num_i * num_j matrix.  Here, alpha and beta are num_i vectors and
# theta is the num_j vector.  Practically we expect num_j >> num_i.
# Combine vectors in the appropriate way to match the shape of y.
def get_logit_p_term(alpha, beta, theta):
    return (np.expand_dims(theta, 0) - np.expand_dims(beta, 1)) * np.expand_dims(alpha, 1)


def simulate_data(num_i, num_j):
    true_params = vb.ModelParamsDict('true')
    true_params.push_param(vb.VectorParam('alpha', size=num_i, lb=0))
    true_params.push_param(vb.VectorParam('beta', size=num_i))
    true_params.push_param(vb.VectorParam('theta', size=num_j))
    true_params.push_param(vb.VectorParam('mu', size=2))
    true_params.push_param(vb.PosDefMatrixParam('sigma', size=2))

    true_params['alpha'].set(np.exp(np.random.random(num_i)))
    true_params['beta'].set(np.random.random(num_i) - 0.5)
    true_params['theta'].set(np.random.random(num_j) - 0.5)
    true_params['mu'].set(np.random.random(2))
    true_params['sigma'].set(np.eye(2))

    logit_p = get_logit_p_term(alpha=true_params['alpha'].get(),
                               beta=true_params['beta'].get(),
                               theta=true_params['theta'].get())
    y_prob = osp.special.expit(logit_p)
    y = osp.stats.bernoulli.rvs(y_prob)

    return true_params, y, y_prob


# Get default prior parameters.
def get_prior_params():
    prior_params = vb.ModelParamsDict('prior')
    prior_params.push_param(vb.VectorParam('mu_mean', size=2, val=np.array([0., 0.])))

    # mu[0] <-> log_alpha
    # mu[1] <-> beta
    mu_prior_cov = np.array([[1., 0.], [0., 25.]])
    prior_params.push_param(vb.PosDefMatrixParam('mu_info', size=2, val=np.linalg.inv(mu_prior_cov)))
    prior_params.push_param(vb.ScalarParam('theta_mean', val=0.0))
    prior_params.push_param(vb.ScalarParam('theta_var', val=1.0))
    prior_params.push_param(vb.VectorParam('tau_param', size=2, val=np.array([0.1, 0.1])))
    prior_params.push_param(vb.ScalarParam('lkj_param', val=4.))

    return prior_params


def get_vb_params(num_i, num_j):
    vb_params = vb.ModelParamsDict('params')
    vb_params.push_param(vb.UVNParamVector('log_alpha', length=num_i))
    vb_params.push_param(vb.UVNParamVector('beta', length=num_i))
    vb_params.push_param(vb.UVNParamVector('theta', length=num_j))
    vb_params.push_param(vb.MVNParam('mu', dim=2))
    vb_params.push_param(vb.WishartParam('sigma_inv', size=2))
    return vb_params

class Model(object):
    def __init__(self, y, vb_params, prior_params, num_draws):
        self.y = deepcopy(y)
        self.vb_params = deepcopy(vb_params)
        self.prior_params = deepcopy(prior_params)
        self.std_draws = modeling.get_standard_draws(num_draws)

        self.num_i = self.vb_params['log_alpha'].mean.size()
        self.num_j = self.vb_params['theta'].mean.size()

    def get_e_log_data_likelihood(self):
        # P(y = 1) = expit(z)
        log_alpha = self.vb_params['log_alpha']
        beta = self.vb_params['beta']
        theta = self.vb_params['theta']

        e_z = get_logit_p_term(alpha=log_alpha.e_exp(), beta=beta.e(), theta=theta.e())

        # var_z = E[alpha^2] * (Var(beta) + Var(theta)) +
        #         Var(alpha) * (E[beta^2] - 2 E[beta] E[theta] + E[theta^2])

        var_alpha = np.expand_dims(log_alpha.var_exp(), 1)
        e2_alpha = np.expand_dims(log_alpha.e2_exp(), 1)

        var_beta = np.expand_dims(beta.var(), 1)
        e2_beta = np.expand_dims(beta.e_outer(), 1)
        e_beta = np.expand_dims(beta.e(), 1)

        var_theta = np.expand_dims(theta.var(), 0)
        e2_theta = np.expand_dims(theta.e_outer(), 0)
        e_theta = np.expand_dims(theta.e(), 0)

        var_z = e2_alpha * (var_beta + var_theta) + \
                var_alpha *(e2_beta - 2 * e_beta * e_theta + e2_theta)

        y_logit_term = modeling.get_e_logistic_term(self.y, e_z, np.sqrt(var_z), self.std_draws)

        return y_logit_term

    def get_e_log_hierarchy_likelihood(self):
        log_alpha = self.vb_params['log_alpha']
        beta = self.vb_params['beta']
        mu = self.vb_params['mu']
        sigma_inv = self.vb_params['sigma_inv']

        # Refer to the combined (log_alpha, beta) vector as 'ab'.
        e_ab = np.array([ log_alpha.e(), beta.e() ])
        e_outer_ab = np.array([[ log_alpha.e_outer(),      beta.e() * log_alpha.e() ],
                               [ beta.e() * log_alpha.e(), beta.e_outer() ] ])

        e_sigma_inv = sigma_inv.e()

        # TODO: are the two einsums for the cross term actually necessary?
        return -0.5 * (np.einsum('ij,ji', e_sigma_inv, mu.e_outer()) * self.num_i - \
                       np.einsum('i,ij,jn->', mu.e(), e_sigma_inv, e_ab) - \
                       np.einsum('in,ij,j->', e_ab, e_sigma_inv, mu.e()) + \
                       np.einsum('ij,jin->', e_sigma_inv, e_outer_ab)) + \
               0.5 * sigma_inv.e_log_det() * self.num_i

    def get_e_log_prior(self):
        mu = self.vb_params['mu']
        theta = self.vb_params['theta']
        sigma_inv = self.vb_params['sigma_inv']

        prior_params = self.prior_params

        e_log_prior = 0.

        # Mu
        e_log_prior += ef.mvn_prior(
            prior_params['mu_mean'].get(), prior_params['mu_info'].get(),
            mu.e(), mu.cov())

        # Theta
        e_log_prior += np.sum(ef.uvn_prior(
            prior_params['theta_mean'].get(), prior_params['theta_var'].get(),
            theta.e(), theta.var()))

        # Sigma
        e_log_prior += np.sum(ef.exponential_prior(
            prior_params['tau_param'].get(), np.diag(sigma_inv.e_inv())))
        e_log_prior += sigma_inv.e_log_lkj_inv_prior(prior_params['lkj_param'].get())

        return e_log_prior

    def get_e_log_likelihood(self):
        return \
            self.get_e_log_data_likelihood() + \
            self.get_e_log_hierarchy_likelihood() + \
            self.get_e_log_prior()

    def get_entropy(self):
        return \
            self.vb_params['log_alpha'].entropy() + \
            self.vb_params['beta'].entropy() + \
            self.vb_params['theta'].entropy() + \
            self.vb_params['mu'].entropy() + \
            self.vb_params['sigma_inv'].entropy()

    def get_kl(self):
        return -1 * (self.get_e_log_likelihood() + self.get_entropy())

    def get_negative_e_log_likelihood(self):
        # For debugging
        self.vb_params['log_alpha'].info.set(np.full(self.num_i, 100.))
        self.vb_params['beta'].info.set(np.full(self.num_i, 100.))
        self.vb_params['theta'].info.set(np.full(self.num_j, 100.))
        self.vb_params['mu'].info.set(np.eye(2) * 100.)
        self.vb_params['sigma_inv'].params['v'].set(np.eye(2) * 0.1)
        self.vb_params['sigma_inv'].params['df'].set(5.0)
        #print(self.vb_params)
        return -1 * self.get_e_log_likelihood()
