import autograd.numpy as np
import autograd.scipy as sp
import math

def multivariate_digamma(x, size):
    x_vec = x - 0.5 * np.linspace(0, size - 1., size)
    return np.sum(sp.special.digamma(x_vec))


def multivariate_gammaln(x, size):
    x_vec = x - 0.5 * np.linspace(0, size - 1., size)
    return np.sum(sp.special.gammaln(x_vec)) + \
           0.25 * np.log(np.pi) * size * (size - 1.0);


##################################################
# Entropies.  Note that autograd hasn't defined these yet and doesn't seem to
# work with the sp.stats functions.

def univariate_normal_entropy(info_obs):
    # np.sum(sp.stats.norm.entropy(scale=np.sqrt(var_obs)))
    return 0.5 * np.sum(-1 * np.log(info_obs) + 1 + np.log(2 * math.pi))

def multivariate_normal_entropy(info_obs):
    sign, logdet = np.linalg.slogdet(info_obs)
    assert sign > 0
    k = info_obs.shape[0]
    return 0.5 * (-1 * logdet + k + k * np.log(2 * math.pi))

def gamma_entropy(shape, rate):
    return np.sum(shape - np.log(rate) + sp.special.gammaln(shape) + \
                  (1 - shape) * sp.special.digamma(shape))

# alpha can be an array of Dirichlet parameteres where dimension 0 is the
# random variable dimension.
# Returns an array of entropies.
# TODO: make the behavior consistent with other array functions --
# for example, the UVN entropy sums the entropies, and this doesn't.
# I think they should not sum by default.
def dirichlet_entropy(alpha):
    dirichlet_dim = alpha.shape[0]
    sum_alpha = np.sum(alpha, axis=0, keepdims=True)
    log_beta = np.sum(sp.special.gammaln(alpha), axis=0, keepdims=True) - \
               sp.special.gammaln(sum_alpha)
    entropy = \
        log_beta - \
        (dirichlet_dim - sum_alpha) * sp.special.digamma(sum_alpha) - \
        np.sum((alpha - 1) * sp.special.digamma(alpha), axis=0)
    return np.squeeze(entropy, axis=0)

def beta_entropy(tau):
    digamma_tau0 = sp.special.digamma(tau[:, 0])
    digamma_tau1 = sp.special.digamma(tau[:, 1])
    digamma_tausum = sp.special.digamma(np.sum(tau, 1))

    lgamma_tau0 = sp.special.gammaln(tau[:, 0])
    lgamma_tau1 = sp.special.gammaln(tau[:, 1])
    lgamma_tausum = sp.special.gammaln(np.sum(tau, 1))

    lbeta = lgamma_tau0 + lgamma_tau1 - lgamma_tausum

    return np.sum(
        lbeta - \
        (tau[:, 0] - 1.) * digamma_tau0 - \
        (tau[:, 1] - 1.) * digamma_tau1 + \
        (tau[:, 0] + tau[:, 1] - 2) * digamma_tausum)


def wishart_entropy(df, v):
    k = float(v.shape[0])
    assert v.shape[0] == v.shape[1]
    s, log_det_v = np.linalg.slogdet(v)
    assert s > 0
    return \
        0.5 * (k + 1) * log_det_v + \
        0.5 * k * (k + 1) * np.log(2) + \
        multivariate_gammaln(0.5 * df, k) - \
        0.5 * (df - k - 1) * multivariate_digamma(0.5 * df, k) + \
        0.5 * df * k

#############################
# Expectations

# If \Sigma ~ Wishart(v, df), return E[log |\Sigma|]
def e_log_det_wishart(df, v):
    k = float(v.shape[0])
    assert v.shape[0] == v.shape[1]
    s, log_det_v = np.linalg.slogdet(v)
    assert s > 0
    return multivariate_digamma(0.5 * df, k) + \
           k * np.log(2) + log_det_v

# If \Sigma ~ Wishart(v, df), return E[log(diag(\Sigma ^ -1))]
def e_log_inv_wishart_diag(df, v):
    k = float(v.shape[0])
    assert v.shape[0] == v.shape[1]
    v_inv_diag = np.diag(np.linalg.inv(v))
    return np.log(v_inv_diag) - \
           sp.special.digamma(0.5 * (df - k + 1)) - np.log(2)

def get_e_lognormal(mu, sigma_sq):
    return np.exp(mu + 0.5 * sigma_sq)

def get_var_lognormal(mu, sigma_sq):
    e_lognormal = get_e_lognormal(mu, sigma_sq)
    return (np.exp(sigma_sq) - 1) * (e_lognormal ** 2)

def get_e_log_gamma(shape, rate):
    return sp.special.digamma(shape) - np.log(rate)

def get_e_dirichlet(alpha):
    denom = np.sum(alpha, 0, keepdims = True)
    return alpha / denom

def get_e_log_dirichlet(alpha):
    digamma_sum = sp.special.digamma(np.sum(alpha, 0, keepdims=True))
    return sp.special.digamma(alpha) - digamma_sum


##########################
# Numeric integration functions

def get_e_fun_normal(means, infos, gh_loc, gh_weights, fun):
    # compute E(fun(X)) where X is an array of normals defined by parameters
    # means and infos, and fun is a function that can evaluate arrays
    # componentwise

    # gh_loc and g_weights are sample points and weights, respectively,
    # chosen such that they will correctly integrate p(x) \exp(-x^2) over
    # (-Inf, Inf), for any polynomial p of degree 2*deg - 1 or less

    assert means.shape == infos.shape
    draws_axis = means.ndim
    change_of_vars = np.sqrt(2) * gh_loc * \
                1/np.sqrt(np.expand_dims(infos, axis = draws_axis)) + \
                np.expand_dims(means, axis = draws_axis)

    integrand = fun(change_of_vars)

    return np.sum(1/np.sqrt(np.pi) * gh_weights * integrand, axis = draws_axis)

def get_e_logitnormal(lognorm_means, lognorm_infos, gh_loc, gh_weights):
    # get the expectation of a logit normal distribution
    identity_fun = lambda x : sp.special.expit(x)

    return get_e_fun_normal(lognorm_means, lognorm_infos, \
                            gh_loc, gh_weights, identity_fun)

def get_e_log_logitnormal(lognorm_means, lognorm_infos, gh_loc, gh_weights):
    # get expectation of Elog(X) and E[1 - log(X)] when X follows a logit normal

    # the function below is log(expit(v))
    log_v = lambda x : np.maximum(-np.log(1 + np.exp(-x)), -1e16) * (x > -1e2)\
                                + x * (x <= -1e2)

    # I believe that the above will avoid the numerical issues. If x is very small,
    # log(1 + e^(-x)) is basically -x, hence the two cases.
    # the maximum in the first term is taken so that when
    # -np.log(1 + np.exp(-x)) = -Inf, it really just returns -1e16;
    # apparently -Inf * 0.0 is NaN in python.

    e_log_v = get_e_fun_normal(lognorm_means, lognorm_infos, \
                            gh_loc, gh_weights, log_v)
    e_log_1mv = - lognorm_means + e_log_v
    return e_log_v, e_log_1mv


##########################
# Updating from natural parameters

# Return the mean and info for a univarite normal distribution in x where
# log p(x) = e_term * x + e2_term * x^2 + C
def get_uvn_from_natural_parameters(e_term, e2_term):
    x_info = -2.0 * e2_term
    x_mean = e_term / x_info
    return x_mean, x_info


#############################
# Priors

# TODO: perhaps rename these expected_*_prior
def mvn_prior(prior_mean, prior_info, e_obs, cov_obs):
    obs_diff = e_obs - prior_mean
    return -0.5 * (np.dot(obs_diff, np.matmul(prior_info, obs_diff)) + \
                   np.trace(np.matmul(prior_info, cov_obs)))

def uvn_prior(prior_mean, prior_info, e_obs, var_obs):
    return -0.5 * (prior_info * ((e_obs - prior_mean) ** 2 + var_obs))

def gamma_prior(prior_shape, prior_rate, e_obs, e_log_obs):
    return (prior_shape - 1) * e_log_obs - prior_rate * e_obs

def exponential_prior(lambda_par, e_obs):
    return -1 * lambda_par * e_obs

def dirichlet_prior(alpha, log_e_obs):
    assert np.shape(alpha) == np.shape(log_e_obs), \
            'shape of alpha and log_e_obs do not match'
    return np.dot(alpha - 1, log_e_obs)

# If \Sigma^{-1} ~ Wishart(v, df), return the LKJ prior
# proportional to (lkj_param - 1) * \log |\Sigma|
def expected_ljk_prior(lkj_param, df, v):
    e_log_r = -1 * e_log_det_wishart(df, v) - \
              np.sum(e_log_inv_wishart_diag(df, v))
    return (lkj_param - 1) * e_log_r

def get_e_dp_prior_logitnorm_approx(alpha, lognorm_means, lognorm_infos, \
                            gh_loc, gh_weights):
    # get the expectation of the dp prior with a logit normal approximation
    # for the beta sticks

    e_log_v, e_log_1mv =\
        get_e_log_logitnormal(lognorm_means, lognorm_infos, gh_loc, gh_weights)

    return (alpha - 1) * e_log_1mv
