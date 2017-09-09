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

def dirichlet_entropy(alpha):
    sum_alpha = np.sum(alpha)
    log_beta = np.sum(sp.special.gammaln(alpha)) \
                - sp.special.gammaln(sum_alpha)
    return log_beta - (len(alpha) - sum_alpha) * sp.special.digamma(sum_alpha) \
            - np.dot((alpha - 1), sp.special.digamma(alpha))

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
