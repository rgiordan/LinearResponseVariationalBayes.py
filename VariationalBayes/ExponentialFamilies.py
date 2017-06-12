import autograd.numpy as np
import autograd.scipy as asp
import math

# Entropies.  Note that autograd hasn't defined these yet.

def UnivariateNormalEntropy(info_obs):
    # np.sum(asp.stats.norm.entropy(scale=np.sqrt(var_obs)))
    return 0.5 * np.sum(-1 * np.log(info_obs) + 1 + np.log(2 * math.pi))

def MultivariateNormalEntropy(info_obs):
    sign, logdet = np.linalg.slogdet(info_obs)
    assert sign > 0
    k = info_obs.shape[0]
    return 0.5 * (-1 * logdet + k + k * np.log(2 * math.pi))

def GammaEntropy(shape, rate):
    return np.sum(shape - np.log(rate) + asp.special.gammaln(shape) + \
                  (1 - shape) * asp.special.digamma(shape))

def DirichletEntropy(alpha):
    sum_alpha = np.sum(alpha)
    log_beta = np.sum(asp.special.gammaln(alpha)) \
                - asp.special.gammaln(sum_alpha)
    return log_beta - (np.shape(alpha) - sum_alpha) * asp.special.digamma(sum_alpha) \
            - np.dot((alpha - 1), asp.special.digamma(alpha))


# Priors

def MVNPrior(prior_mean, prior_info, e_obs, cov_obs):
    obs_diff = e_obs - prior_mean
    return -0.5 * (np.dot(obs_diff, np.matmul(prior_info, obs_diff)) + \
                   np.trace(np.matmul(prior_info, cov_obs)))

def UVNPrior(prior_mean, prior_info, e_obs, var_obs):
    return -0.5 * (prior_info * ((e_obs - prior_mean) ** 2 + var_obs))

def GammaPrior(prior_shape, prior_rate, e_obs, e_log_obs):
    return (prior_shape - 1) * e_log_obs - prior_rate * e_obs

def DirichletPrior(alpha, log_e_obs):
    assert np.shape(alpha) == np.shape(log_e_obs), \
            'shape of alpha and log_e_obs do not match'
    return np.dot(alpha - 1, log_e_obs)
