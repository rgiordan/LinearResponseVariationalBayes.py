import autograd.numpy as np

# Entropies

def UnivariateNormalEntropy(var_obs):
    return 0.5 * np.sum(np.log(var_obs))

def MultivariateNormalEntropy(cov_obs):
    sign, logdet = np.linalg.slogdet(cov_obs)
    assert sign > 0
    return 0.5 * logdet

def GammaEntropy(shape, rate):
    return np.sum(shape - np.log(rate) + asp.special.gammaln(shape) + \
                  (1 - shape) * asp.special.digamma(shape))

# Priors

def MVNPrior(prior_mean, prior_info, e_obs, cov_obs):
    obs_diff = e_obs - prior_mean
    return -0.5 * (np.dot(obs_diff, np.matmul(prior_info, obs_diff)) + \
                   np.trace(np.matmul(prior_info, cov_obs)))

def UVNPrior(prior_mean, prior_info, e_obs, var_obs):
    return -0.5 * (prior_info * ((e_obs - prior_mean) ** 2 + var_obs))

def GammaPrior(prior_shape, prior_rate, e_obs, e_log_obs):
    return (prior_shape - 1) * e_log_obs - prior_rate * e_obs
