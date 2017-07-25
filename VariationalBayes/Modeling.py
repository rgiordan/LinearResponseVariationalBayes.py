
import autograd.numpy as np
import autograd.scipy as sp
import scipy as osp
import math


# This numerically estimates an intractable expectation that
# occurs in logistic regression.  Given
#
# z ~ N(\mu, \sigma^2)
# p = exp(z) / (1 + exp(z))
#
# we want to calculate
# E[y * z - log(1 - p)].
def get_e_logistic_term(y, z_mean, z_sd, std_draws):
    assert z_sd.ndim == y.ndim
    assert z_mean.ndim == y.ndim

    # The last axis will be the standard draws axis.
    draws_axis = z_sd.ndim
    z_draws = \
        np.expand_dims(z_sd, axis=draws_axis) * std_draws + \
        np.expand_dims(z_mean, axis=draws_axis)

    # By dividing by the number of standard draws after summing,
    # we add the sample means for all the observations.
    # Note that
    # log(1 - p) = log(1 / (1 + exp(z))) = -log(1 + exp(z))
    logit_term = \
        np.sum(np.log1p(np.exp(z_draws))) / std_draws.size
    return np.sum(y * z_mean) - logit_term


def get_standard_draws(num_draws):
    draw_spacing = 1 / float(num_draws + 1)
    target_quantiles = np.linspace(draw_spacing, 1 - draw_spacing, num_draws)
    return osp.stats.norm.ppf(target_quantiles)
