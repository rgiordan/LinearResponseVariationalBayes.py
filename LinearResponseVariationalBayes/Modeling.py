
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


def get_standard_draws(num_draws):
    draw_spacing = 1 / float(num_draws + 1)
    target_quantiles = np.linspace(draw_spacing, 1 - draw_spacing, num_draws)
    return osp.stats.norm.ppf(target_quantiles)


def univariate_normal_log_prob(u, u_mean, u_info):
    return -0.5 * u_info * (u - u_mean)**2 + 0.5 * u_info \
           - 0.5 * np.log(2 * np.pi)


# Calculate \int q(z) fun(z) dz where q(z) = N(z_mean, z_sd ** 2)
# using a differentiable transformation of the standard normal draws
# std_draws and importance sampling.
#
# This is like an importance sampling version of the reparameterization trick.
# It uses a Taylor expansion of log_fun around z0.
# def importance_sampling_integrate_univariate_normal(
#     z_mean, z_sd, z0, log_fun, log_fun_grad, log_fun_hess, std_draws,
#     aggregate_all=True):
#
#     # q(u) will be a univariate normal importance sampling distribution.
#     # Its natural parameters are given by a Taylor expansion of log_fun.
#     z_info = 1 / z_sd ** 2
#
#     z_nat_param = z_info * z_mean
#     z2_nat_param = -0.5 * z_info
#
#     u_nat_param = z_nat_param + log_fun_grad(z0)
#     u2_nat_param = z2_nat_param + log_fun_hess(z0)
#
#     u_info = -2 * u2_nat_param
#     u_sd = 1. / np.sqrt(u_info)
#     u_mean = u_nat_param / u_info
#
#     draws_axis = u_sd.ndim
#     u_draws = \
#         np.expand_dims(u_sd, axis=draws_axis) * std_draws + \
#         np.expand_dims(u_mean, axis=draws_axis)
#
#     u_log_prob = univariate_normal_log_prob(
#         u_draws,
#         np.expand_dims(u_mean, axis=draws_axis),
#         np.expand_dims(u_info, axis=draws_axis))
#     z_log_prob = univariate_normal_log_prob(
#         u_draws,
#         np.expand_dims(z_mean, axis=draws_axis),
#         np.expand_dims(z_info, axis=draws_axis))
#     log_f_z = log_fun(u_draws)
#
#     # Importance sampling
#     log_imp_weights = np.exp(z_log_prob + log_f_z - u_log_prob)
#
#     if aggregate_all:
#         return np.sum(log_imp_weights) / len(std_draws)
#     else:
#         return np.sum(log_imp_weights, axis=draws_axis) / len(std_draws)
