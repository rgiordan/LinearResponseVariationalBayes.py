
import VariationalBayes as vb
import VariationalBayes.ExponentialFamilies as ef
import VariationalBayes.Modeling as modeling
import VariationalBayes.SparseObjectives as obj_lib
from VariationalBayes.Parameters import convert_vector_to_free_hessian

import autograd
import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy as sp
import scipy as osp
from scipy import sparse
import numpy as onp

import json
import copy

def load_json_data(json_filename):
    json_file = open(json_filename, 'r')
    json_dat = json.load(json_file)
    json_file.close()

    stan_dat = json_dat['stan_dat']

    print(stan_dat.keys())
    K = stan_dat['K'][0]
    NObs = stan_dat['N'][0]
    NG = stan_dat['NG'][0]

    y_g_vec = np.array(stan_dat['y_group'])
    y_vec = np.array(stan_dat['y'])
    x_mat = np.array(stan_dat['x'])

    glmm_par = get_glmm_parameters(K=K, NG=NG)

    # Define a class to contain prior parameters.
    prior_par = get_default_prior_params(K)
    prior_par['beta_prior_mean'].set(np.array(stan_dat['beta_prior_mean']))

    prior_par['beta_prior_info'].set(np.array(stan_dat['beta_prior_info']))

    prior_par['mu_prior_mean'].set(stan_dat['mu_prior_mean'][0])
    prior_par['mu_prior_info'].set(stan_dat['mu_prior_info'][0])

    prior_par['tau_prior_alpha'].set(stan_dat['tau_prior_alpha'][0])
    prior_par['tau_prior_beta'].set(stan_dat['tau_prior_beta'][0])

    # An index set to make sure jacobians match the order expected by R.
    prior_par_indices = copy.deepcopy(prior_par)
    prior_par_indices.set_name('Prior Indices')
    prior_par_indices.set_vector(np.array(range(prior_par_indices.vector_size())))

    return y_g_vec, y_vec, x_mat, glmm_par, prior_par


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

    return np.array(x_mat), np.array(y_g_vec), np.array(y_vec), true_rho, true_u


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


####### Modeling functions
def get_data_log_lik_terms(glmm_par, x_mat, y_vec, y_g_vec, gh_x, gh_w):
    e_beta = glmm_par['beta'].e()
    var_beta = glmm_par['beta'].var()

    # atleast_1d is necessary for indexing by y_g_vec to work right.
    e_u = np.atleast_1d(glmm_par['u'].e())
    var_u = np.atleast_1d(glmm_par['u'].var())

    # Log likelihood from data.
    z_mean = e_u[y_g_vec] + np.squeeze(np.matmul(x_mat, e_beta))
    z_sd = np.sqrt(
        var_u[y_g_vec] +
        np.squeeze(np.einsum('nk,k,nk->n', x_mat, var_beta, x_mat)))
    return \
        y_vec * z_mean - \
        modeling.get_e_logistic_term_guass_hermite(
            z_mean, z_sd, gh_x, gh_w, aggregate_all=False)


def get_re_log_lik(glmm_par):
    e_mu = glmm_par['mu'].e()
    var_mu = glmm_par['mu'].var()
    e_tau = glmm_par['tau'].e()
    e_log_tau = glmm_par['tau'].e_log()
    e_u = glmm_par['u'].e()
    var_u = glmm_par['u'].var()

    return -0.5 * e_tau * np.sum(
        ((e_mu - e_u) ** 2) + var_mu + var_u) + \
        0.5 * e_log_tau * glmm_par['u'].size()


def get_global_entropy(glmm_par):
    info_mu = glmm_par['mu']['info'].get()
    info_beta = glmm_par['beta']['info'].get()
    tau_shape = glmm_par['tau']['shape'].get()
    tau_rate = glmm_par['tau']['rate'].get()

    return \
        ef.univariate_normal_entropy(info_mu) + \
        ef.univariate_normal_entropy(info_beta) + \
        ef.gamma_entropy(tau_shape, tau_rate)


def get_local_entropy(glmm_par):
    info_u = glmm_par['u']['info'].get()
    return ef.univariate_normal_entropy(info_u)


def get_e_log_prior(glmm_par, prior_par):
    e_beta = glmm_par['beta']['mean'].get()
    info_beta = glmm_par['beta']['info'].get()
    #cov_beta = np.linalg.inv(info_beta)
    cov_beta = np.diag(1. / info_beta)
    beta_prior_info = prior_par['beta_prior_info'].get()
    beta_prior_mean = prior_par['beta_prior_mean'].get()
    e_mu = glmm_par['mu']['mean'].get()
    info_mu = glmm_par['mu']['info'].get()
    var_mu = 1 / info_mu
    e_tau = glmm_par['tau'].e()
    e_log_tau = glmm_par['tau'].e_log()

    e_log_p_beta = ef.mvn_prior(
        prior_mean = prior_par['beta_prior_mean'].get(),
        prior_info = prior_par['beta_prior_info'].get(),
        e_obs = e_beta,
        cov_obs = cov_beta)

    e_log_p_mu = ef.uvn_prior(
        prior_mean = prior_par['mu_prior_mean'].get(),
        prior_info = prior_par['mu_prior_info'].get(),
        e_obs = e_mu,
        var_obs = var_mu)

    e_log_p_tau = ef.gamma_prior(
        prior_shape = prior_par['tau_prior_alpha'].get(),
        prior_rate = prior_par['tau_prior_beta'].get(),
        e_obs = e_tau,
        e_log_obs = e_log_tau)

    return e_log_p_beta + e_log_p_mu + e_log_p_tau


class LogisticGLMM(object):
    def __init__(
        self, glmm_par, prior_par, x_mat, y_vec, y_g_vec, num_gh_points):

        self.glmm_par = copy.deepcopy(glmm_par)
        self.prior_par = copy.deepcopy(prior_par)
        self.x_mat = np.array(x_mat)
        self.y_vec = np.array(y_vec)
        self.y_g_vec = np.array(y_g_vec)
        self.set_gh_points(num_gh_points)

        self.beta_dim = self.x_mat.shape[1]
        self.num_groups = np.max(self.y_g_vec) + 1

        self.use_weights = False
        self.weights = np.full(self.x_mat.shape[0], 1.0)

        assert np.min(y_g_vec) == 0
        assert np.max(y_g_vec) == self.glmm_par['u'].size() - 1

        self.objective = obj_lib.Objective(self.glmm_par, self.get_kl)

        self.get_prior_model_grad = \
            autograd.grad(self.get_e_log_prior_from_args, argnum=0)
        self.get_prior_hess = \
            autograd.jacobian(self.get_prior_model_grad, argnum=1)

    def set_gh_points(self, num_gh_points):
        self.num_gh_points = num_gh_points
        self.gh_x, self.gh_w = onp.polynomial.hermite.hermgauss(num_gh_points)

    def get_e_log_prior(self):
        return get_e_log_prior(self.glmm_par, self.prior_par)

    def get_data_log_lik_terms(self):
        return get_data_log_lik_terms(
            glmm_par = self.glmm_par,
            x_mat = self.x_mat,
            y_vec = self.y_vec,
            y_g_vec = self.y_g_vec,
            gh_x = self.gh_x,
            gh_w = self.gh_w)

    def get_log_lik(self):
        if self.use_weights:
            data_log_lik = np.sum(self.weights * self.get_data_log_lik_terms())
        else:
            data_log_lik = np.sum(self.get_data_log_lik_terms())

        # Log likelihood from random effect terms.
        re_log_lik = get_re_log_lik(self.glmm_par)

        return data_log_lik + re_log_lik

    def get_entropy(self):
        return get_global_entropy(self.glmm_par) + \
               get_local_entropy(self.glmm_par)

    def get_elbo(self):
        log_lik = self.get_log_lik()
        entropy = self.get_entropy()
        e_log_prior = self.get_e_log_prior()
        return np.squeeze(log_lik + entropy + e_log_prior)

    def get_kl(self):
        return -1 * self.get_elbo()

    def get_e_log_prior_from_args(self, prior_vec, free_par):
        self.glmm_par.set_free(free_par)
        self.prior_par.set_vector(prior_vec)
        return self.get_e_log_prior()

    def tr_optimize(self, trust_init, num_gh_points,
                    print_every=5, gtol=1e-6, maxiter=500):
        self.set_gh_points(num_gh_points)
        self.objective.logger.initialize()
        self.objective.logger.print_every = print_every
        vb_opt = osp.optimize.minimize(
            lambda par: self.objective.fun_free(par, verbose=True),
            x0=trust_init,
            method='trust-ncg',
            jac=self.objective.fun_free_grad,
            hessp=self.objective.fun_free_hvp,
            tol=1e-6,
            options={'maxiter': maxiter, 'disp': True, 'gtol': gtol })
        return vb_opt


class MomentWrapper(object):
    def __init__(self, glmm_par):
        self.glmm_par = copy.deepcopy(glmm_par)
        K = glmm_par['beta']['mean'].size()
        NG =  glmm_par['u']['mean'].size()
        self.moment_par = vb.ModelParamsDict('Moment Parameters')
        self.moment_par.push_param(vb.VectorParam('e_beta', K))
        self.moment_par.push_param(vb.ScalarParam('e_mu'))
        self.moment_par.push_param(vb.ScalarParam('e_tau'))
        self.moment_par.push_param(vb.ScalarParam('e_log_tau'))
        self.moment_par.push_param(vb.VectorParam('e_u', NG))

        self.get_moment_jacobian = \
            autograd.jacobian(self.get_moment_vector_from_free)

    def __str__(self):
        return str(self.moment_par)

    def set_moments(self, free_par_vec):
        self.glmm_par.set_free(free_par_vec)
        self.moment_par['e_beta'].set(self.glmm_par['beta'].e())
        self.moment_par['e_mu'].set(self.glmm_par['mu'].e())
        self.moment_par['e_tau'].set(self.glmm_par['tau'].e())
        self.moment_par['e_log_tau'].set(self.glmm_par['tau'].e_log())
        self.moment_par['e_u'].set(self.glmm_par['u'].e())

    # Return a posterior moment of interest as a function of unconstrained parameters.
    def get_moment_vector_from_free(self, free_par_vec):
        self.set_moments(free_par_vec)
        return self.moment_par.get_vector()


#####################################
# A sparse version of the objective to construct sparse Hessians

# Since we never use the free version of the observation parameters,
# we don't need to set the minimum allowable values.
def get_group_parameters(K, num_groups=1):
    group_par = vb.ModelParamsDict('Single group GLMM parameters')
    group_par.push_param(vb.UVNParam('mu'))
    group_par.push_param(vb.GammaParam('tau'))
    group_par.push_param(vb.UVNParamVector('beta', K))
    group_par.push_param(vb.UVNParamVector('u', num_groups))
    return group_par

# Since we never use the free version of the global parameters, we don't need to
# set the minimum allowable values.
def get_global_parameters(K):
    global_par = vb.ModelParamsDict('Global GLMM parameters')
    global_par.push_param(vb.UVNParam('mu'))
    global_par.push_param(vb.GammaParam('tau'))
    global_par.push_param(vb.UVNParamVector('beta', K))
    return global_par

def set_group_parameters(glmm_par, group_par, groups):
    # Stupid assert fails with integers.  Why can't an integer just have
    # len() == 1?
    assert(len(groups) == group_par['u'].size())
    group_par['beta'].set_vector(glmm_par['beta'].get_vector())
    group_par['mu'].set_vector(glmm_par['mu'].get_vector())
    group_par['tau'].set_vector(glmm_par['tau'].get_vector())

    group_par['u']['mean'].set(glmm_par['u']['mean'].get()[groups])
    group_par['u']['info'].set(glmm_par['u']['info'].get()[groups])

def set_global_parameters(glmm_par, global_par):
    global_par['beta'].set_vector(glmm_par['beta'].get_vector())
    global_par['mu'].set_vector(glmm_par['mu'].get_vector())
    global_par['tau'].set_vector(glmm_par['tau'].get_vector())


# Evaluate the model at only a certain select set of groups.
class SubGroupsModel(object):
    def __init__(self, model, num_sub_groups=1):
        self.model = model
        self.glmm_par = self.model.glmm_par
        self.num_sub_groups = num_sub_groups

        self.full_indices = obj_lib.make_index_param(self.glmm_par)

        self.group_par = get_group_parameters(
            self.model.beta_dim, self.num_sub_groups)
        self.group_indices = obj_lib.make_index_param(self.group_par)

        self.set_group_parameters(np.arange(0, self.num_sub_groups))

        self.group_rows = [ self.model.y_g_vec == g \
                            for g in range(np.max(self.model.y_g_vec) + 1)]

        self.kl_objective = obj_lib.Objective(
            self.group_par, self.get_group_kl)

        self.data_kl_objective = obj_lib.Objective(
            self.group_par, self.get_group_data_elbo)
        self.get_data_kl_objective_jac = autograd.jacobian(
            self.data_kl_objective.fun_vector)

    # Set the group parameters from the global parameters and
    # return a vector of the indices within the full model.
    def set_group_parameters(self, groups):
        assert(np.max(groups) < self.model.num_groups)
        assert(len(groups) == self.num_sub_groups)
        self.groups = groups

        set_group_parameters(self.glmm_par, self.group_par, groups)
        set_group_parameters(self.full_indices, self.group_indices, groups)
        return self.group_par.get_vector(), self.group_indices.get_vector()

    # For a vector of groups, return the subset of the data needed to evaluate
    # the model at those groups, including a y_g vector appropriate to a set
    # of parameters that only contains parameters for these groups.
    def get_data_for_groups(self, groups):
        # Rows in the dataset corresponding to these groups:
        all_group_rows = onp.logical_or.reduce(
            [self.group_rows[g] for g in groups])

        # Which indices within the groups vector correspond to these rows:
        y_g_sub = np.hstack([ np.full(np.sum(self.group_rows[groups[ig]]), ig) \
                              for ig in range(len(groups))])
        return all_group_rows, y_g_sub

    # Likelihood functions:
    def get_group_kl(self):
        all_group_rows, y_g_sub = self.get_data_for_groups(self.groups)
        data_log_lik = np.sum(get_data_log_lik_terms(
            glmm_par = self.group_par,
            y_g_vec = y_g_sub,
            x_mat = self.model.x_mat[all_group_rows, :],
            y_vec = self.model.y_vec[all_group_rows],
            gh_x = self.model.gh_x,
            gh_w = self.model.gh_w))
        re_log_lik = get_re_log_lik(self.group_par)
        u_entropy = get_local_entropy(self.group_par)

        return -1 * np.squeeze(data_log_lik + re_log_lik + u_entropy)

    # Each entry returned by get_data_log_lik_terms corresponds to a
    # single data point.  This is intended to be the derivative of the
    # elbo with respect to weights on each data point.
    def get_group_data_elbo(self):
        all_group_rows, y_g_sub = self.get_data_for_groups(self.groups)
        data_log_lik = get_data_log_lik_terms(
            glmm_par = self.group_par,
            y_g_vec = y_g_sub,
            x_mat = self.model.x_mat[all_group_rows, :],
            y_vec = self.model.y_vec[all_group_rows],
            gh_x = self.model.gh_x,
            gh_w = self.model.gh_w)

        return -1 * np.squeeze(data_log_lik)

    # This is as a function of vector parameters.
    def get_sparse_kl_vec_hessian(self, print_every_n=-1):
        full_hess_dim = self.glmm_par.vector_size()
        sparse_group_hess = \
            osp.sparse.csr_matrix((full_hess_dim, full_hess_dim))
        for g in range(self.model.num_groups):
            if g % print_every_n == print_every_n:
                print('Group {} of {}.'.format(g, self.model.num_groups - 1))
            group_par_vec, group_indices = self.set_group_parameters([g])
            group_vec_hessian = \
                self.kl_objective.fun_vector_hessian(group_par_vec)
            sparse_group_hess += \
                obj_lib.get_sparse_sub_hessian(
                    sub_hessian = group_vec_hessian,
                    full_indices = group_indices,
                    full_hess_dim = full_hess_dim)
        return sparse_group_hess

    # This is as a function of the vector parameters.
    def get_sparse_weight_vec_jacobian(self, print_every_n=-1):
        vector_param_size = self.glmm_par.vector_size()
        n_obs = self.model.x_mat.shape[0]
        weight_indices = np.arange(0, n_obs)
        sparse_weight_jacobian = \
            osp.sparse.csr_matrix((n_obs, vector_param_size))
        for g in range(self.model.num_groups):
            if g % print_every_n == print_every_n:
                print('Group {} of {}'.format(g, self.model.num_groups - 1))
            group_weight_indices = weight_indices[self.group_rows[g]]
            group_par_vec, group_indices = self.set_group_parameters([g])
            group_obs_jac = np.atleast_2d(
                self.get_data_kl_objective_jac(group_par_vec))
            sparse_weight_jacobian += \
                obj_lib.get_sparse_sub_matrix(
                    sub_matrix = group_obs_jac,
                    col_indices = group_indices,
                    row_indices = group_weight_indices,
                    col_dim = vector_param_size,
                    row_dim = n_obs)

        return sparse_weight_jacobian


# Evaluate the global part of the model only.
class GlobalModel(object):
    def __init__(self, model):
        self.model = model
        self.glmm_par = self.model.glmm_par
        self.full_indices = obj_lib.make_index_param(self.glmm_par)
        self.global_par = get_global_parameters(self.model.beta_dim)
        self.global_indices = obj_lib.make_index_param(self.global_par)

        self.kl_objective = obj_lib.Objective(
            self.global_par, self.get_global_kl)

    def set_global_parameters(self):
        set_global_parameters(self.glmm_par, self.global_par)
        set_global_parameters(self.full_indices, self.global_indices)
        return self.global_par.get_vector(), self.global_indices.get_vector()

    def get_global_kl(self):
        return -1 * np.squeeze(
            get_global_entropy(self.global_par) + \
            get_e_log_prior(self.global_par, self.model.prior_par))

    # This is as a function of the vector parameters.
    def get_sparse_kl_vec_hessian(self, print_every_n=-1):
        full_hess_dim = self.glmm_par.vector_size()
        global_par_vec, global_indices = self.set_global_parameters()
        global_vec_hessian = \
            self.kl_objective.fun_vector_hessian(global_par_vec)
        sparse_global_hess = obj_lib.get_sparse_sub_hessian(
            sub_hessian = global_vec_hessian,
            full_indices = global_indices,
            full_hess_dim = full_hess_dim)
        return sparse_global_hess


def get_sparse_weight_free_jacobian(
    group_model, vector_jac=None, print_every_n=-1):
    if vector_jac is None:
        vector_jac = group_model.get_sparse_weight_vec_jacobian(
            print_every_n=print_every_n)
    free_to_vec_jacobian = \
        group_model.glmm_par.free_to_vector_jac(group_model.glmm_par.get_free())
    return vector_jac * free_to_vec_jacobian


def get_free_hessian(glmm_model, group_model, global_model,
                     vector_hess=None, print_every_n=-1):
    if vector_hess is None:
        group_vec_hess = group_model.get_sparse_kl_vec_hessian(
            print_every_n=print_every_n)
        global_vec_hess = global_model.get_sparse_kl_vec_hessian()
        vector_hess = global_vec_hess + group_vec_hess

    vector_grad = glmm_model.objective.fun_vector_grad(
        glmm_model.glmm_par.get_vector())

    return convert_vector_to_free_hessian(
        glmm_model.glmm_par,
        glmm_model.glmm_par.get_free(),
        vector_grad,
        vector_hess)


###################################
# MLE (MAP) estimators

def get_mle_parameters(K, NG):
    mle_par = vb.ModelParamsDict('GLMER Parameters')
    mle_par.push_param(vb.VectorParam('mu'))
    mle_par.push_param(vb.VectorParam('tau'))
    mle_par.push_param(vb.VectorParam('beta', K))
    mle_par.push_param(vb.VectorParam('u', NG))

    return mle_par

def get_mle_data_log_lik_terms(mle_par, x_mat, y_vec, y_g_vec):
    beta = mle_par['beta'].get()

    # atleast_1d is necessary for indexing by y_g_vec to work right.
    e_u = np.atleast_1d(mle_par['u'].get())

    # Log likelihood from data.
    z = e_u[y_g_vec] + np.squeeze(np.matmul(x_mat, beta))

    return y_vec * z - np.log1p(np.exp(z))

def get_mle_re_log_lik(mle_par):
    mu = mle_par['mu'].get()
    tau = mle_par['tau'].get()
    u = mle_par['u'].get()

    return -0.5 * tau * np.sum(
        ((mu - u) ** 2)) +  0.5 * np.log(tau) * mle_par['u'].size()

def get_mle_log_prior(mle_par, prior_par):
    beta = mle_par['beta'].get()
    mu = mle_par['mu'].get()
    tau = mle_par['tau'].get()

    K = len(beta)
    log_p_beta = ef.mvn_prior(
        prior_mean = prior_par['beta_prior_mean'].get(),
        prior_info = prior_par['beta_prior_info'].get(),
        e_obs = beta,
        cov_obs = np.zeros((K, K)))

    log_p_mu = ef.uvn_prior(
        prior_mean = prior_par['mu_prior_mean'].get(),
        prior_info = prior_par['mu_prior_info'].get(),
        e_obs = mu,
        var_obs = 0.0)

    log_p_tau = ef.gamma_prior(
        prior_shape = prior_par['tau_prior_alpha'].get(),
        prior_rate = prior_par['tau_prior_beta'].get(),
        e_obs = tau,
        e_log_obs = np.log(tau))

    return log_p_beta + log_p_mu + log_p_tau


def set_moment_par_from_mle(moment_par, mle_par):
    moment_par.set_vector(np.full(moment_par.vector_size(), np.nan))
    moment_par['e_beta'].set(mle_par['beta'].get())
    moment_par['e_mu'].set(mle_par['mu'].get())
    moment_par['e_tau'].set(mle_par['tau'].get())
    moment_par['e_log_tau'].set(np.log(mle_par['tau'].get()))
    moment_par['e_u'].set(mle_par['u'].get())

    return moment_par


class LogisticGLMMMaximumLikelihood(object):
    def __init__(self, mle_par, prior_par, x_mat, y_vec, y_g_vec):

        self.mle_par = copy.deepcopy(mle_par)
        self.prior_par = copy.deepcopy(prior_par)
        self.x_mat = np.array(x_mat)
        self.y_vec = np.array(y_vec)
        self.y_g_vec = np.array(y_g_vec)

        assert np.min(y_g_vec) == 0
        assert np.max(y_g_vec) == self.mle_par['u'].size() - 1


    def get_log_lik(self):
        data_log_lik = np.sum(get_mle_data_log_lik_terms(
            mle_par = self.mle_par,
            x_mat = self.x_mat,
            y_vec = self.y_vec,
            y_g_vec = self.y_g_vec))
        re_log_lik = get_mle_re_log_lik(self.mle_par)
        log_prior = get_mle_log_prior(self.mle_par, self.prior_par)
        return np.squeeze(data_log_lik + re_log_lik + log_prior)

    def get_log_loss(self):
        return -1 * self.get_log_lik()
