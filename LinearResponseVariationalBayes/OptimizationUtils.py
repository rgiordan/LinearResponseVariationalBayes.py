import numpy as np
import scipy as sp

# Get the matrix inverse square root of a symmetric matrix with eigenvalue
# thresholding.  This is particularly useful for calculating preconditioners.
def get_sym_matrix_inv_sqrt(hessian, ev_min=None, ev_max=None):
    hessian_sym = 0.5 * (hessian + hessian.T)
    eig_val, eig_vec = np.linalg.eigh(hessian_sym)

    if not ev_min is None:
        eig_val[eig_val <= ev_min] = ev_min
    if not ev_max is None:
        eig_val[eig_val >= ev_max] = ev_max

    hess_corrected = np.matmul(eig_vec,
                               np.matmul(np.diag(eig_val), eig_vec.T))

    hess_inv_sqrt = \
        np.matmul(eig_vec, np.matmul(np.diag(1 / np.sqrt(eig_val)), eig_vec.T))
    return np.array(hess_inv_sqrt), np.array(hess_corrected)


# Set the preconditioner attribute of objective to the inverse square root
# of the Hessian matrix at the current value.
def set_objective_preconditioner(
    objective, free_par=None, hessian=None, ev_min=None, ev_max=None):

    if free_par is None and hessian is None:
        raise ValueError(
            'You must specify either a Hessian or the free_par at which ' +
            'the objective\'s Hessian is to be evaluated.')

    if hessian is None:
        hessian = objective.fun_free_hessian(free_par)

    inv_hess_sqrt, hessian_corrected = \
        get_sym_matrix_inv_sqrt(hessian, ev_min=ev_min, ev_max=ev_max)

    objective.preconditioner = inv_hess_sqrt

    return hessian, inv_hess_sqrt, hessian_corrected


def minimize_objective_trust_ncg(
    objective, init_x, precondition,
    maxiter = 50, gtol = 1e-6, disp = True,
    print_every = None, init_logger = True):

    if init_logger:
        objective.logger.initialize()
    if print_every is not None:
        objective.logger.print_every = print_every
    objective.preconditioning = precondition
    if precondition:
        assert objective.preconditioner is not None
        init_x_cond = np.linalg.solve(objective.preconditioner, init_x)
        obj_opt = sp.optimize.minimize(
            lambda par: objective.fun_free_cond(par, verbose=disp),
            x0=init_x_cond,
            jac=objective.fun_free_grad_cond,
            hessp=objective.fun_free_hvp_cond,
            method='trust-ncg',
            options={'maxiter': maxiter, 'gtol': gtol, 'disp': disp})
        opt_x = objective.uncondition_x(obj_opt.x)
    else:
        obj_opt = sp.optimize.minimize(
            lambda par: objective.fun_free(par, verbose=disp),
            x0=init_x,
            jac=objective.fun_free_grad,
            hessp=objective.fun_free_hvp,
            method='trust-ncg',
            options={'maxiter': maxiter, 'gtol': gtol, 'disp': disp})
        opt_x = obj_opt.x

    return opt_x, obj_opt


def minimize_objective_bfgs(
    objective, init_x, precondition=False,
    maxiter=500, disp=True, print_every=None,
    init_logger=True):

    if init_logger:
        objective.logger.initialize()

    if print_every is not None:
        objective.logger.print_every = print_every
    objective.preconditioning = precondition
    if precondition:
        assert objective.preconditioner is not None
        init_x_cond = np.linalg.solve(objective.preconditioner, init_x)
        obj_opt = sp.optimize.minimize(
            lambda par: objective.fun_free_cond(par, verbose=disp),
            x0=init_x_cond,
            jac=objective.fun_free_grad_cond,
            method='BFGS',
            options={'maxiter': maxiter, 'disp': disp})
        opt_x = objective.uncondition_x(obj_opt.x)
    else:
        obj_opt = sp.optimize.minimize(
            lambda par: objective.fun_free(par, verbose=disp),
            x0=init_x,
            jac=objective.fun_free_grad,
            method='BFGS',
            options={'maxiter': maxiter, 'disp': disp})
        opt_x = obj_opt.x

    return opt_x, obj_opt


# Repeatedly optimize until convergence.
# optimization_fun (and initial_optimization_fun) must take a single argument,
# the starting point, and return an optimal x and an optimization result.
def repeatedly_optimize(
    objective, optimization_fun, init_x,
    initial_optimization_fun=None,
    max_iter=100, gtol=1e-8, ftol=1e-8, xtol=1e-8, disp=False,
    keep_intermediate_optimizations=False):

    opt_results = []
    if initial_optimization_fun is not None:
        if disp:
            print('Running intitial optimization.')
        init_x, init_opt = initial_optimization_fun(init_x)
        if keep_intermediate_optimizations:
            opt_results.append(init_opt)

    x_diff = float('inf')
    f_diff = float('inf')
    converged = False
    f_val = objective.fun_free(init_x)
    i = 0
    x = init_x

    while i < max_iter and (not converged):
        if disp:
            print('\n---------------------------------\n' +
                  'Repeated optimization iteration ', i)
        i += 1
        new_x, obj_opt = optimization_fun(x)
        if keep_intermediate_optimizations:
            opt_results.append(obj_opt)

        # Check convergence.
        new_f_val = objective.fun_free(new_x)
        grad_val = objective.fun_free_grad(new_x)

        x_diff = np.sum(np.abs(new_x - x))
        f_diff = np.abs(new_f_val - f_val)
        grad_l1 = np.sum(np.abs(grad_val))
        x_conv = x_diff < xtol
        f_conv = f_diff < ftol
        grad_conv = grad_l1 < gtol

        x = new_x
        f_val = new_f_val
        converged = x_conv or f_conv or grad_conv
        if disp:
            print('Iter {}: x_diff = {}, f_diff = {}, grad_l1 = {}'.format(
                i, x_diff, f_diff, grad_l1))

    return new_x, converged, x_conv, f_conv, grad_conv, obj_opt, opt_results
