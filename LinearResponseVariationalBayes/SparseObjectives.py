import LinearResponseVariationalBayes as vb
from LinearResponseVariationalBayes.Parameters import \
    convert_vector_to_free_hessian
from LinearResponseVariationalBayes import ModelParamsDict
import autograd
import autograd.numpy as np
#import autograd.scipy as sp

import json_tricks

import scipy as sp
from scipy import sparse

from copy import deepcopy

import time

import warnings

# Apparently this can be replaced by the @ operator in Python 3.6.
def safe_matmul(x, y):
    if sp.sparse.issparse(x) or sp.sparse.issparse(y):
        return x * y
    else:
        return np.matmul(x, y)

def compress(x):
    if sp.sparse.issparse(x):
        return np.squeeze(np.asarray(x.todense()))
    else:
        return np.squeeze(np.asarray(x))


# An object to save various runtimes.
class Timer(object):
    def __init__(self):
        self.time_dict = {}
    def tic(self):
        self.tic_time = time.time()
    def toc(self, time_name, verbose=True):
        self.time_dict[time_name] = time.time() - self.tic_time
        if verbose:
            print('{}: {} seconds'.format(time_name, self.time_dict[time_name]))
    def __str__(self):
        return str(self.time_dict)


class Logger(object):
    def __init__(self, print_every=1):
        self.print_every = print_every
        self.initialize()
        self.print_x_diff = True
        self.callback = None

    def initialize(self):
        self.iter = 0
        self.last_x = None
        self.x = None
        self.value = None
        self.last_value = None
        self.x_array = []
        self.val_array = []

    def print_message(self):
        print('Iter ', self.iter, ' value: ', self.value)

    def log(self, value, x):
        self.value = value
        self.x = x
        self.x_array.append(x)
        self.val_array.append(value)

        # TODO: use the arrays instead of last_*
        if self.last_x is None:
            x_diff = float('inf')
        else:
            x_diff = np.max(np.abs(self.x - self.last_x))

        self.last_x = x
        self.last_value = value

        if self.iter % self.print_every == 0:
            if self.callback is None:
                self.print_message()
            else:
                self.callback(self)
        self.iter += 1

# TODO: replace the Objective classes with decorators.

# par should be a Parameter type.
# fun should be a function that takes no arguments but which is
# bound to par, i.e. which evaluates to a float that depends on the
# value of the parameters par.
class Objective(object):
    def __init__(self, par, fun):
        # TODO: redo this in the style of TwoParameterObjective

        self.par = par
        self.fun = fun

        self.ag_fun_free_grad = autograd.grad(self.fun_free, argnum=0)
        self.ag_fun_free_hessian = autograd.hessian(self.fun_free, argnum=0)
        self.ag_fun_free_jacobian = autograd.jacobian(self.fun_free, argnum=0)
        self.ag_fun_free_hvp = autograd.hessian_vector_product(
            self.fun_free, argnum=0)

        self.ag_fun_vector_grad = autograd.grad(self.fun_vector, argnum=0)
        self.ag_fun_vector_hessian = autograd.hessian(self.fun_vector, argnum=0)
        self.ag_fun_vector_jacobian = autograd.jacobian(
            self.fun_vector, argnum=0)
        self.ag_fun_vector_hvp = autograd.hessian_vector_product(
            self.fun_vector, argnum=0)

        self.preconditioner = None
        self.logger = Logger()

    # TODO: in a future version, make verbose a class attribute rather than
    # a keyword argument.
    def fun_free(self, free_val, *argv, verbose=False, **argk):
        self.par.set_free(free_val)
        val = self.fun(*argv, **argk)
        if verbose:
            self.logger.log(val, free_val)
        return val

    def fun_vector(self, vec_val, *argv, **argk):
        self.par.set_vector(vec_val)
        return self.fun(*argv, **argk)

    # Autograd wrappers.
    # Autograd functions populate parameter objects with ArrayBox types,
    # which can be inconvenient when calling get() or get_free() after
    # asking for a gradient, since you almost always want a numeric
    # value from get() or get_free().
    #
    # To get around this problem, the derivative functions in the objective
    # cache the value of the parameters passed to the function, which
    # are presumably numeric, and set the parameters to those values
    # after the autograd function is called.

    def cache_free_and_eval(self, autograd_fun, free_val, *argv, **argk):
        result = autograd_fun(free_val, *argv, **argk)
        self.par.set_free(free_val)
        return result

    def cache_vector_and_eval(self, autograd_fun, vec_val, *argv, **argk):
        result = autograd_fun(vec_val, *argv, **argk)
        self.par.set_vector(vec_val)
        return result

    def fun_free_grad(self, free_val, *argv, **argk):
        return self.cache_free_and_eval(
            self.ag_fun_free_grad, free_val, *argv, **argk)

    def fun_free_hessian(self, free_val, *argv, **argk):
        return self.cache_free_and_eval(
            self.ag_fun_free_hessian, free_val, *argv, **argk)

    def fun_free_jacobian(self, free_val, *argv, **argk):
        return self.cache_free_and_eval(
            self.ag_fun_free_jacobian, free_val, *argv, **argk)

    def fun_vector_grad(self, vec_val, *argv, **argk):
        return self.cache_vector_and_eval(
            self.ag_fun_vector_grad, vec_val, *argv, **argk)

    def fun_vector_hessian(self, vec_val, *argv, **argk):
        return self.cache_vector_and_eval(
            self.ag_fun_vector_hessian, vec_val, *argv, **argk)

    def fun_vector_jacobian(self, vec_val, *argv, **argk):
        return self.cache_vector_and_eval(
            self.ag_fun_vector_jacobian, vec_val, *argv, **argk)

    # Have to treat hvps separately for the additional argument.  :(
    #
    # Note that argument order, which is determined by autograd --
    # first comes the argument at which the Hessian is evaluated,
    # then other *argv arguments, then the vector by which the Hessian
    # is to be multiplied, then the keyword arguments.
    # See the definition of hessian_tensor_product in autograd.
    def fun_free_hvp(self, *argv, **argk):
        args, vec = argv[:-1], argv[-1]
        result = self.ag_fun_free_hvp(*args, vec, **argk)
        self.par.set_free(args[0])
        return result

    def fun_vector_hvp(self, *argv, **argk):
        args, vec = argv[:-1], argv[-1]
        result = self.ag_fun_vector_hvp(*args, vec, **argk)
        self.par.set_vector(args[0])
        return result


    # Pre-conditioned versions of the free functions.  The value at which
    # they are evaluted is assumed to include the preconditioner, i.e.
    # to be free_val = a * x.
    # Note that you must initialize the preconditioned objective with
    # preconditioner^{-1} init_x
    # if init_x is a guess of the original (unconditioned) value.
    def get_conditioned_x(self, free_val):
        return safe_matmul(self.preconditioner, free_val)

    # TODO: in a future version, make verbose a class attribute rather than
    # a keyword argument.
    def fun_free_cond(self, free_val, *argv, verbose=False, **argk):
        assert self.preconditioner is not None
        y = self.get_conditioned_x(free_val)
        return self.fun_free(y, *argv, verbose=verbose, **argk)

    def fun_free_grad_cond(self, free_val, *argv, **argk):
        assert self.preconditioner is not None
        y = self.get_conditioned_x(free_val)
        grad = self.fun_free_grad(y, *argv, **argk)
        return safe_matmul(self.preconditioner.T, grad)

    def fun_free_hessian_cond(self, free_val, *argv, **argk):
        assert self.preconditioner is not None
        y = self.get_conditioned_x(free_val)
        hess = self.fun_free_hessian(y, *argv, **argk)
        return safe_matmul(self.preconditioner.T,
                           safe_matmul(hess, self.preconditioner))

    # The argument order is the same as fun_free_hvp.
    def fun_free_hvp_cond(self, *argv, **argk):
        assert self.preconditioner is not None
        args, vec = argv[1:-1], argv[-1]
        free_val = argv[0]
        y = self.get_conditioned_x(free_val)
        return safe_matmul(
            self.preconditioner.T,
            self.fun_free_hvp(
                y, *args, safe_matmul(self.preconditioner, vec), **argk)
                )

    # Convert the optimum of the conditioned problem to the
    # value (with tests to be sure you're doing it right).
    def uncondition_x(self, cond_x):
        return safe_matmul(self.preconditioner, cond_x)


# A class for getting the Jacobian of the conversion from
# one parameter into another.
class ParameterConverter(object):
    def __init__(self, par_in, par_out, converter):
        self.par_in = par_in
        self.par_out = par_out
        self.converter = converter

        self.ag_free_to_vec_jacobian = \
            autograd.jacobian(self.converter_free_to_vec)
        self.ag_free_to_free_jacobian = \
            autograd.jacobian(self.converter_free_to_free)
        self.ag_vec_to_vec_jacobian = \
            autograd.jacobian(self.converter_vec_to_vec)
        self.ag_vec_to_free_jacobian = \
            autograd.jacobian(self.converter_vec_to_free)

    def converter_free_to_vec(self, free_par_in):
        self.par_in.set_free(free_par_in)
        self.converter()
        return self.par_out.get_vector()

    def converter_free_to_free(self, free_par_in):
        self.par_in.set_free(free_par_in)
        self.converter()
        return self.par_out.get_free()

    def converter_vec_to_vec(self, vec_par_in):
        self.par_in.set_vector(vec_par_in)
        self.converter()
        return self.par_out.get_vector()

    def converter_vec_to_free(self, vec_par_in):
        self.par_in.set_vector(vec_par_in)
        self.converter()
        return self.par_out.get_free()

    def cache_free_and_eval(self, autograd_fun, free_val_in):
        vec_val_out = self.par_out.get_vector()
        result = autograd_fun(free_val_in)
        self.par_in.set_free(free_val_in)
        self.par_out.set_vector(vec_val_out)
        return result

    def cache_vector_and_eval(self, autograd_fun, vec_val_in):
        vec_val_out = self.par_out.get_vector()
        result = autograd_fun(vec_val_in)
        self.par_in.set_vector(vec_val_in)
        self.par_out.set_vector(vec_val_out)
        return result

    def free_to_free_jacobian(self, free_par_in):
        return self.cache_free_and_eval(
            self.ag_free_to_free_jacobian, free_par_in)

    def free_to_vec_jacobian(self, free_par_in):
        return self.cache_free_and_eval(
            self.ag_free_to_vec_jacobian, free_par_in)

    def vec_to_free_jacobian(self, vec_par_in):
        return self.cache_vector_and_eval(
            self.ag_vec_to_free_jacobian, vec_par_in)

    def vec_to_vec_jacobian(self, vec_par_in):
        return self.cache_vector_and_eval(
            self.ag_vec_to_vec_jacobian, vec_par_in)


# TODO: delete this and only use ModelSensitivity.set_par.
def set_par(par, val, is_free):
    if is_free:
        par.set_free(val)
    else:
        par.set_vector(val)


# Like Objective, but with two parameters.  This is only useful for evaluating
# off-diagonal Hessians.
class TwoParameterObjective(object):
    def __init__(self, par1, par2, fun):
        self.par1 = par1
        self.par2 = par2
        self.fun = fun

        # hessian12 has par1 in the rows and par2 in the columns.
        # hessian21 has par2 in the rows and par1 in the columns.
        #
        # Note that, generally, autograd will be faster if you use hessian12
        # and par2 is the larger parameter.

        self._fun_grad1 = autograd.grad(self.eval_fun, argnum=0)
        self._fun_grad2 = autograd.grad(self.eval_fun, argnum=1)

        self._fun_hessian12 = \
            autograd.jacobian(self._fun_grad1, argnum=1)
        self._fun_hessian21 = \
            autograd.jacobian(self._fun_grad2, argnum=0)

    def cache_and_eval(
        self, autograd_fun, val1, val2, val1_is_free, val2_is_free,
        *argv, **argk):

        result = autograd_fun(
            val1, val2, val1_is_free, val2_is_free, *argv, **argk)

        set_par(self.par1, val1, val1_is_free)
        set_par(self.par2, val2, val2_is_free)

        return result

    def eval_fun(
        self, val1, val2, val1_is_free, val2_is_free,
        *argv, **argk):

        set_par(self.par1, val1, val1_is_free)
        set_par(self.par2, val2, val2_is_free)
        return self.fun(*argv, **argk)

    def fun_free(self, free_val1, free_val2, *argv, **argk):
        return self.eval_fun(
            free_val1, free_val2,
            True, True,
            *argv, **argk)

    def fun_vector(self, vec_val1, vec_val2, *argv, **argk):
        return self.eval_fun(
            vec_val1, vec_val2,
            False, False,
            *argv, **argk)

    def fun_grad1(
        self, val1, val2, val1_is_free, val2_is_free, *argv, **argk):
        return self.cache_and_eval(
            self._fun_grad1,
            val1, val2,
            val1_is_free, val2_is_free,
            *argv, **argk)

    def fun_grad2(
        self, val1, val2, val1_is_free, val2_is_free, *argv, **argk):
        return self.cache_and_eval(
            self._fun_grad2,
            val1, val2,
            val1_is_free, val2_is_free,
            *argv, **argk)

    def fun_free_hessian12(self, free_val1, free_val2, *argv, **argk):
        return self.cache_and_eval(
            self._fun_hessian12,
            free_val1, free_val2,
            True, True,
            *argv, **argk)

    def fun_free_hessian21(self, free_val1, free_val2, *argv, **argk):
        # return self._fun_hessian21(
        #     free_val1, free_val2,
        #     True, True,
        #     *argv, **argk)
        return self.cache_and_eval(
            self._fun_hessian21,
            free_val1, free_val2,
            True, True,
            *argv, **argk)

    def fun_vector_hessian12(self, vec_val1, vec_val2, *argv, **argk):
        # return self._fun_hessian12(
        #     vec_val1, vec_val2,
        #     False, False,
        #     *argv, **argk)
        return self.cache_and_eval(
            self._fun_hessian12,
            vec_val1, vec_val2,
            False, False,
            *argv, **argk)

    def fun_vector_hessian21(self, vec_val1, vec_val2, *argv, **argk):
        # return self._fun_hessian21(
        #     vec_val1, vec_val2,
        #     False, False,
        #     *argv, **argk)
        return self.cache_and_eval(
            self._fun_hessian21,
            vec_val1, vec_val2,
            False, False,
            *argv, **argk)

    def fun_hessian_free1_vector2(self, free_val1, vec_val2, *argv, **argk):
        # return self._fun_hessian12(
        #     free_val1, vec_val2,
        #     True, False,
        #     *argv, **argk)
        return self.cache_and_eval(
            self._fun_hessian12,
            free_val1, vec_val2,
            True, False,
            *argv, **argk)

    def fun_hessian_vector1_free2(self, vec_val1, free_val2, *argv, **argk):
        # return self._fun_hessian12(
        #     vec_val1, free_val2,
        #     False, True,
        #     *argv, **argk)
        return self.cache_and_eval(
            self._fun_hessian12,
            vec_val1, free_val2,
            False, True,
            *argv, **argk)


# A class for calculating parametric sensitivity of a model.
#
# Args:
#   - objective_fun: A target functor to be minimized.  It must take no
#     arguments, but its value should depend on the values of input_par and
#     hyper_par.
#   - input_par: The parameter at which objective_fun is minimized.
#   - output_par: The quantity of interest, a function of input_par.
#   - hyper_par: A hyperparameter at whose fixed value objective_fun is
#     minimized with respect to input_par.
#   - input_to_output_converter: a functor taking no arguments, but depending
#     on the value of input_par, that returns the vectorized value of output_par.
#   - optimal_input_par: Optional. The free value of input_par at which
#     objective_fun is minimized.  If unset, the current value of input_par
#     is used.
#   - objective_hessian: Optional. The Hessian of objective_fun evaluated at
#     optimal_input_par.  If unspecified, it is calculated when a
#     ParametricSensitivity class is instantiated.
#   - hyper_par_objective_fun: Optional.  A functor returning the part of the
#     objective function that depends on the hyperparameters.  If this is not
#     specified, objective_fun is used.  This can be useful for computational
#     efficiency, or to calculate the effect of perturbations that were not
#     implemented in the original model.
#
# Methods:
#   - set_optimal_input_par: Set a new value of optimal_input_par at which to
#     evaluate hyperparameter sensitivity.
#   - predict_input_par_from_hyperparameters: At a new vectorized value of
#     hyper_par, return an estimate of the new optimal input_par in a free
#     parameterization.
#   - predict_output_par_from_hyperparameters: At a new vectorized value of
#     hyper_par, return an estimate of the new output_par.  If linear is true,
#     a linear approximation is used to estimate the dependence of output_par
#     on input_par, otherwise a linear approximation is only used to estimate
#     the dependence of input_par on hyper_par.
class ParametricSensitivity(object):
    def __init__(self,
        objective_fun, input_par, output_par, hyper_par,
        input_to_output_converter,
        optimal_input_par=None,
        objective_hessian=None,
        hyper_par_objective_fun=None):

        warnings.warn(
            'ParametricSensitivity is deprecated.  ' +
            'Please use ParametricSensitivityTaylorExpansion or ' +
            'ParametricSensitivityLinearApproximation.',
            DeprecationWarning)

        self.input_par = input_par
        self.output_par = output_par
        self.hyper_par = hyper_par
        self.input_to_output_converter = input_to_output_converter
        self.objective_fun = objective_fun
        self.input_to_output_converter = input_to_output_converter

        if hyper_par_objective_fun is None:
            self.hyper_par_objective_fun = objective_fun
        else:
            self.hyper_par_objective_fun = hyper_par_objective_fun

        # For now, assume that the largest parameter is input_par.
        # TODO: detect automatically which is larger and choose the appropriate
        # sub-Hessian for maximal efficiency.
        self.parameter_converter = ParameterConverter(
            input_par, output_par, self.input_to_output_converter)
        self.objective = Objective(self.input_par, self.objective_fun)
        self.sensitivity_objective = TwoParameterObjective(
            self.input_par, self.hyper_par, self.hyper_par_objective_fun)

        self.set_optimal_input_par(optimal_input_par, objective_hessian)

    def set_optimal_input_par(
        self,
        optimal_input_par=None,
        objective_hessian=None):

        if optimal_input_par is None:
            self.optimal_input_par = self.input_par.get_free()
        else:
            self.optimal_input_par = deepcopy(optimal_input_par)

        if objective_hessian is None:
            self.objective_hessian = self.objective.fun_free_hessian(
                self.optimal_input_par)
        else:
            self.objective_hessian = objective_hessian
        self.hessian_chol = sp.linalg.cho_factor(self.objective_hessian)

        self.dout_din = self.parameter_converter.free_to_vec_jacobian(
            self.optimal_input_par)

        self.optimal_hyper_par = self.hyper_par.get_vector()
        self.hyper_par_cross_hessian = \
            self.sensitivity_objective.fun_hessian_free1_vector2(
                self.optimal_input_par, self.optimal_hyper_par)

        self.optimal_output_par = self.output_par.get_vector()
        self.hyper_par_sensitivity = \
            -1 * sp.linalg.cho_solve(
                self.hessian_chol, self.hyper_par_cross_hessian)

    def get_dinput_dhyper(self):
        return self.hyper_par_sensitivity

    def get_doutput_dhyper(self):
        return self.dout_din @ self.hyper_par_sensitivity

    def predict_input_par_from_hyperparameters(self, new_hyper_par):
        hyper_par_diff = new_hyper_par - self.optimal_hyper_par
        return \
            self.optimal_input_par + self.hyper_par_sensitivity @ hyper_par_diff

    def predict_output_par_from_hyperparameters(self, new_hyper_par, linear):
        if linear:
            hyper_par_diff = new_hyper_par - self.optimal_hyper_par
            return \
                self.optimal_output_par + \
                self.dout_din @ self.hyper_par_sensitivity @ hyper_par_diff
        else:
            return self.parameter_converter.converter_free_to_vec(
                self.predict_input_par_from_hyperparameters(new_hyper_par))

########################
# Other functions

# It's useful, especially when constructing sparse Hessians, to know
# which location in the parameter vector each variable goes.  The
# index parameter tells you that.
def make_index_param(param):
    index_param = deepcopy(param)
    index_param.set_vector(np.arange(0, index_param.vector_size()))
    return index_param


# Return a sparse matrix of size (full_hess_dim, full_hess_dim), where
# the entries of the dense matrix sub_hessian are in the locations
# indicated by the vector full_indices.
# TODO: test this formally.
def get_sparse_sub_hessian(sub_hessian, full_indices, full_hess_dim):
    return get_sparse_sub_matrix(
        sub_matrix=sub_hessian,
        row_indices=full_indices,
        col_indices=full_indices,
        row_dim=full_hess_dim,
        col_dim=full_hess_dim)


# Return a sparse matrix of size (full_hess_dim, full_hess_dim), where
# the entries of the dense matrix sub_hessian are in the locations
# indicated by the vector full_indices.
# TODO: test this formally.
def get_sparse_sub_matrix(
    sub_matrix, row_indices, col_indices, row_dim, col_dim):

    mat_vals = [] # These will be the entries of the Hessian
    mat_rows = [] # These will be the z indices
    mat_cols = [] # These will be the data indices

    for row in range(sub_matrix.shape[0]):
        for col in range(sub_matrix.shape[1]):
            if sub_matrix[row, col] != 0:
                mat_vals.append(sub_matrix[row, col])
                mat_rows.append(int(row_indices[row]))
                mat_cols.append(int(col_indices[col]))

    return sp.sparse.csr_matrix(
        (mat_vals, (mat_rows, mat_cols)), (row_dim, col_dim))



# Utilities for pickling and unpickling sparse matrices.
def pack_csr_matrix(sp_mat):
    sp_mat = sp.sparse.csr_matrix(sp_mat)
    return { 'data': sp_mat.data,
             'indices': sp_mat.indices,
             'indptr': sp_mat.indptr,
             'shape': sp_mat.shape }

def unpack_csr_matrix(sp_mat_dict):
    return sp.sparse.csr_matrix(
        ( sp_mat_dict['data'], sp_mat_dict['indices'], sp_mat_dict['indptr']),
        shape = sp_mat_dict['shape'])


# TODO: copy the tests over for these.

# Pack a sparse csr_matrix in a json-seralizable format.
def json_pack_csr_matrix(sp_mat):
    assert sparse.isspmatrix_csr(sp_mat)
    sp_mat = sparse.csr_matrix(sp_mat)
    return { 'data': json_tricks.dumps(sp_mat.data),
             'indices': json_tricks.dumps(sp_mat.indices),
             'indptr': json_tricks.dumps(sp_mat.indptr),
             'shape': sp_mat.shape,
             'type': 'csr_matrix' }


# Convert the output of pack_csr_matrix back into a csr_matrix.
def json_unpack_csr_matrix(sp_mat_dict):
    assert sp_mat_dict['type'] == 'csr_matrix'
    data = json_tricks.loads(sp_mat_dict['data'])
    indices = json_tricks.loads(sp_mat_dict['indices'])
    indptr = json_tricks.loads(sp_mat_dict['indptr'])
    return sparse.csr_matrix(
        ( data, indices, indptr), shape = sp_mat_dict['shape'])


# Get the matrix inverse square root of a symmetric matrix with eigenvalue
# thresholding.  This is particularly useful for calculating preconditioners.
def get_sym_matrix_inv_sqrt(block_hessian, ev_min=None, ev_max=None):
    raise DeprecationWarning(
        'Deprecated.  Use OptimizationUtils.get_sym_matrix_inv_sqrt instead.')

    hessian_sym = 0.5 * (block_hessian + block_hessian.T)
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
