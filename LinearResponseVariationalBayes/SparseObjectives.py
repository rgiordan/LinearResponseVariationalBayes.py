import LinearResponseVariationalBayes as vb
from LinearResponseVariationalBayes.Parameters import \
    convert_vector_to_free_hessian
from LinearResponseVariationalBayes import ModelParamsDict
import autograd
import autograd.numpy as np
#import autograd.scipy as sp

import scipy as sp
from scipy import sparse

from copy import deepcopy

import time


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


# par should be a Parameter type.
# fun should be a function that takes no arguments but which is
# bound to par, i.e. which evaluates to a float that depends on the
# value of the parameters par.
class Objective(object):
    def __init__(self, par, fun):
        self.par = par
        self.fun = fun

        self.fun_free_grad = autograd.grad(self.fun_free)
        self.fun_free_hessian = autograd.hessian(self.fun_free)
        self.fun_free_hvp = autograd.hessian_vector_product(self.fun_free)

        self.fun_vector_grad = autograd.grad(self.fun_vector)
        self.fun_vector_hessian = autograd.hessian(self.fun_vector)
        self.fun_vector_hvp = autograd.hessian_vector_product(self.fun_vector)

        self.preconditioner = None
        self.logger = Logger()

    def fun_free(self, free_val, verbose=False):
        self.par.set_free(free_val)
        val = self.fun()
        if verbose:
            self.logger.log(val, free_val)
        return val

    def fun_vector(self, vec_val):
        self.par.set_vector(vec_val)
        return self.fun()

    # Pre-conditioned versions of the free functions.  The value at which
    # they are evaluted is assumed to include the preconditioner, i.e.
    # to be free_val = a * x
    def get_conditioned_x(self, free_val):
        return safe_matmul(self.preconditioner, free_val)

    def fun_free_cond(self, free_val, verbose=False):
        assert self.preconditioner is not None
        y = self.get_conditioned_x(free_val)
        return self.fun_free(y, verbose=verbose)

    def fun_free_grad_cond(self, free_val):
        assert self.preconditioner is not None
        y = self.get_conditioned_x(free_val)
        grad = self.fun_free_grad(y)
        return safe_matmul(self.preconditioner.T, grad)

    def fun_free_hessian_cond(self, free_val):
        assert self.preconditioner is not None
        y = self.get_conditioned_x(free_val)
        hess = self.fun_free_hessian(y)
        return safe_matmul(self.preconditioner.T,
                           safe_matmul(hess, self.preconditioner))

    def fun_free_hvp_cond(self, free_val, vec):
        assert self.preconditioner is not None
        y = self.get_conditioned_x(free_val)
        return safe_matmul(
            self.preconditioner.T,
            self.fun_free_hvp(y, safe_matmul(self.preconditioner, vec)))

    # Convert the optimum of the conditioned problem to the
    # value (with tests to be sure you're doing it right).
    def uncondition_x(self, cond_x):
        return safe_matmul(self.preconditioner, cond_x)


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
