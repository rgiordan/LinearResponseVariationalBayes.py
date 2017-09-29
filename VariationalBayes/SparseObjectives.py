import VariationalBayes as vb
from VariationalBayes.Parameters import \
    convert_vector_to_free_hessian
from VariationalBayes import ModelParamsDict
import autograd
import autograd.numpy as np
#import autograd.scipy as sp

import scipy as sp
from scipy import sparse

from copy import deepcopy

import time


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
        return np.squeeze(np.array(np.matmul(self.preconditioner, free_val)))

    def fun_free_cond(self, free_val, verbose=False):
        assert self.preconditioner is not None
        y = self.get_conditioned_x(free_val)
        return self.fun_free(y, verbose=verbose)

    def fun_free_grad_cond(self, free_val):
        assert self.preconditioner is not None
        y = self.get_conditioned_x(free_val)
        grad = self.fun_free_grad(y)
        return np.matmul(self.preconditioner.T, grad)

    def fun_free_hessian_cond(self, free_val):
        assert self.preconditioner is not None
        y = self.get_conditioned_x(free_val)
        hess = self.fun_free_hessian(y)
        return np.matmul(self.preconditioner.T,
                         np.matmul(hess, self.preconditioner))

    def fun_free_hvp_cond(self, free_val, vec):
        assert self.preconditioner is not None
        y = self.get_conditioned_x(free_val)
        return np.matmul(
            self.preconditioner.T,
            self.fun_free_hvp(y, np.matmul(self.preconditioner, vec)))

    # Convert the optimum of the conditioned problem to the
    # value (with tests to be sure you're doing it right).
    def uncondition_x(self, cond_x):
        return np.matmul(self.preconditioner, cond_x)


def make_index_param(param):
    index_param = deepcopy(param)
    index_param.set_vector(np.arange(0, index_param.vector_size()))
    return index_param

# A helper function to get sparse Hessians from models.
#
# Args:
#   - set_parameters_fun: A function taking a single argument (group) and
#     returning a vector of group parameters and a vector of indices in the
#     full model.
#   - get_group_hessian: A function taking a vector of group parameters and
#     returning a dense hessian for that group.
#   - group_range: The range of the group variable.
#   - full_hess_dim: The size of the full Hessian.
#   - print_every: Every n groups, print a little message.
#
# Returns:
#   - A sparse matrix consisting of the sum of all of the group hessians,
#     with entries in the appropriate locations for the full model.
# TODO: formally test this
def get_sparse_hessian(
    set_parameters_fun, get_group_hessian, group_range, full_hess_dim,
    print_every=20):

    hess_vals = [] # These will be the entries of the Hessian
    hess_rows = [] # These will be the z indices
    hess_cols = [] # These will be the data indices

    # Get the dimension using the first element of the group_range.
    group_vector, full_indices = set_parameters_fun(group_range[0])
    group_hess_dim = len(group_vector)

    group_ind = 0
    for group in group_range:
        group_ind += 1
        if group_ind % print_every == 0:
            print('Group {} of {}'.format(group_ind, len(group_range) - 1))
        group_vector, full_indices = set_parameters_fun(group)
        row_hess_val = np.squeeze(get_group_hessian(group_vector, group))

        for row in range(group_hess_dim):
            for col in range(group_hess_dim):
                if row_hess_val[row, col] != 0:
                    hess_vals.append(row_hess_val[row, col])
                    hess_rows.append(int(full_indices[row]))
                    hess_cols.append(int(full_indices[col]))

    print('Done.')
    return sp.sparse.csr_matrix(
        (hess_vals, (hess_rows, hess_cols)), (full_hess_dim, full_hess_dim))


# Return a sparse matrix of size (full_hess_dim, full_hess_dim), where
# the entries of the dense matrix sub_hessian are in the locations indicated
# by the vector full_indices.
# TODO: test this formally.
def get_sparse_sub_hessian(sub_hessian, full_indices, full_hess_dim):
    hess_vals = [] # These will be the entries of the Hessian
    hess_rows = [] # These will be the z indices
    hess_cols = [] # These will be the data indices

    # Get the dimension using the first element of the group_range.
    group_hess_dim = sub_hessian.shape[0]
    assert(sub_hessian.shape[0] == sub_hessian.shape[1])

    for row in range(group_hess_dim):
        for col in range(group_hess_dim):
            if sub_hessian[row, col] != 0:
                hess_vals.append(sub_hessian[row, col])
                hess_rows.append(int(full_indices[row]))
                hess_cols.append(int(full_indices[col]))

    return sp.sparse.csr_matrix(
        (hess_vals, (hess_rows, hess_cols)), (full_hess_dim, full_hess_dim))


# Utilities for pickling and unpickling sparse matrices.
def pack_csr_matrix(sp_mat):
    return { 'data': sp_mat.data,
             'indices': sp_mat.indices,
             'indptr': sp_mat.indptr,
             'shape': sp_mat.shape }

def unpack_csr_matrix(sp_mat_dict):
    return sp.sparse.csr_matrix(
        ( sp_mat_dict['data'], sp_mat_dict['indices'], sp_mat_dict['indptr']),
        shape = sp_mat_dict['shape'])
