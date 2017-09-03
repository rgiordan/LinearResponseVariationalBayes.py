import VariationalBayes as vb
from VariationalBayes.Parameters import \
    convert_vector_to_free_hessian
from VariationalBayes import ModelParamsDict
import autograd
import autograd.numpy as np
#import autograd.scipy as sp

import scipy as sp
from scipy import sparse

import time


class Logger(object):
    def __init__(self, print_every=1):
        self.print_every = print_every
        self.initialize()
        self.print_x_diff = True

    def initialize(self):
        self.iter = 0
        self.last_x = None

    def log(self, value, x):
        if self.last_x is None:
            x_diff = float('inf')
        else:
            x_diff = np.max(np.abs(x - self.last_x))

        self.last_x = x
        if self.iter % self.print_every == 0:
            print('Iter ', self.iter, ' value: ', value)
            if self.print_x_diff:
                print('\tx_diff: ', x_diff)
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

# Note: the function get_sparse_hessian, with the associated template in
# LogisticGLMM, probably makes this class unnecessary.
#
# As before, fun() should evaluate to a float but now should be bound
# to both global_par and local_par.  fun_grad_local() and fun_hess_local()
# should take no arguments, be bound to the values of global_par and local_par,
# and respectively return gradients and hessians of fun() with respect
# to the vector representations of the local parameters.  This is most useful
# when the local grad and / or hessian are sparse.
class SparseObjective(object):
    def __init__(self, par, fun,
                 global_par_name='global',
                 local_par_name='local',
                 fun_vector_local_grad=None,
                 fun_vector_local_hessian=None):

        self.par = par
        self.global_par = self.par[global_par_name]
        self.local_par = self.par[local_par_name]
        self.fun = fun

        # par should contain only the global and local params, and in
        # that order.
        assert len(self.par.param_dict.items()) == 2
        par_keys = list(self.par.param_dict.keys())
        assert par_keys[0] == global_par_name
        assert par_keys[1] == local_par_name

        # Use the dense autograd defaults if the local gradients and
        # Hessian are not specified.
        if not fun_vector_local_grad:
            self.fun_vector_local_grad = \
                autograd.grad(self.fun_vector_split, argnum=1)
        else:
            self.fun_vector_local_grad = fun_vector_local_grad

        if not fun_vector_local_hessian:
            self.fun_vector_local_hessian = \
                autograd.hessian(self.fun_vector_split, argnum=1)
        else:
            self.fun_vector_local_hessian = fun_vector_local_hessian

        # Use autograd to define the global derivatives.
        self.fun_free_global_grad = \
            autograd.grad(self.fun_free_split, argnum=0)
        self.fun_free_global_hessian = \
            autograd.hessian(self.fun_free_split, argnum=0)

        # Use autograd for the Hessian vector product.
        self.fun_free_hvp = \
            autograd.hessian_vector_product(self.fun_free)

        self.fun_vector_global_grad = \
            autograd.grad(self.fun_vector_split, argnum=0)
        self.fun_vector_global_hessian = \
            autograd.hessian(self.fun_vector_split, argnum=0)

        # Calculate the vector Hessian cross term using autograd.
        # TODO: I can't remember which direction is most efficient.
        # I think differentiating the big vector first is best, but maybe
        # it's the other way around.
        # Note: I'm using the dense version here to avoid requiring the user-
        # defined on to be differentiable by autograd.
        # self.fun_vector_local_grad_dense = \
        #     autograd.grad(self.fun_vector_split, argnum=1)
        # self.fun_vector_cross_hessian = autograd.jacobian(
        #     self.fun_vector_local_grad_dense, argnum=0)

        self.fun_vector_cross_hessian = autograd.jacobian(
            self.fun_vector_global_grad, argnum=1)

    def fun_free(self, free_val, verbose=False):
        self.par.set_free(free_val)
        val = self.fun()
        if verbose:
            print('Value: ', val)
        return val

    def fun_free_split(self, global_free_val, local_free_val):
        self.global_par.set_free(global_free_val)
        self.local_par.set_free(local_free_val)
        return self.fun()

    def fun_vector(self, vec_val):
        self.par.set_vector(vec_val)
        return self.fun()

    # Define the vector derivatives.
    def fun_vector_split(self, global_vec_val, local_vec_val):
        self.global_par.set_vector(global_vec_val)
        self.local_par.set_vector(local_vec_val)
        return self.fun()

    def fun_vector_grad_split(self, global_vec_val, local_vec_val):
        sp_grad = sparse.hstack([
            self.fun_vector_global_grad(global_vec_val, local_vec_val),
            self.fun_vector_local_grad(global_vec_val, local_vec_val) ])

        # TODO: maybe sometimes you want the sparse version.
        return np.squeeze(np.array(sp_grad.T.toarray()))

    # Define the free derivatives as trasnforms of the vector derivatives.
    def fun_vector_hessian_split(self, global_vec_val, local_vec_val):
        self.global_par.set_vector(global_vec_val)
        self.local_par.set_vector(local_vec_val)

        global_hess = self.fun_vector_global_hessian(
            global_vec_val, local_vec_val)

        cross_hess = self.fun_vector_cross_hessian(
            global_vec_val, local_vec_val).T

        local_hess = self.fun_vector_local_hessian(
            global_vec_val, local_vec_val)
        sp_hess =  sp.sparse.bmat([ [global_hess,  cross_hess.T],
                                    [cross_hess,   local_hess]], format='csr')

        # TODO: maybe sometimes you want the sparse version.
        return np.array(sp_hess.toarray())

    # Use the sparse transforms to change the vector gradiend into a free
    # gradient.
    def fun_free_grad_sparse(self, free_val):
        self.par.set_free(free_val)
        global_vec_val = self.global_par.get_vector()
        local_vec_val = self.local_par.get_vector()

        fun_vector_grad = self.fun_vector_grad_split(
            global_vec_val, local_vec_val)
        free_to_vec_jacobian = self.par.free_to_vector_jac(free_val)

        # If you don't convert to an array, it returns a matrix type, which
        # seems to cause mysterious problems with scipy.optimize.minimize.
        free_grad = np.squeeze(free_to_vec_jacobian.T * fun_vector_grad)
        return free_grad

    # Use the sparse transforms to change the vector Hessian into a free
    # Hessian.
    def fun_free_hessian_sparse(self, free_val):
        self.par.set_free(free_val)
        global_vec_val = self.global_par.get_vector()
        local_vec_val = self.local_par.get_vector()

        fun_vector_hessian = self.fun_vector_hessian_split(
            global_vec_val, local_vec_val)

        fun_vector_grad = self.fun_vector_grad_split(
            global_vec_val, local_vec_val)

        # This is taking most of the time.
        fun_hessian_sparse = convert_vector_to_free_hessian(
            self.par, free_val, fun_vector_grad, fun_vector_hessian)

        # If you don't convert to an array, it returns a matrix type, which
        # seems to cause mysterious problems with scipy.optimize.minimize.
        return np.array(fun_hessian_sparse)


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

    group_vector, full_indices = set_parameters_fun(0)
    group_hess_dim = len(group_vector)

    for group in group_range:
        if group % print_every == 0:
            print('Group {} of {}'.format(group, np.max(group_range) - 1))
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


# TODO: test these formally

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
