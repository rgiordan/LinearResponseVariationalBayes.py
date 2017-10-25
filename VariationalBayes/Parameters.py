import math
import copy
import numbers

import autograd
import autograd.numpy as np
import autograd.scipy as sp
from autograd.core import primitive

import scipy as osp
from scipy.sparse import coo_matrix, csr_matrix, block_diag

def unconstrain_array(vec, lb, ub):
    if not (vec <= ub).all():
        raise ValueError('Elements larger than the upper bound')
    if not (vec >= lb).all():
        raise ValueError('Elements smaller than the lower bound')
    return unconstrain(vec, lb, ub).flatten()


def unconstrain_scalar(val, lb, ub):
    if not val <= ub:
        raise ValueError('Value larger than the upper bound')
    if not val >= lb:
        raise ValueError('Value smaller than the lower bound')
    return unconstrain(val, lb, ub)


def unconstrain(vec, lb, ub):
    if ub <= lb:
        raise ValueError('Upper bound must be greater than lower bound')
    if ub == float("inf"):
        if lb == -float("inf"):
            # TODO: I'm not sure this copy work with autodiff.
            return copy.copy(vec)
        else:
            return np.log(vec - lb)
    else: # the upper bound is finite
        if lb == -float("inf"):
            return -1 * np.log(ub - vec)
        else:
            return np.log(vec - lb) - np.log(ub - vec)


def constrain(free_vec, lb, ub):
    if ub <= lb:
        raise ValueError('Upper bound must be greater than lower bound')
    if ub == float("inf"):
        if lb == -float("inf"):
            # TODO: I'm not sure this copy work with autodiff.
            return copy.copy(free_vec)
        else:
            return np.exp(free_vec) + lb
    else: # the upper bound is finite
        if lb == -float("inf"):
            return ub - np.exp(-1 * free_vec)
        else:
            exp_vec = np.exp(free_vec)
            return (ub - lb) * exp_vec / (1 + exp_vec) + lb

constrain_scalar_jac = autograd.jacobian(constrain)
constrain_scalar_hess = autograd.hessian(constrain)

def get_inbounds_value(lb, ub):
    assert lb < ub
    if lb > -float('inf') and ub < float('inf'):
        return 0.5 * (ub - lb)
    else:
        if lb > -float('inf'):
            # The upper bound is infinite.
            return lb + 1.0
        elif ub < float('inf'):
            # The lower bound is infinite.
            return ub - 1.0
        else:
            # Both are infinie.
            return 0.0


class ScalarParam(object):
    def __init__(self, name='', lb=-float('inf'), ub=float('inf'), val=None):
        self.name = name
        if lb >= ub:
            raise ValueError('Upper bound must strictly exceed lower bound')
        self.__lb = lb
        self.__ub = ub
        assert lb >= -float('inf')
        assert ub <= float('inf')
        if val is not None:
            self.set(val)
        else:
            self.set(get_inbounds_value(lb, ub))

        self.free_to_vector_jac_dense = autograd.jacobian(self.free_to_vector)
        self.free_to_vector_hess_dense = autograd.hessian(self.free_to_vector)

    def __str__(self):
        return self.name + ': ' + str(self.__val)
    def names(self):
        return [ self.name ]
    def dictval(self):
        # Assume it's either a numpy array or a float. :(
        if isinstance(self.__val, numbers.Number):
            return self.__val
        else:
            return self.__val.tolist()

    def set(self, val):
        # TODO: Make checking an option.
        # if val < self.__lb:
        #     error_msg = 'val is less than the lower bound: ' + \
        #         str(val) + ' <= ' + str(self.__lb)
        #     raise ValueError(error_msg)
        # if val > self.__ub:
        #     raise ValueError('val is greater than the upper bound.')
        self.__val = val
    def get(self):
        return self.__val

    def set_free(self, free_val):
        self.set(constrain(free_val, self.__lb, self.__ub))
    def get_free(self):
        return np.reshape(unconstrain_scalar(self.__val, self.__lb, self.__ub), 1)

    def free_to_vector(self, free_val):
        self.set_free(free_val)
        return self.get()
    def free_to_vector_jac(self, free_val):
        return coo_matrix(self.free_to_vector_jac_dense(free_val))
    def free_to_vector_hess(self, free_val):
        hess_dense = self.free_to_vector_hess_dense(free_val)
        return np.array([ coo_matrix(hess_dense[ind, :, :])
                          for  ind in range(hess_dense.shape[0]) ])

    def set_vector(self, val):
        self.set(val)
    def get_vector(self):
        return np.reshape(self.__val, 1)

    def size(self):
        return 1
    def free_size(self):
        return 1
    def vector_size(self):
        return 1


# TODO: perhaps this could just be replaced by ArrayParam.
class VectorParam(object):
    def __init__(self, name='', size=1, lb=-float("inf"), ub=float("inf"),
                 val=None):
        self.name = name
        self.__size = int(size)
        self.__lb = lb
        self.__ub = ub
        assert lb >= -float('inf')
        assert ub <= float('inf')
        if lb >= ub:
            raise ValueError('Upper bound must strictly exceed lower bound')
        if val is not None:
            self.set(val)
        else:
            inbounds_value = get_inbounds_value(lb, ub)
            self.set(np.full(self.__size, inbounds_value))

    def __str__(self):
        return self.name + ':\n' + str(self.__val)
    def names(self):
        return [ self.name + '_' + str(k) for k in range(self.size()) ]
    def dictval(self):
        return self.__val.tolist()

    def set(self, val):
        if val.size != self.size():
            raise ValueError('Wrong size for vector ' + self.name)
        # if (np.array(val < self.__lb)).any():
        #     raise ValueError('Value beneath lower bound.')
        # if (np.array(val > self.__ub)).any():
        #     raise ValueError('Value above upper bound.')
        self.__val = val
    def get(self):
        return self.__val

    def set_free(self, free_val):
        if free_val.size != self.size():
            raise ValueError('Wrong size for vector ' + self.name)
        self.set(constrain(free_val, self.__lb, self.__ub))
    def get_free(self):
        return unconstrain_array(self.__val, self.__lb, self.__ub)
    def free_to_vector(self, free_val):
        self.set_free(free_val)
        return self.get_vector()
    def free_to_vector_jac(self, free_val):
        rows_indices = np.array(range(self.vector_size()))
        grads = [ constrain_scalar_jac(free_val[vec_ind], self.__lb, self.__ub) \
                  for vec_ind in range(self.vector_size()) ]
        return coo_matrix((grads,
                          (rows_indices, rows_indices)),
                          (self.vector_size(), self.free_size()))
    def free_to_vector_hess(self, free_val):
        def get_ind_hess(vec_ind):
            hess = constrain_scalar_hess(free_val[vec_ind], self.__lb, self.__ub)
            return coo_matrix(([ hess ],
                               ([vec_ind], [vec_ind])),
                               (self.free_size(), self.vector_size()))
        return np.array([ get_ind_hess(vec_ind)
                          for vec_ind in range(self.vector_size()) ])


    def set_vector(self, val):
        self.set(val)
    def get_vector(self):
        return self.__val

    def size(self):
        return self.__size
    def free_size(self):
        return self.__size
    def vector_size(self):
        return self.__size


class ArrayParam(object):
    def __init__(self, name='', shape=(1, 1),
                 lb=-float("inf"), ub=float("inf"), val=None):
        self.name = name
        self.__shape = shape
        self.__lb = lb
        self.__ub = ub
        assert lb >= -float('inf')
        assert ub <= float('inf')
        if lb >= ub:
            raise ValueError('Upper bound must strictly exceed lower bound')
        if val is not None:
            self.set(val)
        else:
            inbounds_value = get_inbounds_value(lb, ub)
            self.set(np.full(self.__shape, inbounds_value))

    def __str__(self):
        return self.name + ':\n' + str(self.__val)
    def names(self):
        return self.name
    def dictval(self):
        return self.__val.tolist()

    def set(self, val):
        if val.shape != self.shape():
            raise ValueError('Wrong size for array ' + self.name + \
                             ' Expected shape: ' + str(self.shape()) + \
                             ' Got shape: ' + str(val.shape))
        # if (np.array(val < self.__lb)).any():
        #     raise ValueError('Value beneath lower bound.')
        # if (np.array(val > self.__ub)).any():
        #     raise ValueError('Value above upper bound.')
        self.__val = val
    def get(self):
        return self.__val

    def set_free(self, free_val):
        if free_val.size != self.free_size():
            error_string = \
                'Wrong size for array {}.  Expected {}, got {}'.format(
                    self.name, str(self.free_size()), str(free_val.size))
            raise ValueError(error_string)
        self.set(constrain(free_val, self.__lb, self.__ub).reshape(self.__shape))
    def get_free(self):
        return unconstrain_array(self.__val, self.__lb, self.__ub)
    def free_to_vector(self, free_val):
        self.set_free(free_val)
        return self.get_vector()
    def free_to_vector_jac(self, free_val):
        rows_indices = np.array(range(self.vector_size()))
        grads = [ constrain_scalar_jac(free_val[vec_ind], self.__lb, self.__ub) \
                  for vec_ind in range(self.vector_size()) ]
        return coo_matrix((grads,
                          (rows_indices, rows_indices)),
                          (self.vector_size(), self.free_size()))
    def free_to_vector_hess(self, free_val):
        def get_ind_hess(vec_ind):
            hess = constrain_scalar_hess(free_val[vec_ind], self.__lb, self.__ub)
            return coo_matrix(([ hess ],
                               ([vec_ind], [vec_ind])),
                               (self.free_size(), self.vector_size()))
        return np.array([ get_ind_hess(vec_ind)
                          for vec_ind in range(self.vector_size()) ])

    def set_vector(self, val):
        if val.size != self.vector_size():
            error_string = \
                'Wrong size for array {}.  Expected {}, got {}'.format(
                    self.name, str(self.vector_size()), str(val.size))
            raise ValueError(error_string)
        self.set(val.reshape(self.__shape))
    def get_vector(self):
        return self.__val.flatten()

    def shape(self):
        return self.__shape
    def free_size(self):
        return int(np.product(self.__shape))
    def vector_size(self):
        return int(np.product(self.__shape))

# Sets the param using the slice in free_vec starting at offset.
# Returns the next offset.
def set_free_offset(param, free_vec, offset):
    param.set_free(free_vec[offset:(offset + param.free_size())])
    return offset + param.free_size()

# Sets the value of vec starting at offset with the param's free value.
# Returns the next offset.
def get_free_offset(param, vec, offset):
    vec[offset:(offset + param.free_size())] = param.get_free()
    return offset + param.free_size()


# Sets the param using the slice in free_vec starting at offset.
# Returns the next offset.
def set_vector_offset(param, vec, offset):
    param.set_vector(vec[offset:(offset + param.vector_size())])
    return offset + param.vector_size()


# Sets the value of vec starting at offset with the param's free value.
# Returns the next offset.
def get_vector_offset(param, vec, offset):
    vec[offset:(offset + param.vector_size())] = param.get_vector()
    return offset + param.vector_size()


def free_to_vector_jac_offset(param, free_vec, free_offset, vec_offset):
    free_slice = slice(free_offset, free_offset + param.free_size())
    jac = param.free_to_vector_jac(free_vec[free_slice])
    return free_offset + param.free_size(), \
           vec_offset + param.vector_size(), \
           jac


# Define a sparse matrix with spmat offset by offset_shape and with
# total shape give by total_shape.  This is useful for placing a sub-Hessian
# in the middle of a larger Hessian.
def offset_sparse_matrix(spmat, offset_shape, full_shape):
    return coo_matrix(
        (spmat.data,
         (spmat.row + offset_shape[0], spmat.col + offset_shape[1])),
         shape=full_shape)


# Append the parameter Hessian to the array of full sparse hessians and
# return the amount by which to increment the offset in the free vector.
def free_to_vector_hess_offset(
    param, free_vec, hessians, free_offset, full_shape):
    free_slice = slice(free_offset, free_offset + param.free_size())
    hess = param.free_to_vector_hess(free_vec[free_slice])
    for vec_ind in range(len(hess)):
        hessians.append(offset_sparse_matrix(
            hess[vec_ind], (free_offset, free_offset), full_shape))
    return free_offset + param.free_size()


# Using sparse jacobians and hessians, convert a hessian with respect
# to a parameters vector to a hessian with respect to the free parameters.
#
# Args:
#   - param: A Parameter type (from the Parameters library)
#   - free_val: The unconstrained "free value" of the parameters.
#   - vector_grad: The gradient of the objective with respect to the
#     constrained "vector value" of the parameters
#   - vector_hess: The (possibly sparse) Hessian of the objective with
#     respect to the constrained "vector value" of the parameters.
#
# Returns:
#  - The Hessian of the objective with respect to the unconstrained "free"
#    values of the parameters.
def convert_vector_to_free_hessian(param, free_val, vector_grad, vector_hess):
    #free_hess = csr_matrix((param.free_size(), param.free_size()))

    param.set_free(free_val)
    free_to_vec_jacobian = param.free_to_vector_jac(free_val)
    free_to_vec_hessian = param.free_to_vector_hess(free_val)

    # Accumulate the third order terms, which are sparse.  Use the fact
    # that elements of a coo_matrix add when converted to any other type.
    free_hess_size = (param.free_size(), param.free_size())
    vec_range = range(param.vector_size())
    free_hess_vals = np.hstack([
        free_to_vec_hessian[vec_ind].data * vector_grad[vec_ind]
        for vec_ind in vec_range ])
    free_hess_rows = np.hstack([
        free_to_vec_hessian[vec_ind].row for vec_ind in vec_range ])
    free_hess_cols = np.hstack([
        free_to_vec_hessian[vec_ind].col for vec_ind in vec_range ])
    free_hess = coo_matrix(
        (free_hess_vals, (free_hess_rows, free_hess_cols)), free_hess_size)

    # Then add the second-order terms, which may be dense depending on the
    # vec_hess_target.
    free_hess += \
        free_to_vec_jacobian.T * vector_hess * free_to_vec_jacobian

    return free_hess











#
