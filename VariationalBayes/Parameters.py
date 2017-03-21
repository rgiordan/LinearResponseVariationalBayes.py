import math
import copy
import numbers

import autograd.numpy as np
from autograd.core import primitive

from collections import OrderedDict

def unconstrain_vector(vec, lb, ub):
    if not all(vec <= ub):
        raise ValueError('Elements larger than the upper bound')
    if not all(vec >= lb):
        raise ValueError('Elements smaller than the lower bound')
    return unconstrain(vec, lb, ub)


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


class ScalarParam(object):
    def __init__(self, name, lb=-float("inf"), ub=float("inf"), val=None):
        self.name = name
        if lb >= ub:
            raise ValueError('Upper bound must strictly exceed lower bound')
        self.__lb = lb
        self.__ub = ub
        if val is None:
            self.__val = 0.5 * (ub + lb)
        else:
            self.set(val)
    def __str__(self):
        return self.name + ': ' + str(self.__val)
    def names(self):
        return [ self.name ]

    def set(self, val):
        # Asserting that you are getting something of length one doesn't
        # seem trivial in python.
        # if not isinstance(val, numbers.Number):
        #     if len(val) != 1:
        #         raise ValueError('val must be a number or length-one array.')
        if val <= self.__lb:
            raise ValueError('val is less than the lower bound.')
        if val >= self.__ub:
            raise ValueError('val is less than the lower bound.')
        self.__val = val
    def get(self):
        return self.__val

    def set_free(self, free_val):
        self.set(constrain(free_val, self.__lb, self.__ub))
    def get_free(self):
        return unconstrain_scalar(self.__val, self.__lb, self.__ub)

    def set_vector(self, val):
        self.set(val)
    def get_vector(self):
        return self.__val

    def size(self):
        return 1
    def free_size(self):
        return 1
    def vector_size(self):
        return 1


class VectorParam(object):
    def __init__(self, name, size, lb=-float("inf"), ub=float("inf"), val=None):
        self.name = name
        self.__size = size
        self.__lb = lb
        self.__ub = ub
        if val is None:
            self.__val = np.empty(size)
        else:
            self.set(val)
        if lb >= ub:
            raise ValueError('Upper bound must strictly exceed lower bound')
    def __str__(self):
        return self.name + ':\n' + str(self.__val)
    def names(self):
        return [ self.name + '_' + str(k) for k in range(self.size()) ]

    def set(self, val):
        if val.size != self.size():
            raise ValueError('Wrong size for vector ' + self.name)
        if any(val <= self.__lb):
            raise ValueError('Value beneath lower bound.')
        if any(val >= self.__ub):
            raise ValueError('Value above upper bound.')
        self.__val = val
    def get(self):
        return self.__val

    def set_free(self, free_val):
        if free_val.size != self.size():
            raise ValueError('Wrong size for vector ' + self.name)
        self.set(constrain(free_val, self.__lb, self.__ub))
    def get_free(self):
        return unconstrain_vector(self.__val, self.__lb, self.__ub)

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


# Uses 0-indexing. (row, col) = (k1, k2)
def SymIndex(k1, k2):
    def LDInd(k1, k2):
        return k2 + k1 * (k1 + 1) / 2

    if k2 <= k1:
        return LDInd(k1, k2)
    else:
        return LDInd(k2, k1)


def VectorizeLDMatrix(mat):
    nrow, ncol = np.shape(mat)
    if nrow != ncol: raise ValueError('mat must be square')
    return mat[np.tril_indices(nrow)]


# Because we cannot use autograd with array assignment, just define the
# vector jacobian product directly.
@primitive
def UnvectorizeLDMatrix(vec):
    mat_size = int(0.5 * (math.sqrt(1 + 8 * vec.size) - 1))
    if mat_size * (mat_size + 1) / 2 != vec.size: \
        raise ValueError('Vector is an impossible size')
    mat = np.zeros((mat_size, mat_size))
    for k1 in range(mat_size):
        for k2 in range(k1 + 1):
            mat[k1, k2] = vec[SymIndex(k1, k2)]
    return mat


# TODO: test this as a derivative
def UnvectorizeLDMatrix_vjp(g, ans, vs, gvs, vec):
    assert g.shape[0] == g.shape[1]
    # mat_size = g.shape[0]
    # vjp = np.zeros(mat_size * (mat_size + 1) / 2)
    return VectorizeLDMatrix(g)
    # for k1 in range(mat_size):
    #     for k2 in range(k1 + 1):
    #         vjp[SymIndex(k1, k2)] += g[k1, k2]
    # return vjp

UnvectorizeLDMatrix.defvjp(UnvectorizeLDMatrix_vjp)

def pack_posdef_matrix(mat):
    return VectorizeLDMatrix(np.linalg.cholesky(mat))


def unpack_posdef_matrix(free_vec):
    mat_chol = UnvectorizeLDMatrix(free_vec)
    return np.matmul(mat_chol, mat_chol.T)


class PosDefMatrixParam(object):
    def __init__(self, name, size, val=None):
        self.name = name
        self.__size = size
        self.__vec_size = size * (size + 1) / 2
        if val is None:
            self.__val = np.matrix(np.zeros([size, size]))
        else:
            self.set(val)
    def __str__(self):
        return self.name + ':\n' + str(self.__val)
    def names(self):
        return [ self.name ]

    def set(self, val):
        nrow, ncol = np.shape(val)
        if nrow != self.__size or ncol != self.__size:
            raise ValueError('Matrix is a different size')
        if not (val.transpose() == val).all():
            raise ValueError('Matrix is not symmetric')
        self.__val = val
    def get(self):
        return self.__val

    def set_free(self, free_val):
        if free_val.size != self.__vec_size:
            raise ValueError('Free value is the wrong length')
        self.set(unpack_posdef_matrix(free_val))
    def get_free(self):
        return pack_posdef_matrix(self.__val)

    def set_vector(self, vec_val):
        if vec_val.size != self.__vec_size:
            raise ValueError('Vector value is the wrong length')
        ld_mat = UnvectorizeLDMatrix(vec_val)
        mat_val = ld_mat + ld_mat.transpose()
        # We have double counted the diagonal.  For some reason the autograd
        # diagonal functions require axis1=-1 and axis2=-2
        mat_val = mat_val - \
            np.make_diagonal(np.diagonal(ld_mat, axis1=-1, axis2=-2),
                             axis1=-1, axis2=-2)
        self.set(mat_val)
    def get_vector(self):
        return VectorizeLDMatrix(self.__val)

    def size(self):
        return self.__size
    def free_size(self):
        return self.__vec_size
    def vector_size(self):
        return self.__vec_size


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


class ModelParamsDict(object):
    def __init__(self, name='ModelParamsDict'):
        self.param_dict = OrderedDict()
        self.__name = name
        # You will want free_size and vector_size to be different when you
        # are encoding simplexes.
        self.__free_size = 0
        self.__vector_size = 0
    def __str__(self):
        return self.__name + ':\n' + \
            '\n'.join([ '\t' + str(param) for param in self.param_dict.values() ])
    def __getitem__(self, key):
        return self.param_dict[key]
    def push_param(self, param):
        self.param_dict[param.name] = param
        self.__free_size = self.__free_size + param.free_size()
        self.__vector_size = self.__vector_size + param.vector_size()
    def set_name(self, name):
        self.__name = name

    def set_free(self, vec):
        if vec.size != self.__free_size: raise ValueError("Wrong size.")
        offset = 0
        for param in self.param_dict.values():
            offset = set_free_offset(param, vec, offset)
    def get_free(self):
        vec = np.empty(self.free_size())
        offset = 0
        for param in self.param_dict.values():
            offset = get_free_offset(param, vec, offset)
        return vec

    def set_vector(self, vec):
        if vec.size != self.__vector_size: raise ValueError("Wrong size.")
        offset = 0
        for param in self.param_dict.values():
            offset = set_vector_offset(param, vec, offset)
    def get_vector(self):
        vec = np.empty(self.vector_size())
        offset = 0
        for param in self.param_dict.values():
            offset = get_vector_offset(param, vec, offset)
        return vec

    def names(self):
        return np.concatenate([ param.names() for param in self.param_dict.values()])
    def free_size(self):
        return self.__free_size
    def vector_size(self):
        return self.__vector_size


# Not to be confused with a VectorParam -- this is a vector of abstract
# parameter types.  Note that for the purposes of vectorization it might
# be better to use an object with arrays of attributes rather than an array
# of parameters with singelton attributes.
# This is not currently tested.
class ParamVector(object):
    def __init__(self, name, param_vec):
        self.name = name
        self.params = param_vec
        self.__free_size = np.sum([ par.free_size() for par in self.params ])
    def __str__(self):
        return '\n'.join([ str(par) for par in self.params ])
    def __len__(self):
        return len(self.params)
    def names(self):
        return '\n'.join([ names(par) for par in self.params ])
    def set_free(self, free_val):
        if free_val.size != self.__free_size:
            raise ValueError('Wrong size for ParamVector ' + self.name)
        offset = 0
        for par in self.params:
            offset = set_free_offset(par, free_val, offset)
    def get_free(self):
        vec = np.empty(self.__free_size)
        offset = 0
        for par in self.params:
            offset = get_free_offset(par, vec, offset)
        return vec
    def free_size(self):
        return self.__free_size
    def __getitem__(self, key):
        return self.params[key]
    def __setitem__(self, key, value):
        self.params[key] = value
