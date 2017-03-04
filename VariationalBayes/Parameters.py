import math
#import numpy as np # Won't work with autodiff
import autograd.numpy as np
import copy

def unconstrain_vector(vec, lb, ub):
    if not all(vec <= ub): raise ValueError('Elements larger than the upper bound')
    if not all(vec >= lb): raise ValueError('Elements smaller than the lower bound')
    return unconstrain(vec, lb, ub)


def unconstrain_scalar(val, lb, ub):
    if not val <= ub: raise ValueError('Value larger than the upper bound')
    if not val >= lb: raise ValueError('Value smaller than the lower bound')
    return unconstrain(val, lb, ub)


def unconstrain(vec, lb, ub):
    if ub <= lb: raise ValueError('Upper bound must be greater than lower bound')
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
    if ub <= lb: raise ValueError('Upper bound must be greater than lower bound')
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


class VectorParam(object):
    def __init__(self, name, size, lb=-float("inf"), ub=float("inf")):
        self.name = name
        self.__size = size
        self.__free_size = size
        self.__val = np.empty(size)
        if lb >= ub: raise ValueError('Upper bound must strictly exceed lower bound')
        self.__lb = lb
        self.__ub = ub
    def __str__(self):
        return self.name + ':\n' + str(self.__val)
    def names(self):
        return [ self.name + '_' + str(k) for k in range(self.size()) ]
    def set(self, val):
        if val.size != self.size(): raise ValueError('Wrong size for vector ' + self.name)
        self.__val = val
    def get(self):
        return self.__val
    def set_free(self, free_val):
        if free_val.size != self.size(): raise ValueError('Wrong size for vector ' + self.name)
        self.set(constrain(free_val, self.__lb, self.__ub))
    def get_free(self):
        return unconstrain_vector(self.__val, self.__lb, self.__ub)
    def size(self):
        return self.__size
    def free_size(self):
        return self.__free_size

    
class ScalarParam(object):
    def __init__(self, name, lb=-float("inf"), ub=float("inf")):
        self.name = name
        if lb >= ub: raise ValueError('Upper bound must strictly exceed lower bound')
        self.__val = 0.5 * (ub + lb)
        self.__lb = lb
        self.__ub = ub
    def __str__(self):
        return self.name + ': ' + str(self.__val)
    def names(self):
        return [ self.name ]
    def set(self, val):
        self.__val = val
    def get(self):
        return self.__val
    def set_free(self, free_val):
        self.set(constrain(free_val, self.__lb, self.__ub))
    def get_free(self):
        return unconstrain_scalar(self.__val, self.__lb, self.__ub)
    def size(self):
        return 1
    def free_size(self):
        return 1


# Uses 0-indexing.
def sym_index(row, col):
    if row <= col:
        return col * (col + 1) / 2 + row
    else:
        return row * (row + 1) / 2 + col
        

def vectorize_matrix(mat, ld=True):
    nrow, ncol = np.shape(mat)
    if nrow != ncol: raise ValueError('mat must be square')
    vec_size = nrow * (nrow + 1) / 2
    vec = np.empty(vec_size)
    for col in range(ncol):
        for row in range(col + 1):
            if ld:
                vec[sym_index(row, col)] = mat[col, row]                
            else:
                vec[sym_index(row, col)] = mat[row, col]
    return vec


def unvectorize_matrix(vec, ld_only=False):
    mat_size = int(0.5 * (math.sqrt(1 + 8 * vec.size) - 1))
    if mat_size * (mat_size + 1) / 2 != vec.size: raise ValueError('Vector is an impossible size')
    mat = np.matrix(np.zeros([ mat_size, mat_size ]))
    for row in range(mat_size):
        for col in range(row + 1):
            mat[row, col] = vec[sym_index(row, col)]
            if (not ld_only) and row != col:
                mat[col, row] = mat[row, col]
    return mat


def pack_posdef_matrix(mat):
    return vectorize_matrix(np.linalg.cholesky(mat), ld=True)


def unpack_posdef_matrix(free_vec):
    mat_chol = unvectorize_matrix(free_vec, ld_only=True)
    return mat_chol * mat_chol.T


class PosDefMatrixParam(object):
    def __init__(self, name, size):
        self.name = name
        self.__size = size
        self.__vec_size = size * (size + 1) / 2
        self.__val = np.matrix(np.zeros([size, size]))
    def __str__(self):
        return self.name + ':\n' + str(self.__val)
    def names(self):
        return [ self.name ]
    def set(self, val):
        nrow, ncol = np.shape(val)
        if nrow != self.__size or ncol != self.__size: raise ValueError('Matrix is a different size')
        self.__val = val
    def get(self):
        return self.__val
    def set_free(self, free_val):
        if free_val.size != self.__vec_size: raise ValueError('Free value is the wrong length')
        self.set(unpack_posdef_matrix(free_val))
    def get_free(self):
        return pack_posdef_matrix(self.__val)
    def size(self):
        return self.__size
    def free_size(self):
        return self.__vec_size


class ModelParamsDict(object):
    def __init__(self):
        self.param_dict = {}
        self.__size = 0
        self.__free_size = 0
    def __str__(self):
        return 'ModelParamsList:\n' + '\n'.join([ '\t' + str(param) for param in self.param_dict.values() ])
    def __getitem__(self, key):
        return self.param_dict[key]
    def push_param(self, param):
        self.param_dict[param.name] = param
        # self.__size = self.__size + param.size()
        self.__free_size = self.__free_size + param.free_size()
    def set_free(self, vec):
        if vec.size != self.__free_size: raise ValueError("Wrong size.")
        offset = 0
        for param in self.param_dict.values():
            offset = set_free_offset(param, vec, offset)
            # param.set_free(vec[offset:(offset + param.free_size())])
            # offset = offset + param.free_size()
    def get_free(self):
        vec = np.empty(self.free_size())
        offset = 0
        for param in self.param_dict.values():
            offset = get_free_offset(param, vec, offset)
            # vec[offset:(offset + param.free_size())] = param.get_free()
            # offset = offset + param.free_size()
        return vec
    def names(self):
        return np.concatenate([ param.names() for param in self.param_dict.values()])
    # def size(self):
    #     return self.__size
    def free_size(self):
        return self.__free_size


# Not to be confused with a VectorParam -- this is a vector of abstract
# parameter types.  Note that for the purposes of vectorization it might
# be better to use an object with arrays of attributes rather than an array
# of parameters with singelton attributes.
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
        if free_val.size != self.__free_size: raise ValueError('Wrong size for ParamVector ' + self.name)
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