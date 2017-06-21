import math
import copy
import numbers

import autograd
import autograd.numpy as np
import autograd.scipy as sp
from autograd.core import primitive

import scipy as osp
from scipy.sparse import csr_matrix

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

# The first index is assumed to index simplicial observations.
def constrain_simplex_matrix(free_mat):
    # The first column is the reference value.
    free_mat_aug = np.hstack([np.full((free_mat.shape[0], 1), 0.), free_mat])
    log_norm = np.expand_dims(sp.misc.logsumexp(free_mat_aug, 1), axis=1)
    return np.exp(free_mat_aug - log_norm)


def unconstrain_simplex_matrix(simplex_mat):
    return np.log(simplex_mat[:, 1:]) - \
           np.expand_dims(np.log(simplex_mat[:, 0]), axis=1)


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
        # Asserting that you are getting something of length one doesn't
        # seem trivial in python.
        # if not isinstance(val, numbers.Number):
        #     if len(val) != 1:
        #         raise ValueError('val must be a number or length-one array.')
        if val < self.__lb:
            error_msg = 'val is less than the lower bound: ' + \
                str(val) + ' <= ' + str(self.__lb)
            raise ValueError(error_msg)
        if val > self.__ub:
            raise ValueError('val is greater than the upper bound.')
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
        return csr_matrix(self.free_to_vector_jac_dense(free_val))
    def free_to_vector_hess(self, free_val):
        return csr_matrix(self.free_to_vector_hess_dense(free_val))

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
        if any(val < self.__lb):
            raise ValueError('Value beneath lower bound.')
        if any(val > self.__ub):
            raise ValueError('Value above upper bound.')
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
        return csr_matrix((grads,
                          (rows_indices, rows_indices)),
                          (self.vector_size(), self.free_size()))
    def free_to_vector_hess(self, free_val):
        def get_ind_hess(vec_ind):
            hess = constrain_scalar_hess(free_val[vec_ind], self.__lb, self.__ub)
            return csr_matrix(([ hess ],
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
            raise ValueError('Wrong size for array ' + self.name)
        if (val < self.__lb).any():
            raise ValueError('Value beneath lower bound.')
        if (val > self.__ub).any():
            raise ValueError('Value above upper bound.')
        self.__val = val
    def get(self):
        return self.__val

    def set_free(self, free_val):
        if free_val.size != self.free_size():
            raise ValueError('Wrong length for array ' + self.name)
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
        return csr_matrix((grads,
                          (rows_indices, rows_indices)),
                          (self.vector_size(), self.free_size()))
    def free_to_vector_hess(self, free_val):
        def get_ind_hess(vec_ind):
            hess = constrain_scalar_hess(free_val[vec_ind], self.__lb, self.__ub)
            return csr_matrix(([ hess ],
                               ([vec_ind], [vec_ind])),
                               (self.free_size(), self.vector_size()))
        return np.array([ get_ind_hess(vec_ind)
                          for vec_ind in range(self.vector_size()) ])

    def set_vector(self, val):
        if val.size != self.vector_size():
            raise ValueError('Wrong length for array ' + self.name)
        self.set(val.reshape(self.__shape))
    def get_vector(self):
        return self.__val.flatten()

    def shape(self):
        return self.__shape
    def free_size(self):
        return int(np.product(self.__shape))
    def vector_size(self):
        return int(np.product(self.__shape))


# Uses 0-indexing. (row, col) = (k1, k2)
def SymIndex(k1, k2):
    def LDInd(k1, k2):
        return int(k2 + k1 * (k1 + 1) / 2)

    if k2 <= k1:
        return LDInd(k1, k2)
    else:
        return LDInd(k2, k1)


def vectorize_ld_matrix(mat):
    nrow, ncol = np.shape(mat)
    if nrow != ncol: raise ValueError('mat must be square')
    return mat[np.tril_indices(nrow)]


# Because we cannot use autograd with array assignment, just define the
# vector jacobian product directly.
@primitive
def unvectorize_ld_matrix(vec):
    mat_size = int(0.5 * (math.sqrt(1 + 8 * vec.size) - 1))
    if mat_size * (mat_size + 1) / 2 != vec.size: \
        raise ValueError('Vector is an impossible size')
    mat = np.zeros((mat_size, mat_size))
    for k1 in range(mat_size):
        for k2 in range(k1 + 1):
            mat[k1, k2] = vec[SymIndex(k1, k2)]
    return mat


def unvectorize_ld_matrix_vjp(g, ans, vs, gvs, vec):
    assert g.shape[0] == g.shape[1]
    return vectorize_ld_matrix(g)

unvectorize_ld_matrix.defvjp(unvectorize_ld_matrix_vjp)

def exp_matrix_diagonal(mat):
    assert mat.shape[0] == mat.shape[1]
    # make_diagonal() is only defined in the autograd version of numpy
    mat_exp_diag = np.make_diagonal(
        np.exp(np.diag(mat)), offset=0, axis1=-1, axis2=-2)
    mat_diag = np.make_diagonal(np.diag(mat), offset=0, axis1=-1, axis2=-2)
    return mat_exp_diag + mat - mat_diag


def log_matrix_diagonal(mat):
    assert mat.shape[0] == mat.shape[1]
    # make_diagonal() is only defined in the autograd version of numpy
    mat_log_diag = np.make_diagonal(
        np.log(np.diag(mat)), offset=0, axis1=-1, axis2=-2)
    mat_diag = np.make_diagonal(np.diag(mat), offset=0, axis1=-1, axis2=-2)
    return mat_log_diag + mat - mat_diag


def pack_posdef_matrix(mat, diag_lb=0.0):
    k = mat.shape[0]
    mat_lb = mat - np.make_diagonal(
        np.full(k, diag_lb), offset=0, axis1=-1, axis2=-2)
    return vectorize_ld_matrix(log_matrix_diagonal(np.linalg.cholesky(mat_lb)))


def unpack_posdef_matrix(free_vec, diag_lb=0.0):
    mat_chol = exp_matrix_diagonal(unvectorize_ld_matrix(free_vec))
    mat = np.matmul(mat_chol, mat_chol.T)
    k = mat.shape[0]
    return mat + np.make_diagonal(np.full(k, diag_lb), offset=0, axis1=-1, axis2=-2)


# Convert a vector containing the lower diagonal portion of a symmetric
# matrix into the full symmetric matrix.
def unvectorize_symmetric_matrix(vec_val):
    ld_mat = unvectorize_ld_matrix(vec_val)
    mat_val = ld_mat + ld_mat.transpose()
    # We have double counted the diagonal.  For some reason the autograd
    # diagonal functions require axis1=-1 and axis2=-2
    mat_val = mat_val - \
        np.make_diagonal(np.diagonal(ld_mat, axis1=-1, axis2=-2),
                         axis1=-1, axis2=-2)
    return mat_val


def pos_def_matrix_free_to_vector(free_val, diag_lb=0.0):
    mat_val = unpack_posdef_matrix(free_val, diag_lb=diag_lb)
    return vectorize_ld_matrix(mat_val)

pos_def_matrix_free_to_vector_jac = \
    autograd.jacobian(pos_def_matrix_free_to_vector)
pos_def_matrix_free_to_vector_hess = \
    autograd.hessian(pos_def_matrix_free_to_vector)

class PosDefMatrixParam(object):
    def __init__(self, name='', size=2, diag_lb=0.0, val=None):
        self.name = name
        self.__size = int(size)
        self.__vec_size = int(size * (size + 1) / 2)
        self.__diag_lb = diag_lb
        assert diag_lb >= 0
        if val is None:
            self.__val = np.diag(np.full(self.__size, diag_lb + 1.0))
        else:
            self.set(val)

        # These will be dense, so just use autograd directly.
        self.free_to_vector_jac_dense = autograd.jacobian(self.free_to_vector)
        self.free_to_vector_hess_dense = autograd.hessian(self.free_to_vector)

    def __str__(self):
        return self.name + ':\n' + str(self.__val)
    def names(self):
        return [ self.name ]
    def dictval(self):
        return self.__val.tolist()

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
        self.set(unpack_posdef_matrix(free_val, diag_lb=self.__diag_lb))
    def get_free(self):
        return pack_posdef_matrix(self.__val, diag_lb=self.__diag_lb)
    def free_to_vector(self, free_val):
        self.set_free(free_val)
        return self.get_vector()
    def free_to_vector_jac(self, free_val):
        return csr_matrix(self.free_to_vector_jac_dense(free_val))
    def free_to_vector_hess(self, free_val):
        return csr_matrix(self.free_to_vector_hess_dense(free_val))

    def set_vector(self, vec_val):
        if vec_val.size != self.__vec_size:
            raise ValueError('Vector value is the wrong length')
        self.set(unvectorize_symmetric_matrix(vec_val))
    def get_vector(self):
        return vectorize_ld_matrix(self.__val)

    def size(self):
        return self.__size
    def free_size(self):
        return self.__vec_size
    def vector_size(self):
        return self.__vec_size


class PosDefMatrixParamVector(object):
    def __init__(self, name='', length=1, matrix_size=2, diag_lb=0.0, val=None):
        self.name = name
        self.__matrix_size = int(matrix_size)
        self.__matrix_shape = np.array([ int(matrix_size), int(matrix_size) ])
        self.__length = int(length)
        self.__shape = np.append(self.__length, self.__matrix_shape)
        self.__vec_size = int(matrix_size * (matrix_size + 1) / 2)
        self.__diag_lb = diag_lb
        assert diag_lb >= 0
        if val is None:
            default_val = np.diag(np.full(self.__matrix_size, diag_lb + 1.0))
            self.__val = np.broadcast_to(default_val, self.__shape)
        else:
            self.set(val)

    def __str__(self):
        return self.name + ':\n' + str(self.__val)
    def names(self):
        return [ self.name ]
    def dictval(self):
        return self.__val.tolist()

    def set(self, val):
        if (val.shape != self.__shape).all():
            raise ValueError('Array is the wrong size')
        self.__val = val
    def get(self):
        return self.__val

    def free_obs_slice(self, obs):
        assert obs < self.__length
        return slice(self.__vec_size * obs, self.__vec_size * (obs + 1))

    def set_free(self, free_val):
        if free_val.size != self.free_size():
            raise ValueError('Free value is the wrong length')
        self.__val = \
            np.array([ unpack_posdef_matrix(
                free_val[self.free_obs_slice(obs)], diag_lb=self.__diag_lb) \
              for obs in range(self.__length) ])
    def get_free(self):
        return np.hstack([ \
            pack_posdef_matrix(self.__val[obs, :, :], diag_lb=self.__diag_lb) \
                          for obs in range(self.__length)])
    def free_to_vector(self, free_val):
        self.set_free(free_val)
        return self.get_vector()
    def free_to_vector_jac(self, free_val):
        jac_rows = []
        jac_cols = []
        grads = []
        vec_rows = range(self.__vec_size)
        packed_shape = (self.__length, self.__vec_size)

        for row in range(self.__length):
            vec_inds = np.ravel_multi_index([[row], vec_rows], packed_shape)
            row_jac = pos_def_matrix_free_to_vector_jac(free_val[vec_inds])
            for vec_row in vec_rows:
                for free_row in vec_rows:
                    jac_rows.append(vec_inds[vec_row])
                    jac_cols.append(vec_inds[free_row])
                    grads.append(row_jac[vec_row, free_row])

        return csr_matrix(
            (grads, (jac_rows, jac_cols)),
            (self.vector_size(), self.free_size()))

    def free_to_vector_hess(self, free_val):
        hessians = []
        vec_rows = range(self.__vec_size)
        packed_shape = (self.__length, self.__vec_size)
        hess_shape = (self.__vec_size, self.__vec_size)
        for row in range(self.__length):
            vec_inds = np.ravel_multi_index([[row], vec_cols], packed_shape)
            row_hess = pos_def_matrix_free_to_vector_hess(free_val[vec_inds])
            hess_rows = []
            hess_cols = []
            hess_vals = []
            for vec_row in vec_rows:
                for free_row1 in vec_rows:
                    for free_row2 in vec_rows:
                        hess_rows.append(vec_inds[free_row1])
                        hess_cols.append(vec_inds[free_row2])
                        hess_vals.append(row_hess[vec_inds[vec_row],
                                                  free_col1, free_col2])

                # It is important that this traverse vec_inds in order because
                # we simply append the hessians to the end.
                hessians.append(
                    csr_matrix((hess_vals, (hess_rows, hess_cols)), hess_shape))

        return hessians

    def set_vector(self, vec_val):
        if len(vec_val) != self.vector_size():
            raise ValueError('Vector value is the wrong length')
        self.__val = \
            np.array([
              unvectorize_symmetric_matrix(vec_val[self.free_obs_slice(obs)]) \
              for obs in range(self.__length) ])

    def get_vector(self):
        return np.hstack([ vectorize_ld_matrix(self.__val[obs, :, :]) \
                           for obs in range(self.__length) ])

    def length(self):
        return self.__length
    def matrix_size(self):
        return self.__matrix_size
    def free_size(self):
        return self.__vec_size * self.__length
    def vector_size(self):
        return self.__vec_size * self.__length


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
