import math
import copy
import numbers

import autograd
import autograd.numpy as np
import autograd.scipy as sp
from autograd.core import primitive, defvjp, defjvp

import itertools

import scipy as osp
from scipy.sparse import coo_matrix

# Uses 0-indexing. (row, col) = (k1, k2)
def SymIndex(k1, k2):
    def LDInd(k1, k2):
        return int(k2 + k1 * (k1 + 1) / 2)

    if k2 <= k1:
        return LDInd(k1, k2)
    else:
        return LDInd(k2, k1)


# Map a matrix
#
# [ x11 x12 ... x1n ]
# [ x21 x22     x2n ]
# [...              ]
# [ xn1 ...     xnn ]
#
# to the vector
#
# [ x11, x21, x22, x31, ..., xnn ].
#
# The entries above the diagonal are ignored.
def vectorize_ld_matrix(mat):
    nrow, ncol = np.shape(mat)
    if nrow != ncol: raise ValueError('mat must be square')
    return mat[np.tril_indices(nrow)]


# Map a vector
#
# [ v1, v2, ..., vn ]
#
# to the symmetric matrix
#
# [ v1 ...          ]
# [ v2 v3 ...       ]
# [ v4 v5 v6 ...    ]
# [ ...             ]
#
# where the values above the diagonal are determined by symmetry.
#
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

def unvectorize_ld_matrix_vjp(g):
    assert g.shape[0] == g.shape[1]
    return vectorize_ld_matrix(g)

defvjp(unvectorize_ld_matrix,
       lambda ans, vec: lambda g: unvectorize_ld_matrix_vjp(g))

def unvectorize_ld_matrix_jvp(g):
    return unvectorize_ld_matrix(g)

defjvp(unvectorize_ld_matrix, lambda g, ans, x : unvectorize_ld_matrix_jvp(g))


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
        return coo_matrix(self.free_to_vector_jac_dense(free_val))
    def free_to_vector_hess(self, free_val):
        hess_dense = self.free_to_vector_hess_dense(free_val)
        return [ coo_matrix(hess_dense[ind, :, :])
                 for ind in range(hess_dense.shape[0]) ]

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
        # vec_size is the size of a single matrix in vector form.
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
        return np.hstack([ pack_posdef_matrix(
            self.__val[obs, :, :], diag_lb=self.__diag_lb) \
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

        return coo_matrix(
            (grads, (jac_rows, jac_cols)),
            (self.vector_size(), self.free_size()))

    def free_to_vector_hess(self, free_val):
        hessians = []
        vec_rows = range(self.__vec_size)
        packed_shape = (self.__length, self.__vec_size)
        hess_shape = (self.__length * self.__vec_size,
                      self.__length * self.__vec_size)
        for row in range(self.__length):
            vec_inds = np.ravel_multi_index([[row], vec_rows], packed_shape)
            row_hess = pos_def_matrix_free_to_vector_hess(free_val[vec_inds])
            for vec_row in vec_rows:
                hess_rows = []
                hess_cols = []
                hess_vals = []
                for free_row1 in vec_rows:
                    for free_row2 in vec_rows:
                        hess_rows.append(vec_inds[free_row1])
                        hess_cols.append(vec_inds[free_row2])
                        hess_vals.append(row_hess[vec_row, free_row1, free_row2])

                # It is important that this traverse vec_inds in order because
                # we simply append the hessians to the end.
                hessians.append(
                    coo_matrix((hess_vals, (hess_rows, hess_cols)), hess_shape))

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


# In keeping with the vector version, the last two indices are indices
# of the matrix.
class PosDefMatrixParamArray(object):
    def __init__(self, name='', array_shape=(1), matrix_size=2, diag_lb=0.0, val=None):
        self.name = name
        self.__matrix_size = int(matrix_size)
        self.__matrix_shape = np.array([ int(matrix_size), int(matrix_size) ])
        self.__array_shape = array_shape
        self.__array_ranges = [ range(0, t) for t in self.__array_shape ]
        self.__array_length = np.prod(self.__array_shape)
        self.__shape = np.append(self.__array_shape, self.__matrix_shape)
        # __vec_size is the size of a single matrix in vector form.
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
            print(val.shape)
            print(self.__shape)
            raise ValueError('Array is the wrong size')
        self.__val = val

    def get(self):
        return self.__val

    # Get the elements in a free vector correpsonding to element obs of the
    # array, where obs is a tuple indexing into the array of shape
    # self.__array_shape.
    def stacked_obs_slice(self, obs):
        assert len(obs) == len(self.__array_shape)
        linear_obs = np.ravel_multi_index(obs, self.__array_shape) * \
                     self.__vec_size
        return slice(linear_obs, linear_obs + self.__vec_size)

    def set_free(self, free_val):
        if free_val.size != self.free_size():
            raise ValueError('Free value is the wrong length')
        new_val = \
            np.reshape(np.array([ unpack_posdef_matrix(
                free_val[self.stacked_obs_slice(obs)], diag_lb=self.__diag_lb) \
              for obs in itertools.product(*self.__array_ranges) ]),
              self.__shape)
        self.__val = new_val

    # Return an array of the function mat_func applied to each matrix in
    # the array.
    def apply_matrix_function(self, mat_func):
        mat_func_array = np.array([
                mat_func(self.__val[obs]) \
                    for obs in itertools.product(*self.__array_ranges) ])

        # Assume that each output is the same shape.
        new_shape = self.__array_shape + mat_func_array[0].shape
        return np.reshape(mat_func_array, new_shape)

    def get_free(self):
        # I don't know why this doesn't work, but it gives an autograd error
        # about assigning in arrays:
        #return np.hstack(self.apply_matrix_function(pack_posdef_matrix))
        return np.hstack(np.array([ \
            pack_posdef_matrix(self.__val[obs], diag_lb=self.__diag_lb) \
                    for obs in itertools.product(*self.__array_ranges)]))

    def set_vector(self, vec_val):
        if len(vec_val) != self.vector_size():
            raise ValueError('Vector value is the wrong length')
        self.__val = \
            np.reshape(np.array([ unvectorize_symmetric_matrix(
                    vec_val[self.stacked_obs_slice(obs)]) \
                for obs in itertools.product(*self.__array_ranges) ]),
              self.__shape)

    def get_vector(self):
        vec_val = np.hstack([ vectorize_ld_matrix(
            self.__val[obs]) \
            for obs in itertools.product(*self.__array_ranges) ])
        #print('get vector ', vec_val)
        return vec_val

    def free_to_vector(self, free_val):
        self.set_free(free_val)
        return self.get_vector()

    def free_to_vector_jac(self, free_val):
        jac_rows = []
        jac_cols = []
        grads = []
        vec_rows = range(self.__vec_size)
        # This is the shape of an array of the same size as self.__array_size
        # of packed vectors.  We'll use it to pick out
        packed_shape = self.__array_shape + (self.__vec_size,)

        for obs in itertools.product(*self.__array_ranges):
            # This seems like a convoluted expression, but it is what
            # is required by ravel_multi_index.
            array_inds = tuple([ [t] for t in obs]) + (vec_rows,)
            vec_inds = np.ravel_multi_index(array_inds, packed_shape)
            row_jac = pos_def_matrix_free_to_vector_jac(free_val[vec_inds])
            for vec_row in vec_rows:
                for free_row in vec_rows:
                    jac_rows.append(vec_inds[vec_row])
                    jac_cols.append(vec_inds[free_row])
                    grads.append(row_jac[vec_row, free_row])

        return coo_matrix(
            (grads, (jac_rows, jac_cols)),
            (self.vector_size(), self.free_size()))

    def free_to_vector_hess(self, free_val):
        hessians = []
        vec_rows = range(self.__vec_size)
        packed_shape = self.__array_shape + (self.__vec_size,)
        hess_shape = (self.__array_length * self.__vec_size,
                      self.__array_length * self.__vec_size)
        for obs in itertools.product(*self.__array_ranges):
        #for row in range(self.__length):
            #vec_inds = np.ravel_multi_index([[row], vec_rows], packed_shape)
            # vec_inds = self.stacked_obs_slice(obs)
            array_inds = tuple([ [t] for t in obs]) + (vec_rows,)
            vec_inds = np.ravel_multi_index(array_inds, packed_shape)
            row_hess = pos_def_matrix_free_to_vector_hess(free_val[vec_inds])
            for vec_row in vec_rows:
                hess_rows = []
                hess_cols = []
                hess_vals = []
                for free_row1 in vec_rows:
                    for free_row2 in vec_rows:
                        hess_rows.append(vec_inds[free_row1])
                        hess_cols.append(vec_inds[free_row2])
                        hess_vals.append(row_hess[vec_row, free_row1, free_row2])

                # It is important that this traverse vec_inds in order because
                # we simply append the hessians to the end.
                hessians.append(
                    coo_matrix((hess_vals, (hess_rows, hess_cols)), hess_shape))

        return hessians

    def length(self):
        return self.__length
    def matrix_size(self):
        return self.__matrix_size
    def free_size(self):
        return self.__vec_size * self.__array_length
    def vector_size(self):
        return self.__vec_size * self.__array_length
    def get_array_ranges(self):
        return self.__array_ranges
