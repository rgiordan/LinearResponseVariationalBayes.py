import LinearResponseVariationalBayes as vb

import autograd
import autograd.numpy as np
import autograd.scipy as sp

import scipy as osp
from scipy.sparse import coo_matrix

# The first index is assumed to index simplicial observations.
def constrain_simplex_matrix(free_mat):
    # The first column is the reference value.
    free_mat_aug = np.hstack(
        [np.full((free_mat.shape[0], 1), 0.), free_mat])
    log_norm = np.expand_dims(sp.misc.logsumexp(free_mat_aug, 1), axis=1)
    return np.exp(free_mat_aug - log_norm)


def unconstrain_simplex_matrix(simplex_mat):
    return np.log(simplex_mat[:, 1:]) - \
           np.expand_dims(np.log(simplex_mat[:, 0]), axis=1)


def constrain_simplex_vector(free_vec):
    return constrain_simplex_matrix(np.expand_dims(free_vec, 0)).flatten()

# TODO: some compuatation is shared between the Jacobian and Hessian.

# The Jacobian of the constraint is most easily calculated as a function
# of the constrained moments.
def constrain_grad_from_moment(z):
    z_last = z[1:]
    z_jac = -1 * np.outer(z, z_last)
    for k in range(1, len(z)):
        z_jac[k, k - 1] += z[k]
    return z_jac

# The Hessian of the constraint is most easily calculated as a function
# of the constrained moments.
def constrain_hess_from_moment(z):
    # See the notes simplex_derivatives.lyx for a derivation.
    z_last = z[1:]
    z_outer = np.expand_dims(2 * np.outer(z_last, z_last), axis=0)
    z_hess = np.tile(z_outer, (len(z), 1, 1))
    # The first index is different.
    for k in range(1, len(z)):
        z_hess[0, k - 1, k - 1] += -z[k]

    z_hess[0] *= z[0]

    for k in range(1, len(z)):
        z_hess[k, k - 1, :] += -z_last
        z_hess[k, :, k - 1] += -z_last
        for j in range(1, len(z)):
            if j == k:
                z_hess[k, j - 1, j - 1] += 1 - z[j]
            else:
                z_hess[k, j - 1, j - 1] += -z[j]
        z_hess[k, :, :] *= z[k]

    return z_hess


class SimplexParam(object):
    def __init__(self, name='', shape=(1, 2), val=None):
        self.name = name
        self.__shape = shape
        self.__free_shape = (shape[0], shape[1] - 1)
        if val is not None:
            self.set(val)
        else:
            self.set(np.full(shape, 1. / shape[1]))

    def __str__(self):
        return self.name + ': ' + str(self.__val)
    def names(self):
        return [ self.name ]
    def dictval(self):
        return self.__val.tolist()

    def set(self, val):
        if val.shape != self.__shape:
            raise ValueError('Wrong shape for SimplexParam ' + self.name)
        self.__val = val
    def get(self):
        return self.__val

    def set_free(self, free_val):
        if len(free_val) != self.free_size():
            raise ValueError('Wrong free size for SimplexParam ' + self.name)
        free_mat = np.reshape(free_val, self.__free_shape)
        self.set(constrain_simplex_matrix(free_mat))
    def get_free(self):
        return unconstrain_simplex_matrix(self.__val).flatten()

    def free_to_vector(self, free_val):
        self.set_free(free_val)
        return self.get_vector()

    # The rows are the vector and the columns are the free values.
    def free_to_vector_jac(self, free_val):
        jac_rows = []
        jac_cols = []
        grads = []
        free_cols = range(self.free_shape()[1])
        vec_cols = range(self.shape()[1])
        for row in range(self.shape()[0]):
            # Each of the output depends only on one row of the input.
            free_inds = np.ravel_multi_index([[row], free_cols], self.free_shape())
            vec_inds = np.ravel_multi_index([[row], vec_cols], self.shape())
            #row_jac = constrain_grad(free_val[free_inds])
            z = constrain_simplex_vector(free_val[free_inds])
            row_jac = constrain_grad_from_moment(z)
            for vec_col in vec_cols:
                for free_col in free_cols:
                    jac_rows.append(vec_inds[vec_col])
                    jac_cols.append(free_inds[free_col])
                    grads.append(row_jac[vec_col,free_col])

        return coo_matrix((grads, (jac_rows, jac_cols)),
                          (self.vector_size(), self.free_size()))


    def free_to_vector_hess(self, free_val):
        free_cols = range(self.free_shape()[1])
        vec_cols = range(self.shape()[1])
        hesses = []
        hess_shape = (self.free_size(), self.free_size())

        for row in range(self.shape()[0]):
            # Each of the output depends only on one row of the input.
            free_inds = np.ravel_multi_index(
                [[row], free_cols], self.free_shape())
            vec_inds = self.get_vector_indices(row)
            z = constrain_simplex_vector(free_val[free_inds])
            row_hess = constrain_hess_from_moment(z)
            for vec_col in vec_cols:
                vec_ind = vec_inds[vec_col]
                hess_rows = []
                hess_cols = []
                hess_vals = []
                for free_col1 in free_cols:
                    for free_col2 in free_cols:
                        hess_rows.append(free_inds[free_col1])
                        hess_cols.append(free_inds[free_col2])
                        hess_vals.append(row_hess[vec_col, free_col1, free_col2])
                hesses.append(
                    coo_matrix((hess_vals, (hess_rows, hess_cols)), hess_shape))

        return hesses

    def set_vector(self, vec_val):
        if len(vec_val) != self.vector_size():
            raise ValueError('Wrong vector size for SimplexParam ' + self.name)
        self.set(np.reshape(vec_val, self.__shape))
    def get_vector(self):
        return self.__val.flatten()
    def get_vector_indices(self, row):
        # Get the indices in the vector representation for a particular row.
        return np.ravel_multi_index(
            [[row], range(self.shape()[1])], self.shape())

    def shape(self):
        return self.__shape
    def free_shape(self):
        return self.__free_shape
    def free_size(self):
        return np.product(self.__free_shape)
    def vector_size(self):
        return np.product(self.__shape)
