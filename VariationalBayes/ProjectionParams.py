import math
import copy

import autograd
import autograd.numpy as np
import autograd.scipy as sp

import scipy as osp
from scipy.sparse import coo_matrix, csr_matrix, block_diag

# Get a matrix of vectors spanning the space perpendicular to the rows of x.
# The columns of the returned proj_basis are the spanning vectors.
def get_perpendicular_subspace(x):
    xx = np.matmul(x, x.T)
    proj_parallel = np.matmul(x.T, np.linalg.solve(xx, x))
    proj_perp = np.eye(x.shape[1]) - proj_parallel

    # The eigenvectors of the perpendicular projection span the perpendicular
    # subspace.
    evals, evec = np.linalg.eigh(proj_perp)
    keep_cols = np.abs(evals) > 1e-8

    # There should be as many vectors as the dimension minus the constraints.
    assert(np.sum(keep_cols) == x.shape[1] - x.shape[0])
    proj_basis = evec[:, keep_cols]
    return proj_basis

# This is a vector that lives in a subspace perpendicular to
# the space spanned by the rows of perp_subspace.
class SubspaceVectorParam(object):
    def __init__(self, name='', dim=3, val=None, perp_subspace=None):
        self.name = name
        self.__dim = int(dim)
        if perp_subspace is not None:
            self.__perp_subspace = copy.deepcopy(perp_subspace)
        else:
            # By default, the constraint is having zero mean.
            self.__perp_subspace = np.full((1, self.__dim), 1.0)
        if self.__perp_subspace.shape[1] != self.__dim:
            raise ValueError(
                'The rows of <perp_subspace> must be of length <dim>')
        if self.__perp_subspace.shape[0] >= self.__dim:
            raise ValueError(
                '<perp_subspace> must have strictly fewer rows than <dim>')

        self.__proj_basis = get_perpendicular_subspace(self.__perp_subspace)
        self.__proj_basis_t = self.__proj_basis.T
        self.__constraint_dim = self.__perp_subspace.shape[0]
        self.__free_dim = self.__dim - self.__constraint_dim

        if val is not None:
            self.set(val)
        else:
            self.set(np.full(self.__dim, 0.))

    def __str__(self):
        return self.name + ':\n' + str(self.__val)
    def names(self):
        return [ self.name + '_' + str(k) for k in range(self.vector_size()) ]
    def dictval(self):
        return self.__val.tolist()

    # TODO: you need to make some coherent decisions about how to treat
    # setting parameters to illegal values.  You'll want to allow illegal
    # values at least sometime for index parameter types.
    def set(self, val):
        if val.size != self.dim():
            raise ValueError('Wrong size for vector ' + self.name)
        self.__val = val
    def get(self):
        return self.__val

    def set_free(self, free_val):
        if free_val.size != self.free_size():
            raise ValueError('Wrong free size for vector ' + self.name)
        self.set(np.matmul(self.__proj_basis, free_val))
    def get_free(self):
        return np.matmul(self.__proj_basis_t, self.__val)

    def free_to_vector(self, free_val):
        self.set_free(free_val)
        return self.get_vector()
    def free_to_vector_jac(self, free_val):
        return coo_matrix(self.__proj_basis)
    def free_to_vector_hess(self, free_val):
        return np.array([ coo_matrix((self.__free_dim, self.__free_dim))
                          for vec_ind in range(self.vector_size()) ])


    def set_vector(self, val):
        self.set(val)
    def get_vector(self):
        return self.__val

    def dim(self):
        return self.__dim
    def free_size(self):
        return self.__free_dim
    def vector_size(self):
        return self.__dim
