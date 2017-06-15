from VariationalBayes import Parameters as par
import autograd.numpy as np
import autograd.scipy as asp


# The first index is assumed to index simplicial observations.
def constrain_simplex_matrix(free_mat):
    # The first column is the reference value.
    free_mat_aug = np.hstack([np.full((free_mat.shape[0], 1), 0.), free_mat])
    log_norm = np.expand_dims(sp.misc.logsumexp(free_mat_aug, 1), axis=1)
    return np.exp(free_mat_aug - log_norm)


def unconstrain_simplex_matrix(simplex_mat):
    return np.log(simplex_mat[:, 1:]) - \
           np.expand_dims(np.log(simplex_mat[:, 0]), axis=1)


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
        self.set(par.constrain_simplex_matrix(free_mat))
    def get_free(self):
        return par.unconstrain_simplex_matrix(self.__val).flatten()

    def set_vector(self, vec_val):
        if len(vec_val) != self.vector_size():
            raise ValueError('Wrong vector size for SimplexParam ' + self.name)
        self.set(np.reshape(vec_val, self.__shape))
    def get_vector(self):
        return self.__val.flatten()

    def shape(self):
        return self.__shape
    def free_shape(self):
        return self.__free_shape
    def free_size(self):
        return np.product(self.__free_shape)
    def vector_size(self):
        return np.product(self.__shape)
