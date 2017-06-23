from VariationalBayes import Parameters as par
import autograd.numpy as np
import autograd.scipy as asp

from VariationalBayes.Parameters import \
    free_to_vector_jac_offset, free_to_vector_hess_offset

from scipy.sparse import block_diag

class GammaParam(object):
    def __init__(self, name='', min_shape=0.0, min_rate=0.0):
        self.name = name
        self.shape = par.ScalarParam(name + '_shape', lb=min_shape)
        self.rate = par.ScalarParam(name + '_rate', lb=min_rate)
        self.__free_size = self.shape.free_size() + self.rate.free_size()
        self.__vector_size = self.shape.vector_size() + self.rate.vector_size()
    def __str__(self):
        return self.name + ':\n' + str(self.shape) + '\n' + str(self.rate)
    def names(self):
        return self.shape.names() + self.rate.names()
    def dictval(self):
        return { 'shape': self.shape.dictval(), 'rate': self.rate.dictval() }
    def e(self):
        return self.shape.get() / self.rate.get()
    def e_log(self):
        return asp.special.digamma(self.shape.get()) - np.log(self.rate.get())

    def set_free(self, free_val):
        if free_val.size != self.__free_size: \
            raise ValueError('Wrong size for GammaParam ' + self.name)
        offset = 0
        offset = par.set_free_offset(self.shape, free_val, offset)
        offset = par.set_free_offset(self.rate, free_val, offset)
    def get_free(self):
        return np.hstack([ self.shape.get_free(), self.rate.get_free() ])

    def free_to_vector(self, free_val):
        self.set_free(free_val)
        return self.get_vector()
    def free_to_vector_jac(self, free_val):
        free_offset = 0
        vec_offset = 0
        free_offset, vec_offset, mean_jac = free_to_vector_jac_offset(
            self.shape, free_val, free_offset, vec_offset)
        free_offset, vec_offset, info_jac = free_to_vector_jac_offset(
            self.rate, free_val, free_offset, vec_offset)
        return block_diag((mean_jac, info_jac))
    def free_to_vector_hess(self, free_val):
        free_offset = 0
        full_shape = (self.free_size(), self.free_size())
        hessians = []
        free_offset = free_to_vector_hess_offset(
            self.shape, free_val, hessians, free_offset, full_shape)
        free_offset = free_to_vector_hess_offset(
            self.rate, free_val, hessians, free_offset, full_shape)
        return np.array(hessians)

    def set_vector(self, vec):
        if vec.size != self.__vector_size: raise ValueError("Wrong size.")
        offset = 0
        offset = par.set_vector_offset(self.shape, vec, offset)
        offset = par.set_vector_offset(self.rate, vec, offset)
    def get_vector(self):
        return np.hstack([ self.shape.get_vector(), self.rate.get_vector() ])

    def free_size(self):
        return self.__free_size
    def vector_size(self):
        return self.__vector_size
