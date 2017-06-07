from Parameters import ScalarParam
from Parameters import set_free_offset, set_vector_offset
import autograd.numpy as np
import autograd.scipy as asp

class GammaParam(object):
    def __init__(self, name='', min_shape=0.0, min_rate=0.0):
        self.name = name
        self.shape = ScalarParam(name + '_shape', lb=min_shape)
        self.rate = ScalarParam(name + '_rate', lb=min_rate)
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
        offset = set_free_offset(self.shape, free_val, offset)
        offset = set_free_offset(self.rate, free_val, offset)
    def get_free(self):
        return np.hstack([ self.shape.get_free(), self.rate.get_free() ])

    def set_vector(self, vec):
        if vec.size != self.__vector_size: raise ValueError("Wrong size.")
        offset = 0
        offset = set_vector_offset(self.shape, vec, offset)
        offset = set_vector_offset(self.rate, vec, offset)
    def get_vector(self):
        return np.hstack([ self.shape.get_vector(), self.rate.get_vector() ])

    def free_size(self):
        return self.__free_size
    def vector_size(self):
        return self.__vector_size
