from Parameters import ScalarParam, VectorParam, PosDefMatrixParam
from Parameters import set_free_offset, get_free_offset
import numpy as np

class MVNParam(object):
    def __init__(self, name, dim):
        self.name = name
        self.__dim = dim
        self.mean = VectorParam(name + '_mean', dim)
        self.cov = PosDefMatrixParam(name + '_cov', dim)
        self.__free_size = self.mean.free_size() + self.cov.free_size()
    def __str__(self):
        return self.name + ':\n' + str(self.mean) + '\n' + str(self.cov)
    def names(self):
        return self.mean.names() + self.cov.names()
    def e(self):
        return self.mean.get()
    def e_outer(self):
        mean = self.mean.get()
        return np.outer(mean, mean) + self.cov.get()
    def set_free(self, free_val):
        if free_val.size != self.__free_size: \
            raise ValueError('Wrong size for MVNParam ' + self.name)
        offset = 0
        offset = set_free_offset(self.mean, free_val, offset)
        offset = set_free_offset(self.cov, free_val, offset)
    def get_free(self):
        vec = np.empty(self.__free_size)
        offset = 0
        offset = get_free_offset(self.mean, vec, offset)
        offset = get_free_offset(self.cov, vec, offset)
        return vec
    def free_size(self):
        return self.__free_size
    def dim(self):
        return self.__dim
    

class UVNParam(object):
    def __init__(self, name, min_var=0.0):
        self.name = name
        self.mean = ScalarParam(name + '_mean')
        self.var = ScalarParam(name + '_var', lb=min_var)
        self.__free_size = self.mean.free_size() + self.var.free_size()
    def __str__(self):
        return self.name + ':\n' + str(self.mean) + '\n' + str(self.var)
    def names(self):
        return self.mean.names() + self.var.names()
    def e(self):
        return self.mean.get()
    def e_outer(self):
        mean = self.mean.get() ** 2 + self.var.get()
    def set_free(self, free_val):
        if free_val.size != self.__free_size: \
            raise ValueError('Wrong size for UVNParam ' + self.name)
        offset = 0
        offset = set_free_offset(self.mean, free_val, offset)
        offset = set_free_offset(self.var, free_val, offset)
    def get_free(self):
        vec = np.empty(self.__free_size)
        offset = 0
        offset = get_free_offset(self.mean, vec, offset)
        offset = get_free_offset(self.var, vec, offset)
        return vec
    def free_size(self):
        return self.__free_size
    