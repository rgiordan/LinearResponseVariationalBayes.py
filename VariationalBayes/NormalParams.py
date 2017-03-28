from Parameters import ScalarParam, VectorParam, PosDefMatrixParam
from Parameters import set_free_offset, set_vector_offset
import autograd.numpy as np

class MVNParam(object):
    def __init__(self, name, dim, min_info=0.0):
        self.name = name
        self.__dim = dim
        self.mean = VectorParam(name + '_mean', dim)
        self.info = PosDefMatrixParam(name + '_info', dim, diag_lb=min_info)
        self.__free_size = self.mean.free_size() + self.info.free_size()
        self.__vector_size = self.mean.vector_size() + self.info.vector_size()
    def __str__(self):
        return self.name + ':\n' + str(self.mean) + '\n' + str(self.info)
    def names(self):
        return self.mean.names() + self.info.names()
    def dictval(self):
        return { 'mean': self.mean.dictval(), 'info': self.info.dictval() }
    def e(self):
        return self.mean.get()
    def e_outer(self):
        mean = self.mean.get()
        info = self.info.get()
        return np.outer(mean, mean) + np.linalg.inv(info)

    def set_free(self, free_val):
        if free_val.size != self.__free_size: \
            raise ValueError('Wrong size for MVNParam ' + self.name)
        offset = 0
        offset = set_free_offset(self.mean, free_val, offset)
        offset = set_free_offset(self.info, free_val, offset)
    def get_free(self):
        return np.hstack([ self.mean.get_free(), self.info.get_free() ])

    def set_vector(self, vec):
        if vec.size != self.__vector_size: raise ValueError("Wrong size.")
        offset = 0
        offset = set_vector_offset(self.mean, vec, offset)
        offset = set_vector_offset(self.info, vec, offset)
    def get_vector(self):
        return np.hstack([ self.mean.get_vector(), self.info.get_vector() ])

    def free_size(self):
        return self.__free_size
    def vector_size(self):
        return self.__vector_size
    def dim(self):
        return self.__dim


class UVNParam(object):
    def __init__(self, name, min_info=0.0):
        self.name = name
        self.mean = ScalarParam(name + '_mean')
        self.info = ScalarParam(name + '_info', lb=min_info)
        self.__free_size = self.mean.free_size() + self.info.free_size()
        self.__vector_size = self.mean.vector_size() + self.info.vector_size()
    def __str__(self):
        return self.name + ':\n' + str(self.mean) + '\n' + str(self.info)
    def names(self):
        return self.mean.names() + self.info.names()
    def dictval(self):
        return { 'mean': self.mean.dictval(), 'info': self.info.dictval() }
    def e(self):
        return self.mean.get()
    def e_outer(self):
        return self.mean.get() ** 2 + 1 / self.info.get()

    def set_free(self, free_val):
        if free_val.size != self.__free_size: \
            raise ValueError('Wrong size for UVNParam ' + self.name)
        offset = 0
        offset = set_free_offset(self.mean, free_val, offset)
        offset = set_free_offset(self.info, free_val, offset)
    def get_free(self):
        return np.hstack([ self.mean.get_free(), self.info.get_free() ])

    def set_vector(self, vec):
        if vec.size != self.__vector_size: raise ValueError("Wrong size.")
        offset = 0
        offset = set_vector_offset(self.mean, vec, offset)
        offset = set_vector_offset(self.info, vec, offset)
    def get_vector(self):
        return np.hstack([ self.mean.get_vector(), self.info.get_vector() ])

    def free_size(self):
        return self.__free_size
    def vector_size(self):
        return self.__vector_size


class UVNParamVector(object):
    def __init__(self, name, length, min_info=0.0):
        self.name = name
        self.mean = VectorParam(name + '_mean', length)
        self.info = VectorParam(name + '_info', length, lb=min_info)
        self.__free_size = self.mean.free_size() + self.info.free_size()
        self.__vector_size = self.mean.vector_size() + self.info.vector_size()
    def __str__(self):
        return self.name + ':\n' + str(self.mean) + '\n' + str(self.info)
    def names(self):
        return self.mean.names() + self.info.names()
    def dictval(self):
        return { 'mean': self.mean.dictval(), 'info': self.info.dictval() }
    def e(self):
        return self.mean.get()
    def e_outer(self):
        return self.mean.get() ** 2 + 1 / self.info.get()

    def set_free(self, free_val):
        if free_val.size != self.__free_size: \
            raise ValueError('Wrong size for UVNParam ' + self.name)
        offset = 0
        offset = set_free_offset(self.mean, free_val, offset)
        offset = set_free_offset(self.info, free_val, offset)
    def get_free(self):
        return np.hstack([ self.mean.get_free(), self.info.get_free() ])

    def set_vector(self, vec):
        if vec.size != self.__vector_size: raise ValueError("Wrong size.")
        offset = 0
        offset = set_vector_offset(self.mean, vec, offset)
        offset = set_vector_offset(self.info, vec, offset)
    def get_vector(self):
        return np.hstack([ self.mean.get_vector(), self.info.get_vector() ])

    def free_size(self):
        return self.__free_size
    def vector_size(self):
        return self.__vector_size
