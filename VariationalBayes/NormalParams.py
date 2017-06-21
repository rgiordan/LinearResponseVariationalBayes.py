from VariationalBayes import Parameters as par
from VariationalBayes import MatrixParameters as mat_par

from VariationalBayes.Parameters import \
    free_to_vector_jac_offset, free_to_vector_hess_offset

import autograd.numpy as np
from scipy.sparse import block_diag

class MVNParam(object):
    def __init__(self, name='', dim=2, min_info=0.0):
        self.name = name
        self.__dim = dim
        self.mean = par.VectorParam(name + '_mean', dim)
        self.info = mat_par.PosDefMatrixParam(name + '_info', dim, diag_lb=min_info)
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
        e_outer = np.outer(mean, mean) + np.linalg.inv(info)
        return 0.5 * (e_outer + e_outer.transpose())

    def set_free(self, free_val):
        if free_val.size != self.__free_size: \
            raise ValueError('Wrong size for MVNParam ' + self.name)
        offset = 0
        offset = par.set_free_offset(self.mean, free_val, offset)
        offset = par.set_free_offset(self.info, free_val, offset)
    def get_free(self):
        return np.hstack([ self.mean.get_free(), self.info.get_free() ])

    def free_to_vector(self, free_val):
        self.set_free(free_val)
        return self.get_vector()
    def free_to_vector_jac(self, free_val):
        free_offset = 0
        vec_offset = 0
        free_offset, vec_offset, mean_jac = free_to_vector_jac_offset(
            self.mean, free_val, free_offset, vec_offset)
        free_offset, vec_offset, info_jac = free_to_vector_jac_offset(
            self.info, free_val, free_offset, vec_offset)
        return block_diag((mean_jac, info_jac))
    def free_to_vector_jac(self, free_val):
        free_offset = 0
        vec_offset = 0
        free_offset, vec_offset, mean_jac = free_to_vector_jac_offset(
            self.mean, free_val, free_offset, vec_offset)
        free_offset, vec_offset, info_jac = free_to_vector_jac_offset(
            self.info, free_val, free_offset, vec_offset)
        return block_diag((mean_jac, info_jac))
    def free_to_vector_hess(self, free_val):
        free_offset = 0
        hessians = []
        free_offset = free_to_vector_hess_offset(
            self.mean, free_val, hessians, free_offset)
        free_offset = free_to_vector_hess_offset(
            self.info, free_val, hessians, free_offset)
        return np.array(hessians)


    def set_vector(self, vec):
        if vec.size != self.__vector_size: raise ValueError("Wrong size.")
        offset = 0
        offset = par.set_vector_offset(self.mean, vec, offset)
        offset = par.set_vector_offset(self.info, vec, offset)
    def get_vector(self):
        return np.hstack([ self.mean.get_vector(), self.info.get_vector() ])

    def free_size(self):
        return self.__free_size
    def vector_size(self):
        return self.__vector_size
    def dim(self):
        return self.__dim


class UVNParam(object):
    def __init__(self, name='', min_info=0.0):
        self.name = name
        self.mean = par.ScalarParam(name + '_mean')
        self.info = par.ScalarParam(name + '_info', lb=min_info)
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
        offset = par.set_free_offset(self.mean, free_val, offset)
        offset = par.set_free_offset(self.info, free_val, offset)
    def get_free(self):
        return np.hstack([ self.mean.get_free(), self.info.get_free() ])

    def set_vector(self, vec):
        if vec.size != self.__vector_size: raise ValueError("Wrong size.")
        offset = 0
        offset = par.set_vector_offset(self.mean, vec, offset)
        offset = par.set_vector_offset(self.info, vec, offset)
    def get_vector(self):
        return np.hstack([ self.mean.get_vector(), self.info.get_vector() ])

    def free_size(self):
        return self.__free_size
    def vector_size(self):
        return self.__vector_size


class UVNParamVector(object):
    def __init__(self, name='', length=2, min_info=0.0):
        self.name = name
        self.mean = par.VectorParam(name + '_mean', length)
        self.info = par.VectorParam(name + '_info', length, lb=min_info)
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
        offset = par.set_free_offset(self.mean, free_val, offset)
        offset = par.set_free_offset(self.info, free_val, offset)
    def get_free(self):
        return np.hstack([ self.mean.get_free(), self.info.get_free() ])

    def set_vector(self, vec):
        if vec.size != self.__vector_size: raise ValueError("Wrong size.")
        offset = 0
        offset = par.set_vector_offset(self.mean, vec, offset)
        offset = par.set_vector_offset(self.info, vec, offset)
    def get_vector(self):
        return np.hstack([ self.mean.get_vector(), self.info.get_vector() ])

    def free_size(self):
        return self.__free_size
    def vector_size(self):
        return self.__vector_size
