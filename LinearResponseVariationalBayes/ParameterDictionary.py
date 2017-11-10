import math
import copy
import numbers

import autograd.numpy as np
import autograd.scipy as sp
from autograd.core import primitive

from collections import OrderedDict
from LinearResponseVariationalBayes import Parameters as par

from LinearResponseVariationalBayes.Parameters import \
    free_to_vector_jac_offset, free_to_vector_hess_offset

from scipy.sparse import block_diag

import time

class ModelParamsDict(object):
    def __init__(self, name='ModelParamsDict'):
        self.param_dict = OrderedDict()
        self.free_indices_dict = OrderedDict()
        self.vector_indices_dict = OrderedDict()
        self.name = name
        # You will want free_size and vector_size to be different when you
        # are encoding simplexes.
        self.__free_size = 0
        self.__vector_size = 0
    def __str__(self):
        return self.name + ':\n' + \
            '\n'.join([ '\t' + str(param) for param in self.param_dict.values() ])
    def __getitem__(self, key):
        return self.param_dict[key]
    def push_param(self, param):
        self.param_dict[param.name] = param
        self.free_indices_dict[param.name] = \
            range(self.__free_size, self.__free_size + param.free_size())
        self.vector_indices_dict[param.name] = \
            range(self.__vector_size, self.__vector_size + param.vector_size())
        self.__free_size = self.__free_size + param.free_size()
        self.__vector_size = self.__vector_size + param.vector_size()
    def set_name(self, name):
        self.name = name
    def dictval(self):
        result = {}
        for param in self.param_dict.values():
            result[param.name] = param.dictval()
        return result

    def set_free(self, vec):
        if vec.size != self.__free_size:
            error_string = \
                'Wrong size for parameter {}.  Expected {}, got {}'.format(
                    self.name, str(self.__free_size), str(vec.size))
            raise ValueError(error_string)
        offset = 0
        for param in self.param_dict.values():
            offset = par.set_free_offset(param, vec, offset)
    def get_free(self):
        return np.hstack([ par.get_free() for par in self.param_dict.values() ])

    def free_to_vector(self, free_val):
        self.set_free(free_val)
        return self.get_vector()
    def free_to_vector_jac(self, free_val):
        free_offset = 0
        vec_offset = 0
        jacobians = []
        for param in self.param_dict.values():
            free_offset, vec_offset, param_jac = free_to_vector_jac_offset(
                param, free_val, free_offset, vec_offset)
            jacobians.append(param_jac)
        return block_diag(jacobians)
    def free_to_vector_hess(self, free_val):
        free_offset = 0
        full_shape = (self.free_size(), self.free_size())
        hessians = []
        for param in self.param_dict.values():
            free_offset = free_to_vector_hess_offset(
                param, free_val, hessians, free_offset, full_shape)
        return np.array(hessians)

    def set_vector(self, vec):
        if vec.size != self.__vector_size:
            error_string = \
                'Wrong size for parameter {}.  Expected {}, got {}'.format(
                    self.name, str(self.__vector_size), str(vec.size))
            raise ValueError(error_string)
        offset = 0
        for param in self.param_dict.values():
            offset = par.set_vector_offset(param, vec, offset)
    def get_vector(self):
        return np.hstack(
            [ par.get_vector() for par in self.param_dict.values() ])

    def names(self):
        return np.concatenate(
            [ np.atleast_1d(param.names()) for param in self.param_dict.values()])
    def free_size(self):
        return self.__free_size
    def vector_size(self):
        return self.__vector_size
