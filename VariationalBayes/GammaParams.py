from Parameters import ScalarParam, set_free_offset, get_free_offset
import numpy as np
from scipy.special import digamma

class GammaParam(object):
    def __init__(self, name, min_rate=0.0):
        self.name = name
        self.shape = ScalarParam(name + '_shape')
        self.rate = ScalarParam(name + '_rate', lb=min_rate)
        self.__free_size = self.shape.free_size() + self.rate.free_size()
    def __str__(self):
        return self.name + ':\n' + str(self.shape) + '\n' + str(self.rate)
    def names(self):
        return self.shape.names() + self.rate.names()
    def e(self):
        return self.shape.get() / self.rate.get()
    def e_log(self):
        return digamma(self.shape.get()) - np.log(self.rate.get())
    def set_free(self, free_val):
        if free_val.size != self.__free_size: \
            raise ValueError('Wrong size for GammaParam ' + self.name)
        offset = 0
        offset = set_free_offset(self.shape, free_val, offset)
        offset = set_free_offset(self.rate, free_val, offset)
    def get_free(self):
        vec = np.empty(self.__free_size)
        offset = 0
        offset = get_free_offset(self.shape, vec, offset)
        offset = get_free_offset(self.rate, vec, offset)
        return vec
    def free_size(self):
        return self.__free_size
    