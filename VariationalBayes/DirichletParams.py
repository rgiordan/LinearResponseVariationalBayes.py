from VariationalBayes import Parameters as par
import autograd.numpy as np
import autograd.scipy as asp

# a vector drawn according to a Dirichlet(alpha) distribution
class DirichletParamVector(object):
    def __init__(self, name='', dim=2, min_alpha = 0.0):
        self.name = name
        self.__dim = dim
        self.alpha = par.VectorParam(name + '_alpha', size=dim, lb=0.0)
        self.__free_size = self.alpha.free_size()
        self.__vector_size = self.alpha.vector_size()

    def __str__(self):
        return self.name + ':\n' + str(self.alpha)
    def names(self):
        return self.alpha.names()
    def dictval(self):
        return { 'alpha': self.alpha.dictval() }

    def e(self):
        denom = np.sum(self.alpha.get())
        return np.array([self.alpha.get()[i] / denom \
                            for i in range(self.__dim)])
    def e_log(self):
        digamma_sum = asp.special.digamma(np.sum(self.alpha.get()))
        return np.array([np.log(self.alpha.get()[i]) / digamma_sum \
                            for i in range(self.__dim)])

    def set_free(self, free_val):
        if free_val.size != self.__free_size: \
            raise ValueError('Wrong size for DirichletParamVector ' + self.name)
        offset = 0
        offset = par.set_free_offset(self.alpha, free_val, offset)

    def get_free(self):
        return self.alpha.get_free()

    def set_vector(self, vec):
        if vec.size != self.__vector_size: raise ValueError("Wrong size.")
        offset = 0
        offset = par.set_vector_offset(self.alpha, vec, offset)

    def get_vector(self):
        return self.alpha.get_vector()

    def free_size(self):
        return self.__free_size
    def vector_size(self):
        return self.__vector_size
    def dim(self):
        return self.__dim
