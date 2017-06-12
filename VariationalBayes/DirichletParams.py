from VariationalBayes import Parameters as par
import autograd.numpy as np
import autograd.scipy as asp

# a vector drawn according to a Dirichlet(alpha) distribution
class DirichletParamVector(object):
    def __init__(self, name='', dim=2, min_alpha = 0.0):
        self.name = name
        self.__dim = dim
        assert min_alpha >= 0, 'alpha parameter must be non-negative'
        self.alpha = par.VectorParam(name + '_alpha', size=dim, lb=min_alpha)
        self.__free_size = self.alpha.free_size()
        self.__vector_size = self.alpha.vector_size()

    def __str__(self):
        return self.name + ':\n' + str(self.alpha)
    def names(self):
        return self.alpha.names()
    def dictval(self):
        return { 'alpha': self.alpha.dictval() }

    def e(self):
        return self.alpha.get() / np.sum(self.alpha.get())

    def e_log(self):
        digamma_sum = asp.special.digamma(np.sum(self.alpha.get()))
        return np.log(self.alpha.get()) / digamma_sum

    def set_free(self, free_val):
        self.alpha.set_free(free_val)

    def get_free(self):
        return self.alpha.get_free()

    def set_vector(self, vec):
        if vec.size != self.__vector_size:
            raise ValueError("Wrong size.")
        self.alpha.set(vec)

    def get_vector(self):
        return self.alpha.get_vector()

    def free_size(self):
        return self.__free_size
    def vector_size(self):
        return self.__vector_size
    def dim(self):
        return self.__dim
