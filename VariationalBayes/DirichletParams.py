from VariationalBayes import Parameters as par
import autograd.numpy as np
import autograd.scipy as asp

from VariationalBayes.Parameters import \
    free_to_vector_jac_offset, free_to_vector_hess_offset

from scipy.sparse import block_diag

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
        return asp.special.digamma(self.alpha.get()) - digamma_sum

    def set_free(self, free_val):
        self.alpha.set_free(free_val)

    def get_free(self):
        return self.alpha.get_free()

    def free_to_vector(self, free_val):
        self.set_free(free_val)
        return self.get_vector()

    def free_to_vector_jac(self, free_val):
        return self.alpha.free_to_vector_jac(free_val)

    def free_to_vector_hess(self, free_val):
        return self.alpha.free_to_vector_hess(free_val)

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

# each row is a draw from a Dirichlet distribution
class DirichletParamArray(object):
    def __init__(self, name='', shape=(1,2), min_alpha = 0.0):
        self.name = name
        self.__shape = shape
        assert min_alpha >= 0, 'alpha parameter must be non-negative'
        self.alpha = par.ArrayParam(name + '_alpha', shape=shape, lb=min_alpha)
        self.__free_size = self.alpha.free_size()
        self.__vector_size = self.alpha.vector_size()

    def __str__(self):
        return self.name + ':\n' + str(self.alpha)
    def names(self):
        return self.alpha.names()
    def dictval(self):
        return { 'alpha': self.alpha.dictval() }

    def e(self):
        denom = np.sum(self.alpha.get(),1)
        return self.alpha.get() / denom[:,None]

    def e_log(self):
        digamma_sum = asp.special.digamma(np.sum(self.alpha.get(),1))
        return asp.special.digamma(self.alpha.get()) - digamma_sum[:,None]

    def set_free(self, free_val):
        self.alpha.set_free(free_val)

    def get_free(self):
        return self.alpha.get_free()

    def free_to_vector(self, free_val):
        self.set_free(free_val)
        return self.get_vector()

    def free_to_vector_jac(self, free_val):
        return self.alpha.free_to_vector_jac(free_val)

    def free_to_vector_hess(self, free_val):
        return self.alpha.free_to_vector_hess(free_val)

    def set_vector(self, vec):
        if vec.size != self.vector_size():
            raise ValueError("Wrong size.")
        self.alpha.set_vector(vec)

    def get_vector(self):
        return self.alpha.get_vector()

    def free_size(self):
        return self.__free_size
    def vector_size(self):
        return self.__vector_size
    def shape(self):
        return self.__shape
