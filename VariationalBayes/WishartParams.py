from VariationalBayes import Parameters as par
from VariationalBayes import ParameterDictionary as par_dict
import autograd.numpy as np
import autograd.scipy as asp

# from VariationalBayes.Parameters import \
#     free_to_vector_jac_offset, free_to_vector_hess_offset
from VariationalBayes.ExponentialFamilies import \
    multivariate_gammaln, multivariate_digamma

from scipy.sparse import block_diag

class WishartParam(object):
    def __init__(self, name='', size=2, diag_lb=0.0, min_df=None):
        self.name = name
        self.__size = int(size)
        if not min_df:
            min_df = size - 1
        assert min_df >= size - 1
        self.params = par_dict.ModelParamsDict(name='params')
        self.params.push_param(par.ScalarParam('df', lb=min_df))
        self.params.push_param(
            par.PosDefMatrixParam(name + '_mat', diag_lb=min_rate))

    def __str__(self):
        return self.name + ':\n' + str(self.params)
    def names(self):
        return self.params.names()
    def dictval(self):
        return self.params.dictval()

    def e(self):
        return self.df.get() * self.v.get()
    def e_log_det(self):
        s, log_det_v = np.slogdet(self.v.get())
        assert s > 0
        return multivariate_digamma(0.5 * self.df.get(), self.__size) + \
               self.__size * np.log(2) + log_det_v

    def set_free(self, free_val):
        self.params.set_free(free_val)
    def get_free(self):
        return self.params.get_free()

    def free_to_vector(self, free_val):
        return self.free_to_vector(free_val)
    def free_to_vector_jac(self, free_val):
        return self.params.free_to_vector_jac(free_val)
    def free_to_vector_hess(self, free_val):
        return self.params.free_to_vector_hess(free_val)

    def set_vector(self, vec):
        self.params.set_vector(vec)
    def get_vector(self):
        return self.params.get_vector()

    def free_size(self):
        return self.self.params.free_size()
    def vector_size(self):
        return self.self.params.vector_size()
