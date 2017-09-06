
import VariationalBayes as vb
import VariationalBayes.ExponentialFamilies as ef

# TODO: deal with arrays of Dirichlet parameters in ExponentialFamilies
import autograd.scipy as sp

# from VariationalBayes import Parameters as par
# import autograd.numpy as np
# import autograd.scipy as asp
#
# from VariationalBayes.Parameters import \
#     free_to_vector_jac_offset, free_to_vector_hess_offset
#
# from scipy.sparse import block_diag

# a vector drawn according to a Dirichlet(alpha) distribution
class DirichletParamVector(vb.ModelParamsDict):
    def __init__(self, name='', dim=2, min_alpha = 0.0):
        super().__init__(name=name)
        self.__dim = dim
        assert min_alpha >= 0, 'alpha parameter must be non-negative'
        self.push_param(vb.VectorParam('alpha', size=dim, lb=min_alpha))

    def e(self):
        return self['alpha'].get() / np.sum(self['alpha'].get())

    def e_log(self):
        return ef.get_e_log_dirichlet(self['alpha'].get())

    def entropy(self):
        return ef.dirichlet_entropy(self['alpha'].get())


# each row is a draw from a Dirichlet distribution
class DirichletParamArray(vb.ModelParamsDict):
    def __init__(self, name='', shape=(1,2), min_alpha = 0.0):
        super().__init__(name=name)
        self.__shape = shape
        assert min_alpha >= 0, 'alpha parameter must be non-negative'
        self.push_param(vb.ArrayParam('alpha', shape=shape, lb=min_alpha))

    def e(self):
        denom = np.sum(self.alpha.get(),1)
        return self.alpha.get() / denom[:,None]

    def e_log(self):
        digamma_sum = sp.special.digamma(np.sum(self.alpha.get(), 1))
        return sp.special.digamma(self.alpha.get()) - digamma_sum[:, None]
