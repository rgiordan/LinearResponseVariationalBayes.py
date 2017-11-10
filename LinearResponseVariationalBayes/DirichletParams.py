
import LinearResponseVariationalBayes as vb
import LinearResponseVariationalBayes.ExponentialFamilies as ef

# TODO: deal with arrays of Dirichlet parameters in ExponentialFamilies
import autograd.scipy as sp


# The first dimension are the elements of the dirichlet distribution,
# and the other dimensions are an array.
class DirichletParamArray(vb.ModelParamsDict):
    def __init__(self, name='', shape=(1, 2), min_alpha=0.0, val=None):
        super().__init__(name=name)
        self.__shape = shape
        assert min_alpha >= 0, 'alpha parameter must be non-negative'
        self.push_param(
            vb.ArrayParam('alpha', shape=shape, lb=min_alpha, val=val))

    def e(self):
        return ef.get_e_dirichlet(self['alpha'].get())

    def e_log(self):
        return ef.get_e_log_dirichlet(self['alpha'].get())

    def entropy(self):
        return ef.dirichlet_entropy(self['alpha'].get())
