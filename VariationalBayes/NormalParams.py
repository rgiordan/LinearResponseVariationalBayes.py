import VariationalBayes as vb
import VariationalBayes.ExponentialFamilies as ef

from autograd import numpy as np

class MVNParam(vb.ModelParamsDict):
    def __init__(self, name='', dim=2, min_info=0.0):
        super().__init__(name=name)
        self.__dim = dim
        self.push_param(vb.VectorParam('mean', dim))
        self.push_param(vb.PosDefMatrixParam('info', dim, diag_lb=min_info))

    def e(self):
        return self['mean'].get()
    def cov(self):
        return np.linalg.inv(self['info'].get())
    def e_outer(self):
        mean = self['mean'].get()
        e_outer = np.outer(mean, mean) + self.cov()
        return 0.5 * (e_outer + e_outer.transpose())

    def entropy(self):
        return ef.multivariate_normal_entropy(self['info'].get())


class UVNParam(vb.ModelParamsDict):
    def __init__(self, name='', min_info=0.0):
        super().__init__(name=name)
        self.push_param(vb.ScalarParam('mean'))
        self.push_param(vb.ScalarParam('info', lb=min_info))

    def e(self):
        return self['mean'].get()
    def e_outer(self):
        return self['mean'].get() ** 2 + 1 / self['info'].get()
    def var(self):
        return 1. / self['info'].get()
    def e_exp(self):
        return ef.get_e_lognormal(
        self['mean'].get(), 1. / self['info'].get())
    def var_exp(self):
        return ef.get_var_lognormal(
        self['mean'].get(), 1. / self['info'].get())
    def e2_exp(self):
        return self.e_exp() ** 2 + self.var_exp()
    def entropy(self):
        return ef.univariate_normal_entropy(self['info'].get())


# TODO: better to derive this from UVNParamArray?
class UVNParamVector(vb.ModelParamsDict):
    def __init__(self, name='', length=2, min_info=0.0):
        super().__init__(name=name)
        self.push_param(vb.VectorParam('mean', length))
        self.push_param(vb.VectorParam('info', length, lb=min_info))

    def e(self):
        return self['mean'].get()
    def e_outer(self):
        return self['mean'].get() ** 2 + 1 / self['info'].get()
    def var(self):
        return 1. / self['info'].get()
    def e_exp(self):
        return ef.get_e_lognormal(self['mean'].get(),
                                  1. / self['info'].get())
    def var_exp(self):
        return ef.get_var_lognormal(self['mean'].get(),
                                    1. / self['info'].get())
    def e2_exp(self):
        return self.e_exp() ** 2 + self.var_exp()
    def entropy(self):
        return np.sum(ef.univariate_normal_entropy(self['info'].get()))


class UVNParamArray(vb.ModelParamsDict):
    def __init__(self, name='', shape=(1, 1), min_info=0.0):
        super().__init__(name=name)
        self.push_param(vb.ArrayParam('mean', shape))
        self.push_param(vb.ArrayParam('info', shape, lb=min_info))

    def e(self):
        return self['mean'].get()
    def e_outer(self):
        return self['mean'].get() ** 2 + 1 / self['info'].get()
    def var(self):
        return 1. / self['info'].get()
    def e_exp(self):
        return ef.get_e_lognormal(self['mean'].get(),
                                  1. / self['info'].get())
    def var_exp(self):
        return ef.get_var_lognormal(self['mean'].get(),
                                    1. / self['info'].get())
    def e2_exp(self):
        return self.e_exp() ** 2 + self.var_exp()
    def entropy(self):
        return np.sum(ef.univariate_normal_entropy(self['info'].get()))


# Array of multivariate normals
# for now each row is a draw from a MVN with diagonal constant variance ...
# not sure how to handle a collection of matrices yet
# but for the current IBP model, this is all we need
class MVNArray(vb.ModelParamsDict):
    def __init__(self, name='', shape=(2,2), min_info=0.0):
        super().__init__(name=name)
        # self.name = name
        self.__shape = shape
        self.push_param(vb.ArrayParam('mean', shape=shape))
        self.push_param(vb.VectorParam('info', size=shape[0], lb=min_info))

    def e(self):
        return self['mean'].get()
    def e2(self):
        var = 1 / self['info'].get()
        return self['mean'].get()**2 + var[:, None]
