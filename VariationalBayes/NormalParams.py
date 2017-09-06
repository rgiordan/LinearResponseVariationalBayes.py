import VariationalBayes as vb
# from VariationalBayes import Parameters as par
# from VariationalBayes import MatrixParameters as mat_par
import VariationalBayes.ExponentialFamilies as ef

# from VariationalBayes.Parameters import \
#     free_to_vector_jac_offset, free_to_vector_hess_offset
#
# import autograd.numpy as np
# from scipy.sparse import block_diag

class MVNParam(vb.ModelParamsDict):
    def __init__(self, name='', dim=2, min_info=0.0):
        super().__init__(name=name)
        self.__dim = dim
        self.push_param(vb.VectorParam('mean', dim))
        self.push_param(vb.PosDefMatrixParam('info', dim, diag_lb=min_info))
    # def __str__(self):
    #     return self.name + ':\n' + str(self.mean) + '\n' + str(self.info)
    # def names(self):
    #     return self['mean'].names() + self['info'].names()
    # def dictval(self):
    #     return { 'mean': self['mean'].dictval(), 'info': self['info'].dictval() }
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

    # def set_free(self, free_val):
    #     if free_val.size != self.__free_size: \
    #         raise ValueError('Wrong size for MVNParam ' + self.name)
    #     offset = 0
    #     offset = par.set_free_offset(self.mean, free_val, offset)
    #     offset = par.set_free_offset(self.info, free_val, offset)
    # def get_free(self):
    #     return np.hstack([ self['mean'].get_free(), self['info'].get_free() ])
    #
    # def free_to_vector(self, free_val):
    #     self.set_free(free_val)
    #     return self.get_vector()
    # def free_to_vector_jac(self, free_val):
    #     free_offset = 0
    #     vec_offset = 0
    #     free_offset, vec_offset, mean_jac = free_to_vector_jac_offset(
    #         self.mean, free_val, free_offset, vec_offset)
    #     free_offset, vec_offset, info_jac = free_to_vector_jac_offset(
    #         self.info, free_val, free_offset, vec_offset)
    #     return block_diag((mean_jac, info_jac))
    # def free_to_vector_hess(self, free_val):
    #     free_offset = 0
    #     full_shape = (self.free_size(), self.free_size())
    #     hessians = []
    #     free_offset = free_to_vector_hess_offset(
    #         self.mean, free_val, hessians, free_offset, full_shape)
    #     free_offset = free_to_vector_hess_offset(
    #         self.info, free_val, hessians, free_offset, full_shape)
    #     return np.array(hessians)
    #
    # def set_vector(self, vec):
    #     if vec.size != self.__vector_size: raise ValueError("Wrong size.")
    #     offset = 0
    #     offset = par.set_vector_offset(self.mean, vec, offset)
    #     offset = par.set_vector_offset(self.info, vec, offset)
    # def get_vector(self):
    #     return np.hstack([ self['mean'].get_vector(), self['info'].get_vector() ])
    #
    # def free_size(self):
    #     return self.__free_size
    # def vector_size(self):
    #     return self.__vector_size
    # def dim(self):
    #     return self.__dim


class UVNParam(vb.ModelParamsDict):
    def __init__(self, name='', min_info=0.0):
        super().__init__(name=name)
        self.push_param(vb.ScalarParam('mean'))
        self.push_param(vb.ScalarParam('info', lb=min_info))
        # self.__free_size = self['mean'].free_size() + self['info'].free_size()
        # self.__vector_size = self['mean'].vector_size() + self['info'].vector_size()
    # def __str__(self):
    #     return self.name + ':\n' + str(self.mean) + '\n' + str(self.info)
    # def names(self):
    #     return self['mean'].names() + self['info'].names()
    # def dictval(self):
    #     return { 'mean': self['mean'].dictval(), 'info': self['info'].dictval() }
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

    # def set_free(self, free_val):
    #     if free_val.size != self.__free_size: \
    #         raise ValueError('Wrong size for UVNParam ' + self.name)
    #     offset = 0
    #     offset = par.set_free_offset(self.mean, free_val, offset)
    #     offset = par.set_free_offset(self.info, free_val, offset)
    # def get_free(self):
    #     return np.hstack([ self['mean'].get_free(), self['info'].get_free() ])
    #
    # def free_to_vector(self, free_val):
    #     self.set_free(free_val)
    #     return self.get_vector()
    # def free_to_vector_jac(self, free_val):
    #     free_offset = 0
    #     vec_offset = 0
    #     free_offset, vec_offset, mean_jac = free_to_vector_jac_offset(
    #         self.mean, free_val, free_offset, vec_offset)
    #     free_offset, vec_offset, info_jac = free_to_vector_jac_offset(
    #         self.info, free_val, free_offset, vec_offset)
    #     return block_diag((mean_jac, info_jac))
    # def free_to_vector_hess(self, free_val):
    #     free_offset = 0
    #     full_shape = (self.free_size(), self.free_size())
    #     hessians = []
    #     free_offset = free_to_vector_hess_offset(
    #         self.mean, free_val, hessians, free_offset, full_shape)
    #     free_offset = free_to_vector_hess_offset(
    #         self.info, free_val, hessians, free_offset, full_shape)
    #     return np.array(hessians)
    #
    # def set_vector(self, vec):
    #     if vec.size != self.__vector_size: raise ValueError("Wrong size.")
    #     offset = 0
    #     offset = par.set_vector_offset(self.mean, vec, offset)
    #     offset = par.set_vector_offset(self.info, vec, offset)
    # def get_vector(self):
    #     return np.hstack([ self['mean'].get_vector(), self['info'].get_vector() ])
    #
    # def free_size(self):
    #     return self.__free_size
    # def vector_size(self):
    #     return self.__vector_size


class UVNParamVector(vb.ModelParamsDict):
    def __init__(self, name='', length=2, min_info=0.0):
        super().__init__(name=name)
        self.push_param(vb.VectorParam('mean', length))
        self.push_param(vb.VectorParam('info', length, lb=min_info))
        # self.__free_size = self['mean'].free_size() + self['info'].free_size()
        # self.__vector_size = self['mean'].vector_size() + self['info'].vector_size()
    # def __str__(self):
    #     return self.name + ':\n' + str(self.mean) + '\n' + str(self.info)
    # def names(self):
    #     return self['mean'].names() + self['info'].names()
    # def dictval(self):
    #     return { 'mean': self['mean'].dictval(), 'info': self['info'].dictval() }

    def e(self):
        return self['mean'].get()
    def e_outer(self):
        return self['mean'].get() ** 2 + 1 / self['info'].get()
    def var(self):
        return 1. / self['info'].get()
    def e_exp(self):
        return ef.get_e_lognormal(self['mean'].get(), 1. / self['info'].get())
    def var_exp(self):
        return ef.get_var_lognormal(self['mean'].get(), 1. / self['info'].get())
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
        # self.__free_size = self['mean'].free_size() + self['info'].free_size()
        # self.__vector_size = self['mean'].vector_size() + self['info'].vector_size()
    # def __str__(self):
    #     return self.name + ':\n' + str(self.mean) + '\n' + str(self.info)
    # def names(self):
    #     # TODO: for some reason this makes the unittest fail
    #     pass
    #     #return self['mean'].names() + self['info'].names()
    # def dictval(self):
    #     return { 'mean': self['mean'].dictval(), 'info': self['info'].dictval() }
    def e(self):
        return self['mean'].get()
    def e2(self):
        var = 1 / self['info'].get()
        return self['mean'].get()**2 + var[:, None]

    # def set_free(self, free_val):
    #     if free_val.size != self.__free_size: \
    #         raise ValueError('Wrong size for MVNArray ' + self.name)
    #     offset = 0
    #     offset = par.set_free_offset(self.mean, free_val, offset)
    #     offset = par.set_free_offset(self.info, free_val, offset)
    # def get_free(self):
    #     return np.hstack([ self['mean'].get_free(), self['info'].get_free() ])
    #
    # def free_to_vector(self, free_val):
    #     self.set_free(free_val)
    #     return self.get_vector()
    #
    # def free_to_vector_jac(self, free_val):
    #     free_offset = 0
    #     vec_offset = 0
    #     free_offset, vec_offset, mean_jac = free_to_vector_jac_offset(
    #         self.mean, free_val, free_offset, vec_offset)
    #     free_offset, vec_offset, info_jac = free_to_vector_jac_offset(
    #         self.info, free_val, free_offset, vec_offset)
    #     return block_diag((mean_jac, info_jac))
    # def free_to_vector_hess(self, free_val):
    #     free_offset = 0
    #     full_shape = (self.free_size(), self.free_size())
    #     hessians = []
    #     free_offset = free_to_vector_hess_offset(
    #         self.mean, free_val, hessians, free_offset, full_shape)
    #     free_offset = free_to_vector_hess_offset(
    #         self.info, free_val, hessians, free_offset, full_shape)
    #     return np.array(hessians)
    #
    # def set_vector(self, vec):
    #     if vec.size != self.__vector_size: raise ValueError("Wrong size.")
    #     offset = 0
    #     offset = par.set_vector_offset(self.mean, vec, offset)
    #     offset = par.set_vector_offset(self.info, vec, offset)
    # def get_vector(self):
    #     return np.hstack([ self['mean'].get_vector(), self['info'].get_vector() ])
    #
    # def free_size(self):
    #     return self.__free_size
    # def vector_size(self):
    #     return self.__vector_size
