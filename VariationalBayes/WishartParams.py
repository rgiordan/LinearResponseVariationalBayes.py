from VariationalBayes import Parameters as par
from VariationalBayes import MatrixParameters as mat_par
from VariationalBayes import ParameterDictionary as par_dict
import autograd.numpy as np
import autograd.scipy as asp

# from VariationalBayes.Parameters import \
#     free_to_vector_jac_offset, free_to_vector_hess_offset
import VariationalBayes.ExponentialFamilies as ef

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
            mat_par.PosDefMatrixParam('v', diag_lb=diag_lb))

    def __str__(self):
        return self.name + ':\n' + str(self.params)
    def names(self):
        return self.params.names()
    def dictval(self):
        return self.params.dictval()

    def e(self):
        return self.params['df'].get() * self.params['v'].get()
    def e_log_det(self):
        return ef.e_log_det_wishart(self.params['df'].get(),
                                    self.params['v'].get())
    def e_inv(self):
        return self.params['df'].get() * np.linalg.inv(self.params['v'].get())

    def entropy(self):
        return ef.wishart_entropy(self.params['df'].get(),
                                  self.params['v'].get())

    # This is the expected log lkj prior on the inverse of the
    # Wishart-distributed parameter.  E.g. if this object contains the
    # parameters of a Wishart distribution on an information matrix, this
    # returns the expected LKJ prior on the covariance matrix.
    def e_log_lkj_inv_prior(self, lkj_param):
        return ef.expected_ljk_prior(
            lkj_param, self.params['df'].get(), self.params['v'].get())

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
        return self.params.free_size()
    def vector_size(self):
        return self.params.vector_size()
