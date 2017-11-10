import LinearResponseVariationalBayes as vb
import LinearResponseVariationalBayes.ExponentialFamilies as ef

import numpy as np

class WishartParam(vb.ModelParamsDict):
    def __init__(self, name='', size=2, diag_lb=0.0, min_df=None):
        super().__init__(name=name)
        self.__size = int(size)
        if not min_df:
            min_df = size - 1
        assert min_df >= size - 1
        self.push_param(vb.ScalarParam('df', lb=min_df))
        self.push_param(
            vb.PosDefMatrixParam('v', diag_lb=diag_lb))

    def e(self):
        return self['df'].get() * self['v'].get()
    def e_log_det(self):
        return ef.e_log_det_wishart(self['df'].get(),
                                    self['v'].get())
    def e_inv(self):
        return self['df'].get() * np.linalg.inv(self['v'].get())

    def entropy(self):
        return ef.wishart_entropy(self['df'].get(),
                                  self['v'].get())

    # This is the expected log lkj prior on the inverse of the
    # Wishart-distributed parameter.  E.g. if this object contains the
    # parameters of a Wishart distribution on an information matrix, this
    # returns the expected LKJ prior on the covariance matrix.
    def e_log_lkj_inv_prior(self, lkj_param):
        return ef.expected_ljk_prior(
            lkj_param, self['df'].get(), self['v'].get())
