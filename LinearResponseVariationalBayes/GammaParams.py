import LinearResponseVariationalBayes as vb
import LinearResponseVariationalBayes.ExponentialFamilies as ef

class GammaParam(vb.ModelParamsDict):
    def __init__(self, name='', min_shape=0.0, min_rate=0.0):
        super().__init__(name=name)
        self.push_param(vb.ScalarParam('shape', lb=min_shape))
        self.push_param(vb.ScalarParam('rate', lb=min_rate))
    def e(self):
        return self['shape'].get() / self['rate'].get()
    def e_log(self):
        return ef.get_e_log_gamma(
            shape=self['shape'].get(), rate=self['rate'].get())
    def entropy(self):
        return ef.gamma_entropy(
            shape=self['shape'].get(), rate=self['rate'].get())
