import VariationalBayes as vb
from VariationalBayes.Parameters import \
    convert_vector_to_free_hessian
from VariationalBayes import ModelParamsDict
import autograd
import autograd.numpy as np
import autograd.scipy as sp

import scipy as osp


# par should be a Parameter type.
# fun should be a function that takes no arguments but which is
# bound to par, i.e. which evaluates to a float that depends on the
# value of the parameters par.
class Objective(object):
    def __init__(self, par, fun):
        self.par = par
        self.fun = fun

        self.fun_free_grad = autograd.grad(self.fun_free)
        self.fun_free_hessian = autograd.hessian(self.fun_free)
        self.fun_free_hvp = autograd.hessian_vector_product(self.fun_free)

        self.fun_vector_grad = autograd.grad(self.fun_vector)
        self.fun_vector_hessian = autograd.hessian(self.fun_vector)
        self.fun_vector_hvp = autograd.hessian_vector_product(self.fun_vector)

    def fun_free(self, free_val):
        self.par.set_free(free_val)
        return self.fun()

    def fun_vector(self, vec_val):
        self.par.set_vector(vec_val)
        return self.fun()
