from Parameters import ArrayParam
import autograd.numpy as np
import autograd.scipy as asp




class SimplexParam(object):
    def __init__(self, name='', obs=2, dim=2, min_prob=0.0, max_prob=0.0):
        self.name = name


class MultinomialParam(object):
    def __init__(self, name='', obs=2, dim=2, min_prob=0.0, max_prob=0.0):
        pass
