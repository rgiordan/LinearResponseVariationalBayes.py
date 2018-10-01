import autograd
import autograd.numpy as np
import autograd.scipy as sp
from autograd.core import primitive, defvjp, defjvp

from autograd.numpy.linalg import slogdet

def slogdet_jvp(g, ans, x):
    return 0, np.array(np.sum(np.linalg.solve(x, g)))

#defvjp(slogdet, lambda ans, x: lambda g: add2d(g[1]) * T(inv(x)))
defjvp(slogdet, slogdet_jvp)
