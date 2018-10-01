import autograd
import autograd.numpy as np
import autograd.scipy as sp
from autograd.core import primitive, defvjp, defjvp

from autograd.numpy.linalg import slogdet
from autograd.numpy.linalg import solve

def solve_jvp_0(g, ans, x, y):
    return np.linalg.solve(x, g) @ ans

def solve_jvp_1(g, ans, x, y):
    return np.linalg.solve(x, g)

# defjvp(solve, lambda g, ans: lambda x, y: np.linalg.solve(x, g) @ ans)
defjvp(solve, solve_jvp_0, solve_jvp_1)

def slogdet_jvp(g, ans, x):
    print('g', g.shape, g)
    print('ans', ans)
    return 0, np.trace(np.linalg.solve(x, g.T))

#defvjp(slogdet, lambda ans, x: lambda g: add2d(g[1]) * T(inv(x)))
defjvp(slogdet, slogdet_jvp)
