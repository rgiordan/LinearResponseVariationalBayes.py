import autograd

class JacobianVectorProducts(object):
    def __init__(self, base_fun, order=3):
        self.base_fun = base_fun
        self.order = order
        self.jvp_funs = []
        self.jvp_funs.append(self.base_fun)
        for d in range(order):
            self.jvp_funs.append(
                self.append_jvp(self.jvp_funs[d]))

    # fun should be a function that takes arguments (x, v1, v2, ..., )
    # which computes a jvp at x times the vectors v1, v2...
    def append_jvp(self, fun):
        fun_jvp = autograd.make_jvp(fun, argnum=0)
        def obj_jvp_wrapper(*argv):
            x = argv[0]
            v1 = argv[1]
            if (len(argv) > 2):
                vs = argv[2:]
                return fun_jvp(x, *vs)(v1)[1]
            else:
                return fun_jvp(x)(v1)[1]

        return obj_jvp_wrapper
