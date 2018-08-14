#!/usr/bin/env python3

import autograd
from autograd import numpy as np
import scipy as sp
import numpy.testing as np_test
import unittest
import LinearResponseVariationalBayes as vb
import LinearResponseVariationalBayes.SparseObjectives as obj_lib
import LinearResponseVariationalBayes.OptimizationUtils as opt_lib
import LinearResponseVariationalBayes.ConjugateGradient as cg


class Model(object):
    def __init__(self, dim):
        self.dim = dim
        self.x = vb.VectorParam('x', size=dim, lb=-2.0, ub=5.0)
        self.y = vb.VectorParam('y', size=dim, lb=-2.0, ub=5.0)
        self.a_mat = np.full((dim, dim), 0.1) + np.eye(dim)

        # For testing the Jacobian
        self.b_mat = self.a_mat[0:(dim - 1), 0:dim]

        self.set_inits()

        self.opt_x = np.linspace(1., 2., self.dim)

        self.preconditioner = np.eye(dim)

    def set_random(self):
        self.x.set_free(np.random.random(self.x.free_size()))

    def set_inits(self):
        #self.x.set_vector(np.ones(self.dim))
        self.x.set_vector(np.linspace(0., 1., self.dim))

    def set_opt(self):
        self.x.set_vector(self.opt_x)

    def f_of_x(self, x):
        x_c = x - self.opt_x
        return np.matmul(x_c.T, np.matmul(self.a_mat, x_c))

    def f(self):
        return self.f_of_x(self.x.get())

    def get_x_vec(self):
        # For testing the Jacobian
        return np.matmul(self.b_mat, self.x.get())

    def f_conditioned(self):
        # Note that
        # df / dy = (dx' / dy) df / dx = (dy / dx')^{-1} df / dx
        # So the transform should multiply by the inverse of the preconditioner.
        y_free = np.matmul(self.preconditioner, self.x.get_free())
        self.y.set_free(y_free)
        return self.f_of_x(self.y.get())


class TwoParamModel(object):
    def __init__(self, dim=3, x_constrained=True, y_constrained=True):
        self.a = np.random.random((dim, dim))
        self.a = np.matmul(self.a, np.transpose(self.a)) + np.eye(dim)

        self.par = vb.ModelParamsDict()
        if x_constrained:
            self.par.push_param(vb.VectorParam('x', size=dim, lb=0))
        else:
            self.par.push_param(vb.VectorParam('x', size=dim))

        if y_constrained:
            self.par.push_param(vb.VectorParam('y', size=dim, lb=1))
        else:
            self.par.push_param(vb.VectorParam('y', size=dim))

    def set_random(self):
        self.par.set_free(np.random.random(self.par.free_size()))

    def fun(self):
        x = self.par['x'].get()
        y = self.par['y'].get()
        return np.exp(np.sum(x * y) / 3)

    def convert_y_to_x(self, x):
        return x ** 2

    def set_y_from_x(self):
        # Used for testing the ParameterConverter class.
        self.par['y'].set_vector(
            self.convert_y_to_x(self.par['x'].get_vector()))


class TestObjectiveClass(unittest.TestCase):
    # For every parameter type, execute all the required methods.
    def test_objective(self):
        model = Model(dim=3)
        objective = obj_lib.Objective(par=model.x, fun=model.f)

        model.set_inits()
        x_free = model.x.get_free()
        x_vec = model.x.get_vector()

        model.set_opt()
        self.assertTrue(objective.fun_free(x_free) > 0.0)
        np_test.assert_array_almost_equal(
            objective.fun_free(x_free), objective.fun_vector(x_vec))

        grad = objective.fun_free_grad(x_free)
        hess = objective.fun_free_hessian(x_free)
        np_test.assert_array_almost_equal(
            np.matmul(hess, grad), objective.fun_free_hvp(x_free, grad))

        self.assertTrue(objective.fun_vector(x_vec) > 0.0)
        grad = objective.fun_vector_grad(x_vec)
        hess = objective.fun_vector_hessian(x_vec)
        np_test.assert_array_almost_equal(
            np.matmul(hess, grad), objective.fun_vector_hvp(x_free, grad))

        # Test Jacobians.
        vec_objective = obj_lib.Objective(par=model.x, fun=model.get_x_vec)
        vec_jac = vec_objective.fun_vector_jacobian(x_vec)
        np_test.assert_array_almost_equal(model.b_mat, vec_jac)

        free_jac = vec_objective.fun_free_jacobian(x_free)
        x_free_to_vec_jac = \
            model.x.free_to_vector_jac(x_free).todense()
        np_test.assert_array_almost_equal(
            np.matmul(model.b_mat, np.transpose(x_free_to_vec_jac)),
            free_jac)

        # Test the preconditioning
        preconditioner = 2.0 * np.eye(model.dim)
        preconditioner[model.dim - 1, 0] = 0.1 # Add asymmetry for testing!
        objective.preconditioner = preconditioner

        np_test.assert_array_almost_equal(
            objective.fun_free_cond(x_free),
            objective.fun_free(np.matmul(preconditioner, x_free)),
            err_msg='Conditioned function values')

        fun_free_cond_grad = autograd.grad(objective.fun_free_cond)
        grad_cond = objective.fun_free_grad_cond(x_free)
        np_test.assert_array_almost_equal(
            fun_free_cond_grad(x_free), grad_cond,
            err_msg='Conditioned gradient values')

        fun_free_cond_hessian = autograd.hessian(objective.fun_free_cond)
        hess_cond = objective.fun_free_hessian_cond(x_free)
        np_test.assert_array_almost_equal(
            fun_free_cond_hessian(x_free), hess_cond,
            err_msg='Conditioned Hessian values')

        fun_free_cond_hvp = autograd.hessian_vector_product(
            objective.fun_free_cond)
        np_test.assert_array_almost_equal(
            fun_free_cond_hvp(x_free, grad_cond),
            objective.fun_free_hvp_cond(x_free, grad_cond),
            err_msg='Conditioned Hessian vector product values')

    # Test that the objective functions can take additional keyword arguments.
    def test_objective_keywords(self):
        x = vb.VectorParam('x', size=2)
        def objective_fun(y, z=1.):
            return np.sum(x.get()**2) * z * y

        objective = obj_lib.Objective(par=x, fun=objective_fun)
        x_val = np.array([0., 1.])

        # Check that the keywords are passed to fun_free and fun_vector.
        np_test.assert_array_almost_equal(
            1 * 2 * 1, objective.fun_free(x_val, 2))
        np_test.assert_array_almost_equal(
            1 * 2 * 3, objective.fun_free(x_val, 2, z=3))
        np_test.assert_array_almost_equal(
            1 * 2 * 1, objective.fun_free(x_val, 2, verbose=True))
        np_test.assert_array_almost_equal(
            1 * 2 * 3, objective.fun_free(x_val, 2, verbose=True, z=3))
        np_test.assert_array_almost_equal(
            1 * 2 * 3, objective.fun_free(x_val, 2, z=3, verbose=True))
        np_test.assert_array_almost_equal(
            1 * 2 * 1, objective.fun_vector(x_val, 2))
        np_test.assert_array_almost_equal(
            1 * 2 * 3, objective.fun_vector(x_val, 2, z=3))

        hvp_vec = np.array([2., 3.])
        np_test.assert_array_almost_equal(
            2 * x_val * 2 * 3, objective.fun_free_grad(x_val, 2, z=3))
        np_test.assert_array_almost_equal(
            2 * x_val * 2 * 3, objective.fun_vector_grad(x_val, 2, z=3))
        np_test.assert_array_almost_equal(
            2 * x_val * 2 * 3, objective.fun_free_jacobian(x_val, 2, z=3))
        np_test.assert_array_almost_equal(
            2 * x_val * 2 * 3, objective.fun_vector_jacobian(x_val, 2, z=3))
        np_test.assert_array_almost_equal(
            2 * np.eye(2) * 2 * 3, objective.fun_free_hessian(x_val, 2, z=3))
        np_test.assert_array_almost_equal(
            2 * np.eye(2) * 2 * 3, objective.fun_vector_hessian(x_val, 2, z=3))
        np_test.assert_array_almost_equal(
            2 * hvp_vec * 2 * 3,
            objective.fun_free_hvp(x_val, 2, hvp_vec, z=3))
        np_test.assert_array_almost_equal(
            2 * hvp_vec * 2 * 3,
            objective.fun_vector_hvp(x_val, 2, hvp_vec, z=3))

        # Test preconditioned functions.
        objective.preconditioner = 4 * np.eye(2)
        np_test.assert_array_almost_equal(
            1 * 2 * 3 * 16, objective.fun_free_cond(x_val, 2, z=3))
        np_test.assert_array_almost_equal(
            2 * x_val * 2 * 3 * 16,
            objective.fun_free_grad_cond(x_val, 2, z=3))
        np_test.assert_array_almost_equal(
            2 * np.eye(2) * 2 * 3 * 16,
            objective.fun_free_hessian_cond(x_val, 2, z=3))
        np_test.assert_array_almost_equal(
            2 * hvp_vec * 2 * 3 * 16,
            objective.fun_free_hvp_cond(x_val, 2, hvp_vec, z=3))


        x1 = vb.VectorParam('x1', size=2)
        x2 = vb.VectorParam('x1', size=2)
        x1_val = np.array([0., 1.])
        x2_val = np.array([1., 2.])
        def two_param_objective_fun(y, z=1.):
            return np.sum(x1.get() * x2.get()) * z * y
        twopar_obj = obj_lib.TwoParameterObjective(
            x1, x2, two_param_objective_fun)
        np_test.assert_array_almost_equal(
            2 * 2 * 3, twopar_obj.fun_free(x1_val, x2_val, 2, z=3))
        np_test.assert_array_almost_equal(
            2 * 2 * 3, twopar_obj.fun_vector(x1_val, x2_val, 2, z=3))
        np_test.assert_array_almost_equal(
            np.eye(2) * 2 * 3,
            twopar_obj.fun_free_hessian12(x1_val, x2_val, 2, z=3))
        np_test.assert_array_almost_equal(
            np.eye(2) * 2 * 3,
            twopar_obj.fun_free_hessian21(x1_val, x2_val, 2, z=3))
        np_test.assert_array_almost_equal(
            np.eye(2) * 2 * 3,
            twopar_obj.fun_vector_hessian12(x1_val, x2_val, 2, z=3))
        np_test.assert_array_almost_equal(
            np.eye(2) * 2 * 3,
            twopar_obj.fun_vector_hessian21(x1_val, x2_val, 2, z=3))


    def test_parameter_converter(self):
        model = TwoParamModel()
        model.set_random()
        model.set_y_from_x()
        x_free = model.par['x'].get_free()
        y_free = model.par['y'].get_free()
        x_vec = model.par['x'].get_vector()
        y_vec = model.par['y'].get_vector()

        param_converter = obj_lib.ParameterConverter(
            model.par['x'], model.par['y'], model.set_y_from_x)

        np_test.assert_array_almost_equal(
            param_converter.converter_vec_to_vec(x_vec), y_vec)
        np_test.assert_array_almost_equal(
            param_converter.converter_vec_to_free(x_vec), y_free)
        np_test.assert_array_almost_equal(
            param_converter.converter_free_to_vec(x_free), y_vec)
        np_test.assert_array_almost_equal(
            param_converter.converter_free_to_free(x_free), y_free)

        # The function convert_y_to_x corrseponds to the vector to vector
        # map.  Use the free to vec Jacobians to convert to the other maps.
        get_converter_jacobian = autograd.jacobian(model.convert_y_to_x)
        x_free_to_vec_jac = \
            model.par['x'].free_to_vector_jac(x_free).todense()
        y_free_to_vec_jac = \
            model.par['y'].free_to_vector_jac(y_free).todense()

        vec_to_vec_jac = get_converter_jacobian(x_vec)
        np_test.assert_array_almost_equal(
            vec_to_vec_jac,
            param_converter.vec_to_vec_jacobian(x_vec))

        free_to_vec_jac = np.matmul(vec_to_vec_jac, x_free_to_vec_jac)
        np_test.assert_array_almost_equal(
            free_to_vec_jac,
            param_converter.free_to_vec_jacobian(x_free))

        np_test.assert_array_almost_equal(
            np.linalg.solve(y_free_to_vec_jac, vec_to_vec_jac),
            param_converter.vec_to_free_jacobian(x_vec))

        np_test.assert_array_almost_equal(
            np.linalg.solve(y_free_to_vec_jac, free_to_vec_jac),
            param_converter.free_to_free_jacobian(x_free))


    def test_two_parameter_objective(self):
        # The main purpose of the TwoParameterObjective is to get off-diagonal
        # Hessians without calculating the full Hessian.  To test, we compare
        # with the appropriate sub-matrices of the full Hessian.
        model = TwoParamModel()
        model.set_random()

        objective_full = obj_lib.Objective(model.par, model.fun)

        objective = obj_lib.TwoParameterObjective(
            model.par['x'], model.par['y'], model.fun)

        # Get the indices of the sub-matrices.
        # In this case, the indices are the same for the free and vector
        # parameters.
        par_index = obj_lib.make_index_param(model.par)
        ind_12 = np.ix_(par_index['x'].get_vector(),
                        par_index['y'].get_vector())
        ind_21 = np.ix_(par_index['y'].get_vector(),
                        par_index['x'].get_vector())

        par_free = model.par.get_free()
        par_vec = model.par.get_vector()
        x_free = model.par['x'].get_free()
        y_free = model.par['y'].get_free()
        x_vec = model.par['x'].get_vector()
        y_vec = model.par['y'].get_vector()

        np_test.assert_array_almost_equal(
            model.fun(), objective.fun_free(x_free, y_free))
        np_test.assert_array_almost_equal(
            model.fun(), objective.fun_vector(x_vec, y_vec))
        np_test.assert_array_almost_equal(
            model.fun(),
            objective.eval_fun(
                x_free, y_vec, val1_is_free=True, val2_is_free=False))
        np_test.assert_array_almost_equal(
            model.fun(),
            objective.eval_fun(
                x_vec, y_free, val1_is_free=False, val2_is_free=True))

        full_free_hess = objective_full.fun_free_hessian(par_free)
        np_test.assert_array_almost_equal(
            full_free_hess[ind_12],
            objective.fun_free_hessian12(x_free, y_free))
        np_test.assert_array_almost_equal(
            full_free_hess[ind_21],
            objective.fun_free_hessian21(x_free, y_free))

        full_vec_hess = objective_full.fun_vector_hessian(par_vec)
        np_test.assert_array_almost_equal(
            full_vec_hess[ind_12],
            objective.fun_vector_hessian12(x_vec, y_vec))
        np_test.assert_array_almost_equal(
            full_vec_hess[ind_21],
            objective.fun_vector_hessian21(x_vec, y_vec))

        # Get the equivalent of mixed free and vector Hessians by defining
        # x and y as unconstrained.
        model = TwoParamModel(x_constrained=False)
        model.par['x'].set_vector(x_vec)
        model.par['y'].set_vector(y_vec)
        # Check that the test is valid.
        np_test.assert_array_almost_equal(
            model.par['x'].get_vector(), model.par['x'].get_free())
        objective_full = obj_lib.Objective(model.par, model.fun)
        objective = obj_lib.TwoParameterObjective(
            model.par['x'], model.par['y'], model.fun)
        full_free_hess = objective_full.fun_free_hessian(
            model.par.get_free())
        np_test.assert_array_almost_equal(
            full_free_hess[ind_12],
            objective.fun_hessian_vector1_free2(x_vec, y_free))

        model = TwoParamModel(y_constrained=False)
        model.par['x'].set_vector(x_vec)
        model.par['y'].set_vector(y_vec)
        # Check that the test is valid.
        np_test.assert_array_almost_equal(
            model.par['y'].get_vector(), model.par['y'].get_free())
        objective_full = obj_lib.Objective(model.par, model.fun)
        objective = obj_lib.TwoParameterObjective(
            model.par['x'], model.par['y'], model.fun)
        full_free_hess = objective_full.fun_free_hessian(
            model.par.get_free())
        np_test.assert_array_almost_equal(
            full_free_hess[ind_12],
            objective.fun_hessian_free1_vector2(x_free, y_vec))


    def run_optimization_tests(self, use_sparse=False):
        model = Model(dim=3)
        objective = obj_lib.Objective(par=model.x, fun=model.f)
        preconditioner = 2.0 * np.eye(model.dim)
        preconditioner[model.dim - 1, 0] = 0.1 # Add asymmetry for testing!

        if use_sparse:
            objective.preconditioner = preconditioner
        else:
            objective.preconditioner = sp.sparse.csr_matrix(preconditioner)

        model.set_inits()
        x0 = model.x.get_free()
        y0 = np.linalg.solve(preconditioner, x0)

        # Unconditioned
        opt_result = sp.optimize.minimize(
            fun=objective.fun_free,
            jac=objective.fun_free_grad,
            hessp=objective.fun_free_hvp,
            x0=x0,
            method='trust-ncg',
            options={'maxiter': 100, 'disp': False, 'gtol': 1e-6 })
        self.assertTrue(opt_result.success)
        model.x.set_free(opt_result.x)
        np_test.assert_array_almost_equal(
            model.opt_x, model.x.get_vector(),
            err_msg='Trust-NCG Unconditioned')

        # Conditioned:
        opt_result = sp.optimize.minimize(
            fun=objective.fun_free_cond,
            jac=objective.fun_free_grad_cond,
            hessp=objective.fun_free_hvp_cond,
            x0=y0,
            method='trust-ncg',
            options={'maxiter': 100, 'disp': False, 'gtol': 1e-6 })
        self.assertTrue(opt_result.success)
        model.x.set_free(objective.uncondition_x(opt_result.x))
        np_test.assert_array_almost_equal(
            model.opt_x, model.x.get_vector(),
            err_msg='Trust-NCG')

        opt_result = sp.optimize.minimize(
            fun=lambda par: objective.fun_free_cond(par, verbose=False),
            jac=objective.fun_free_grad_cond,
            x0=y0,
            method='BFGS',
            options={'maxiter': 100, 'disp': False, 'gtol': 1e-6 })
        self.assertTrue(opt_result.success)
        model.x.set_free(objective.uncondition_x(opt_result.x))
        np_test.assert_array_almost_equal(
            model.opt_x, model.x.get_vector(), err_msg='BFGS')

        opt_result = sp.optimize.minimize(
            fun=lambda par: objective.fun_free_cond(par, verbose=False),
            jac=objective.fun_free_grad_cond,
            hess=objective.fun_free_hessian_cond,
            x0=y0,
            method='Newton-CG',
            options={'maxiter': 100, 'disp': False })
        self.assertTrue(opt_result.success)
        model.x.set_free(objective.uncondition_x(opt_result.x))
        np_test.assert_array_almost_equal(
            model.opt_x, model.x.get_vector(), err_msg='Newton')

    def test_optimization(self):
        self.run_optimization_tests(use_sparse=True)
        self.run_optimization_tests(use_sparse=False)



class TestSparsePacking(unittest.TestCase):
    def test_safe_matmul(self):
        def safe_matmul_todense(a, b):
            result = obj_lib.safe_matmul(a, b)
            if sp.sparse.issparse(result):
                return np.asarray(result.todense())
            else:
                return np.asarray(result)

        a = np.random.random((3, 3))
        b = np.random.random((3, 3))
        ab = np.matmul(a, b)

        np_test.assert_array_almost_equal(
            ab, safe_matmul_todense(a, b))
        np_test.assert_array_almost_equal(
            ab, safe_matmul_todense(sp.sparse.csr_matrix(a), b))
        np_test.assert_array_almost_equal(
            ab, safe_matmul_todense(a, sp.sparse.csr_matrix(b)))
        np_test.assert_array_almost_equal(
            ab, safe_matmul_todense(np.matrix(a), b))
        np_test.assert_array_almost_equal(
            ab, safe_matmul_todense(a, np.matrix(b)))

    def test_packing(self):
        dense_mat = np.zeros((3, 3))
        dense_mat[0, 0] = 2.0
        dense_mat[0, 1] = 3.0
        dense_mat[2, 1] = 4.0

        sparse_mat = sp.sparse.csr_matrix(dense_mat)
        sparse_mat_packed = obj_lib.pack_csr_matrix(sparse_mat)
        sparse_mat_unpacked = obj_lib.unpack_csr_matrix(sparse_mat_packed)

        np_test.assert_array_almost_equal(
            dense_mat, sparse_mat_unpacked.todense())


class TestIndexParams(unittest.TestCase):
    def test_index_params(self):
        dim = 3
        param = vb.ModelParamsDict('test')
        param.push_param(vb.VectorParam('x', size=dim, lb=-2.0, ub=5.0))
        param.push_param(vb.VectorParam('y', size=dim, lb=-2.0, ub=5.0))

        index_par = obj_lib.make_index_param(param)
        param.set_free(np.random.random(param.free_size()))
        param_vec = param.get_vector()

        for d in range(dim):
            for pname in ['x', 'y']:
                self.assertAlmostEqual(
                    param[pname].get()[d],
                    param_vec[index_par[pname].get()[d]])


class TestConjugateGradient(unittest.TestCase):
    def test_masking(self):
        masks = cg.get_masks(20, 3)
        self.assertTrue(np.max([ np.sum(m) for m in masks]) <= 3)
        all_m = np.full(20, False)
        no_m = np.full(20, True)
        for m in masks:
            all_m = np.logical_or(all_m, m)
            no_m = np.logical_xor(no_m, m)
        self.assertTrue(np.all(all_m))
        self.assertTrue(~np.any(no_m))

    def test_cg(self):
        K = 50
        mat = np.random.random((K, K))
        mat = 0.5 * (mat + mat.transpose()) + 10 * np.eye(K)
        loc = np.array([float(k) / 7. for k in range(K)])
        def ObjPar(par, mat, loc):
            diff = par - loc
            return np.dot(diff, np.matmul(mat, diff))

        x = loc + 0.1 * np.random.rand(K)

        ObjPar(x, mat, loc)
        Obj = lambda x: ObjPar(x, mat, loc)
        ObjHess = autograd.hessian(Obj)
        ObjHessVecProd = autograd.hessian_vector_product(Obj)

        masks = cg.get_masks(K, 10)
        cg_solver = cg.ConjugateGradientSolver(ObjHessVecProd, loc)
        cg_solver.get_hinv_vec_subsets(x, masks, verbose=False)

        hess = ObjHess(loc)
        cho_factor = sp.linalg.cho_factor(hess)

        chol_hinv_vecs = []
        for vec in cg_solver.vecs:
            hinv_vec = sp.linalg.cho_solve(cho_factor, vec)
            chol_hinv_vecs.append(hinv_vec)

        diffs = [ np.max(np.abs(cg_solver.hinv_vecs[ind] -
                         chol_hinv_vecs[ind])) for ind in range(len(masks))]
        self.assertTrue(np.max(diffs) < 1e-8)


if __name__ == '__main__':
    unittest.main()
