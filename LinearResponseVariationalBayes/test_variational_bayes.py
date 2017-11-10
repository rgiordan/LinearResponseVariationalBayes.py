#!/usr/bin/env python3

import autograd.numpy as np
from autograd import grad, jacobian, hessian
from autograd.util import quick_grad_check
import copy
from itertools import product
import numpy.testing as np_test
from LinearResponseVariationalBayes import Parameters
from LinearResponseVariationalBayes import MatrixParameters
from LinearResponseVariationalBayes import SimplexParams
from LinearResponseVariationalBayes.Parameters import \
    ScalarParam, VectorParam, ArrayParam
from LinearResponseVariationalBayes.MatrixParameters import \
    PosDefMatrixParam, PosDefMatrixParamVector
from LinearResponseVariationalBayes import ParameterDictionary as par_dict
from LinearResponseVariationalBayes.NormalParams import MVNParam, UVNParam, UVNParamVector, \
                            MVNArray, UVNParamArray, UVNMomentParamArray
from LinearResponseVariationalBayes.GammaParams import GammaParam
from LinearResponseVariationalBayes.WishartParams import WishartParam
from LinearResponseVariationalBayes.SimplexParams import SimplexParam
from LinearResponseVariationalBayes.SimplexParams import \
    constrain_simplex_vector, constrain_hess_from_moment, \
    constrain_grad_from_moment
from LinearResponseVariationalBayes.DirichletParams import DirichletParamArray

from LinearResponseVariationalBayes.ProjectionParams import \
    SubspaceVectorParam, get_perpendicular_subspace

import unittest
import scipy as sp


# Lower and upper bounds for unit tests.
lbs = [ 0., -2., 1.2, -float("inf")]
ubs = [ 0., -1., 2.1, float("inf")]

def check_sparse_transforms(testcase, param):

    free_param = param.get_free()
    def set_free_and_get_vector(free_param):
        param.set_free(free_param)
        return param.get_vector()

    set_free_and_get_vector_jac = jacobian(set_free_and_get_vector)
    set_free_and_get_vector_hess = hessian(set_free_and_get_vector)

    jac = set_free_and_get_vector_jac(free_param)
    np_test.assert_array_almost_equal(
        jac, param.free_to_vector_jac(free_param).toarray())

    hess = set_free_and_get_vector_hess(free_param)
    sp_hess = param.free_to_vector_hess(free_param)
    testcase.assertEqual(len(sp_hess), hess.shape[0])
    for vec_row in range(len(sp_hess)):
        np_test.assert_array_almost_equal(
            hess[vec_row, :, :], sp_hess[vec_row].toarray())


def execute_required_methods(
    testcase, param, test_autograd=False, test_sparse_transform=True):
    # Execute all the methods requied for a parameter type.

    param.names()
    param.dictval()

    free_param = param.get_free()
    param.set_free(free_param)
    testcase.assertEqual(1, free_param.ndim)

    vec_param = param.get_vector()
    param.set_vector(vec_param)
    testcase.assertEqual(1, vec_param.ndim)
    str(param)

    testcase.assertEqual(param.free_size(), len(free_param))
    testcase.assertEqual(param.vector_size(), len(vec_param))

    if test_autograd:
        def set_free_and_get(free_param):
            param.set_free(free_param)
            return param.get()

        param_value_jacobian = jacobian(set_free_and_get)
        jac = param_value_jacobian(free_param)

    if test_sparse_transform:
        check_sparse_transforms(testcase, param)


class TestParameterMethods(unittest.TestCase):
    # For every parameter type, execute all the required methods.
    def test_scalar(self):
        execute_required_methods(self, ScalarParam(lb=1.0),
            test_autograd=True, test_sparse_transform=True)
    def test_vector(self):
        execute_required_methods(self, VectorParam(lb=1.0),
            test_autograd=True, test_sparse_transform=True)
    def test_array(self):
        execute_required_methods(self, ArrayParam(shape=(2, 3, 2), lb=1.0),
            test_autograd=True, test_sparse_transform=True)
    def test_pos_def_matrix(self):
        single_mat = np.diag([ 1.0, 2.0 ]) + np.full((2, 2), 0.1)
        execute_required_methods(self, PosDefMatrixParam(val=single_mat),
                test_autograd=True, test_sparse_transform=True)
    def test_pos_def_matrix_vector(self):
        single_mat = np.diag([ 1.0, 2.0 ]) + np.full((2, 2), 0.1)
        single_mat = np.expand_dims(single_mat, 0)
        mat = np.tile(single_mat, (2, 1, 1))
        execute_required_methods(self,
            PosDefMatrixParamVector(
                val=mat, length=mat.shape[0], matrix_size=2),
            test_autograd=True, test_sparse_transform=True)
    def test_simplex(self):
        execute_required_methods(self, SimplexParam(shape=(5, 3)),
            test_autograd=True, test_sparse_transform=True)
    def test_projection_params(self):
        execute_required_methods(self, SubspaceVectorParam(),
            test_autograd=True, test_sparse_transform=True)

    def test_mvn(self):
        execute_required_methods(self, MVNParam(), test_sparse_transform=True)
        par = MVNParam()
        par.e()
        par.cov()
        par.e_outer()
        par.entropy()

    def test_uvn(self):
        execute_required_methods(self, UVNParam(), test_sparse_transform=True)
        par = UVNParam()
        par.e()
        par.var()
        par.e_outer()
        par.e_exp()
        par.var_exp()
        par.e2_exp()
        par.entropy()

    def test_uvn_vec(self):
        execute_required_methods(self, UVNParamVector(),
                                 test_sparse_transform=True)
        par = UVNParamVector()
        par.e()
        par.var()
        par.e_outer()
        par.e_exp()
        par.var_exp()
        par.e2_exp()
        par.entropy()
        par.size()


    def test_gamma(self):
        execute_required_methods(self, GammaParam(), test_sparse_transform=True)
        par = GammaParam()
        par.e()
        par.e_log()
        par.entropy()

    def test_dirichlet(self):
        execute_required_methods(self, DirichletParamArray(),
                                 test_sparse_transform=True)
        par = DirichletParamArray()
        par.e()
        par.e_log()
        par.entropy()

    def test_uvn_array(self):
        execute_required_methods(self, UVNParamArray(),
                                 test_sparse_transform=True)
        test_shape = (3, 2, 1)
        par = UVNParamArray(shape=test_shape)
        par.e()
        par.var()
        par.e_outer()
        par.e_exp()
        par.var_exp()
        par.e2_exp()
        par.entropy()
        par.shape()

    def test_uvn_moment_array(self):
        execute_required_methods(self, UVNMomentParamArray(),
                                 test_sparse_transform=True)
        test_shape = (3, 2, 1)
        par = UVNMomentParamArray(shape=test_shape)
        par.e()
        par.var()
        par.e_outer()
        par.e_exp()
        par.var_exp()
        par.e2_exp()
        par.entropy()
        par.shape()

        uvn_par = UVNParamArray(shape=test_shape)
        uvn_par['mean'].set(np.random.random(test_shape))
        uvn_par['info'].set(np.exp(np.random.random(test_shape)))
        par.set_from_uvn_param_array(uvn_par)
        np_test.assert_array_almost_equal(uvn_par.e(), par.e())
        np_test.assert_array_almost_equal(uvn_par.e_outer(), par.e_outer())

        array_par = ArrayParam(shape=test_shape)
        array_par.set(np.random.random(test_shape))
        par.set_from_constant(array_par)
        np_test.assert_array_almost_equal(array_par.get(), par.e())
        np_test.assert_array_almost_equal(array_par.get() ** 2, par.e_outer())

    def test_wishart(self):
        execute_required_methods(self, WishartParam(),
                                 test_sparse_transform=True)
        par = WishartParam()
        par.e()
        par.e_log_det()
        par.e_inv()
        par.entropy()
        par.e_log_lkj_inv_prior(5.0)


class TestConstrainingFunctions(unittest.TestCase):

    def test_scalar(self):
        for lb, ub in product(lbs, ubs):
            if ub > lb:
                val = lb + 0.2
                free_val = Parameters.unconstrain_scalar(val, lb, ub)
                new_val = Parameters.constrain(free_val, lb, ub)
                self.assertAlmostEqual(new_val, val)

    def test_array(self):
        for lb, ub in product(lbs, ubs):
            if ub > lb:
                val = np.array([ lb + 0.1, lb + 0.2 ])
                free_val = Parameters.unconstrain_array(val, lb, ub)
                new_val = Parameters.constrain(free_val, lb, ub)
                np_test.assert_array_almost_equal(new_val, val)

    def test_simplex_mat(self):
        nrow = 5
        ncol = 4
        free_mat = np.random.random((nrow, ncol - 1)) * 2 - 1
        simplex_mat = SimplexParams.constrain_simplex_matrix(free_mat)
        self.assertEqual(simplex_mat.shape, (nrow, ncol))
        np_test.assert_array_almost_equal(
            np.full(nrow, 1.0), np.sum(simplex_mat, 1))
        free_mat2 = SimplexParams.unconstrain_simplex_matrix(simplex_mat)
        self.assertEqual(free_mat2.shape, (nrow, ncol - 1))
        np_test.assert_array_almost_equal(free_mat, free_mat2)

    def test_get_perpendicular_subspace(self):
        dim = 5
        const_dim = 2
        # We will project perpendicular to the rows of x.
        x = np.random.random((const_dim, dim))
        basis = get_perpendicular_subspace(x)
        np_test.assert_array_almost_equal(
            np.zeros((const_dim, dim - const_dim)), np.matmul(x, basis))


class TestParameters(unittest.TestCase):
    def test_vector_param(self):
        lb = -0.1
        ub = 5.2
        k = 4
        val = np.linspace(lb, ub, k)
        bad_val_ub = np.abs(val) + ub
        bad_val_lb = lb - np.abs(val)
        vp = VectorParam('test', k, lb=lb - 0.1, ub=ub + 0.1)

        # Check setting.
        vp_init = VectorParam('test', k, lb=lb - 0.1, ub=ub + 0.1, val=val)
        np_test.assert_array_almost_equal(val, vp_init.get())

        self.assertRaises(ValueError, vp.set, val[-1])
        # Bounds checking is disabled for now until it can be made optional.
        # self.assertRaises(ValueError, vp.set, bad_val_ub)
        # self.assertRaises(ValueError, vp.set, bad_val_lb)
        vp.set(val)

        # Check size.
        self.assertEqual(k, vp.size())
        self.assertEqual(k, vp.free_size())

        # Check getting and free parameters.
        np_test.assert_array_almost_equal(val, vp.get())
        val_free = vp.get_free()
        self.assertEqual(1, len(val_free.shape))
        vp.set(np.full(k, 0.))
        vp.set_free(val_free)
        np_test.assert_array_almost_equal(val, vp.get())

        val_vec = vp.get_vector()
        self.assertEqual(1, len(val_vec.shape))
        vp.set(np.full(k, 0.))
        vp.set_vector(val_vec)
        np_test.assert_array_almost_equal(val, vp.get())


    def test_array_param(self):

        # Check that the default initialization is finite.
        default_init = ArrayParam()
        self.assertTrue(np.isfinite(default_init.get()).all())

        lb = -0.1
        ub = 5.2
        shape = (3, 2)
        val = np.random.random(shape) * (ub - lb) + lb
        bad_val_ub = np.abs(val) + ub
        bad_val_lb = lb - np.abs(val)
        ap = ArrayParam('test', shape, lb=lb - 0.001, ub=ub + 0.001)

        # Check setting.
        ap_init = ArrayParam('test', shape, lb=lb - 0.001, ub=ub + 0.001, val=val)
        np_test.assert_array_almost_equal(val, ap_init.get())

        self.assertRaises(ValueError, ap.set, val[-1, :])

        # Bounds checking is disabled for now until it can be made optional.
        #self.assertRaises(ValueError, ap.set, bad_val_ub)
        #self.assertRaises(ValueError, ap.set, bad_val_lb)
        ap.set(val)

        # Check size.
        self.assertEqual(np.product(shape), ap.vector_size())
        self.assertEqual(np.product(shape), ap.free_size())

        # Check getting and free parameters.
        np_test.assert_array_almost_equal(val, ap.get())
        val_free = ap.get_free()
        ap.set(np.full(shape, 0.))
        ap.set_free(val_free)
        np_test.assert_array_almost_equal(val, ap.get())

        val_vec = ap.get_vector()
        ap.set(np.full(shape, 0.))
        ap.set_vector(val_vec)
        np_test.assert_array_almost_equal(val, ap.get())


    def test_scalar_param(self):
        lb = -0.1
        ub = 5.2
        val = 0.5 * (ub - lb) + lb
        vp = ScalarParam('test', lb=lb - 0.1, ub=ub + 0.1)

        # Check setting.
        vp_init = ScalarParam('test', lb=lb - 0.1, ub=ub + 0.1, val=4.0)
        self.assertEqual(vp_init.get(), 4.0)

        # Asserting that you are getting something of length one doesn't
        # seem trivial in python.
        # self.assertRaises(ValueError, vp.set, np.array([val, val]))
        # Bounds checking is disabled for now until it can be made optional.
        # self.assertRaises(ValueError, vp.set, lb - abs(val))
        # self.assertRaises(ValueError, vp.set, ub + abs(val))
        vp.set(val)
        vp.set(np.array([val]))

        # Check size.
        self.assertEqual(1, vp.free_size())

        # Check getting and free parameters.
        self.assertAlmostEqual(val, vp.get())
        val_free = vp.get_free()
        vp.set(0.)
        vp.set_free(val_free)
        self.assertAlmostEqual(val, vp.get())

        val_vec = vp.get_vector()
        vp.set(0.)
        vp.set_vector(val_vec)
        self.assertAlmostEqual(val, vp.get())


    def test_simplex_param(self):
        shape = (5, 3)

        def random_simplex(shape):
            val = np.random.random(shape)
            val = val / np.expand_dims(np.sum(val, 1), axis=1)
            return val

        val = random_simplex(shape)
        bad_val = random_simplex((shape[0], shape[1] + 1))
        sp = SimplexParam(name='test', shape=shape, val=val)
        np_test.assert_array_almost_equal(val, sp.get())

        self.assertRaises(ValueError, sp.set, bad_val)
        free_val = sp.get_free()
        vec_val = sp.get_vector()

        # Check size.
        self.assertEqual(len(vec_val), sp.vector_size())
        self.assertEqual(len(free_val), sp.free_size())

        # # Check getting and free parameters.
        unif_simplex = np.full(shape, 1. / shape[1])
        sp.set(unif_simplex)

        sp.set(val)
        np_test.assert_array_almost_equal(val, sp.get())

        sp.set(unif_simplex)
        sp.set_free(free_val)
        np_test.assert_array_almost_equal(val, sp.get())

        sp.set(unif_simplex)
        sp.set_vector(vec_val)
        np_test.assert_array_almost_equal(val, sp.get())

        # Check get_vector_indices
        distinct_free_val = np.arange(0., 1., 1. / sp.free_size())
        sp.set_free(distinct_free_val)
        vec_val = sp.get_vector()
        val = sp.get()
        for row in range(sp.shape()[0]):
            row_inds = sp.get_vector_indices(row)
            np_test.assert_array_almost_equal(vec_val[row_inds], val[row, :])

    def test_LDMatrix_helpers(self):
        mat = np.full(4, 0.2).reshape(2, 2) + np.eye(2)
        mat_chol = np.linalg.cholesky(mat)
        vec = MatrixParameters.vectorize_ld_matrix(mat_chol)
        np_test.assert_array_almost_equal(
            mat_chol, MatrixParameters.unvectorize_ld_matrix(vec))

        mat_vec = MatrixParameters.pack_posdef_matrix(mat)
        np_test.assert_array_almost_equal(
            mat, MatrixParameters.unpack_posdef_matrix(mat_vec))

    def test_pos_def_matrix_param(self):
        k = 2
        mat = np.full(k ** 2, 0.2).reshape(k, k) + np.eye(k)

        # not symmetric
        bad_mat = copy.deepcopy(mat)
        bad_mat[1, 0] += + 1

        vp = PosDefMatrixParam('test', k)

        # Check setting.
        vp_init = PosDefMatrixParam('test', k, val=mat)
        np_test.assert_array_almost_equal(mat, vp_init.get())

        self.assertRaises(ValueError, vp.set, bad_mat)
        self.assertRaises(ValueError, vp.set, np.eye(k + 1))
        vp.set(mat)

        # Check size.
        self.assertEqual(k, vp.size())
        self.assertEqual(k * (k + 1) / 2, vp.free_size())

        # Check getting and free parameters.
        np_test.assert_array_almost_equal(mat, vp.get())
        mat_free = vp.get_free()
        vp.set(np.full((k, k), 0.))
        vp.set_free(mat_free)
        np_test.assert_array_almost_equal(mat, vp.get())

        mat_vectorized = vp.get_vector()
        vp.set(np.full((k, k), 0.))
        vp.set_vector(mat_vectorized)
        np_test.assert_array_almost_equal(mat, vp.get())


class TestParameterDictionary(unittest.TestCase):
    def test_model_params_dict(self):
        k = 2
        mat = np.full(k ** 2, 0.2).reshape(k, k) + np.eye(k)

        def clear_model_params(mp):
            mp['scalar'].set(0.)
            mp['vector'].set(np.full(k, 0.))
            mp['matrix'].set(np.full((k, k), 0.))
            mp['simplex'].set(np.full(simp.shape, 1. / np.prod(simp.shape)))

        lb = -0.1
        ub = 5.2
        val = 0.5 * (ub - lb) + lb
        vec = np.linspace(lb, ub, k)
        simp = np.linspace(2., 10., k * 2).reshape(k, 2)
        simp = simp / np.expand_dims(np.sum(simp, 1), axis=1)
        vp_scalar = ScalarParam('scalar', lb=lb - 0.1, ub=ub + 0.1)
        vp_mat = PosDefMatrixParam('matrix', k)
        vp_vec = VectorParam('vector', k, lb=lb - 0.1, ub=ub + 0.1)
        vp_simp = SimplexParam('simplex', shape=simp.shape)

        mp = par_dict.ModelParamsDict(name='ModelParamsDict')
        mp.push_param(vp_scalar)
        mp.push_param(vp_vec)
        mp.push_param(vp_mat)
        mp.push_param(vp_simp)

        execute_required_methods(self, mp, test_autograd=False)

        param_names = ['scalar', 'vector', 'matrix', 'simplex']
        param_vals = \
            { 'scalar': val, 'vector': vec, 'matrix': mat, 'simplex': simp }

        for param in param_names:
            mp[param].set(param_vals[param])
        for param in param_names:
            np_test.assert_array_almost_equal(
                param_vals[param], mp[param].get())

        free_vec = mp.get_free()
        clear_model_params(mp)
        mp.set_free(free_vec)
        for param in param_names:
            np_test.assert_array_almost_equal(
                param_vals[param], mp[param].get())
        self.assertEqual(len(free_vec), mp.free_size())

        param_vec = mp.get_vector()
        clear_model_params(mp)
        mp.set_vector(param_vec)
        for param in param_names:
            np_test.assert_array_almost_equal(
                param_vals[param], mp[param].get())
        self.assertEqual(len(param_vec), mp.vector_size())

        # Check the index dictionaries.
        free_vec = mp.get_free()
        mp.set_free(free_vec)
        for param_id in range(len(param_names)):
            param = param_names[param_id]
            free_vec_peturb = mp.get_free()
            free_vec_peturb[mp.free_indices_dict[param]] += 1.0
            mp.set_free(free_vec_peturb)

            # Check that only the parameter we're looking at has changed.
            self.assertTrue(
                np.max(np.abs(mp[param].get() - param_vals[param]) > 1e-6))
            for unchanged_param in set(param_names) - set([param]):
                np_test.assert_array_almost_equal(
                    param_vals[unchanged_param], mp[unchanged_param].get())
            mp.set_free(free_vec)

        param_vec = mp.get_vector()
        mp.set_vector(param_vec)
        for param_id in range(len(param_names)):
            param = param_names[param_id]
            param_vec_peturb = mp.get_vector()
            param_vec_peturb[mp.vector_indices_dict[param]] += 0.1
            mp.set_vector(param_vec_peturb)

            # Check that only the parameter we're looking at has changed.
            self.assertTrue(
                np.max(np.abs(mp[param].get() - param_vals[param]) > 1e-6))
            for unchanged_param in set(param_names) - set([param]):
                np_test.assert_array_almost_equal(
                    param_vals[unchanged_param], mp[unchanged_param].get())
            mp.set_vector(param_vec)

        # Check the sparse transforms.  Note that the Hessian of this test
        # gives the autograd warning that the output seems to be independent
        # of the input.
        check_sparse_transforms(self, mp)


class TestDifferentiation(unittest.TestCase):
    def test_free_grads(self):
        k = 2
        mat = np.full(k ** 2, 0.2).reshape(k, k) + np.eye(k)

        lb = -0.1
        ub = 5.2
        val = 0.5 * (ub - lb) + lb
        vec = np.linspace(lb, ub, k)

        vp_scalar = ScalarParam('scalar', lb=lb - 0.1, ub=ub + 0.1)
        vp_mat = PosDefMatrixParam('matrix', k)
        vp_vec = VectorParam('vector', k, lb=lb - 0.1, ub=ub + 0.1)

        vp_scalar.set(val)
        vp_vec.set(vec)
        vp_mat.set(mat)

        mp = par_dict.ModelParamsDict()
        mp.push_param(vp_scalar)
        mp.push_param(vp_vec)
        mp.push_param(vp_mat)

        # To take advantage of quick_grad_check(), define scalar functions of
        # each parameter.  It would be nice to have an easy-to-use test for
        # Jacobians.
        def ScalarFun(val_free):
            vp_scalar_ad = copy.deepcopy(vp_scalar)
            vp_scalar_ad.set_free(val_free)
            return vp_scalar_ad.get()

        def VecFun(val_free):
            vp_vec_ad = copy.deepcopy(vp_vec)
            vp_vec_ad.set_free(val_free)
            return np.linalg.norm(vp_vec_ad.get())

        def MatFun(val_free):
            vp_mat_ad = copy.deepcopy(vp_mat)
            vp_mat_ad.set_free(val_free)
            return np.linalg.norm(vp_mat_ad.get())

        def ParamsFun(val_free):
            mp_ad = copy.deepcopy(mp)
            mp_ad.set_free(val_free)
            return mp_ad['scalar'].get() + \
                   np.linalg.norm(mp_ad['vector'].get()) + \
                   np.linalg.norm(mp_ad['matrix'].get())


        quick_grad_check(ScalarFun, vp_scalar.get_free())
        quick_grad_check(VecFun, vp_vec.get_free())
        quick_grad_check(MatFun, vp_mat.get_free())
        quick_grad_check(ParamsFun, mp.get_free())

    def test_vector_grads(self):
        k = 2
        mat = np.full(k ** 2, 0.2).reshape(k, k) + np.eye(k)

        lb = -0.1
        ub = 5.2
        val = 0.5 * (ub - lb) + lb
        vec = np.linspace(lb, ub, k)

        vp_scalar = ScalarParam('scalar', lb=lb - 0.1, ub=ub + 0.1)
        vp_mat = PosDefMatrixParam('matrix', k)
        vp_vec = VectorParam('vector', k, lb=lb - 0.1, ub=ub + 0.1)

        vp_scalar.set(val)
        vp_vec.set(vec)
        vp_mat.set(mat)

        mp = par_dict.ModelParamsDict()
        mp.push_param(vp_scalar)
        mp.push_param(vp_vec)
        mp.push_param(vp_mat)

        # To take advantage of quick_grad_check(), define scalar functions of
        # each parameter.  It would be nice to have an easy-to-use test for
        # Jacobians.
        def ScalarFun(val_vec):
            vp_scalar_ad = copy.deepcopy(vp_scalar)
            vp_scalar_ad.set_vector(val_vec)
            return vp_scalar_ad.get()

        def VecFun(val_vec):
            vp_vec_ad = copy.deepcopy(vp_vec)
            vp_vec_ad.set_vector(val_vec)
            return np.linalg.norm(vp_vec_ad.get())

        def MatFun(val_vec):
            vp_mat_ad = copy.deepcopy(vp_mat)
            vp_mat_ad.set_vector(val_vec)
            return np.linalg.norm(vp_mat_ad.get())

        def ParamsFun(val_vec):
            mp_ad = copy.deepcopy(mp)
            mp_ad.set_vector(val_vec)
            return mp_ad['scalar'].get() + \
                   np.linalg.norm(mp_ad['vector'].get()) + \
                   np.linalg.norm(mp_ad['matrix'].get())

        quick_grad_check(ScalarFun, vp_scalar.get_vector())
        quick_grad_check(VecFun, vp_vec.get_vector())
        quick_grad_check(MatFun, vp_mat.get_vector())
        quick_grad_check(ParamsFun, mp.get_vector())

    def test_LDMatrixParamDerivatives(self):
        # Test the LD matrix extra carefully since we define our own
        # autograd derivatives.
        k = 2
        mat = np.full(k ** 2, 0.2).reshape(k, k) + np.eye(k)
        diag_lb = 0.5
        vp = PosDefMatrixParam('test', k, diag_lb=diag_lb)
        vp.set(mat)
        mat_free = vp.get_free()

        def MatFun(mat_free):
            vp_ad = copy.deepcopy(vp)
            vp_ad.set_free(mat_free)
            return vp_ad.get()

        MatFunJac = jacobian(MatFun)
        MatFunHess = hessian(MatFun)

        # Test the jacobian
        eps = 1e-4
        for ind in range(len(mat_free)):
            mat_free_eps = copy.deepcopy(mat_free)
            mat_free_eps[ind] += eps
            num_grad = MatFun(mat_free_eps) - MatFun(mat_free)
            np_test.assert_array_almost_equal(
                num_grad, eps * MatFunJac(mat_free)[:, :, ind])

        # Test the hessian
        eps = 1e-4
        for ind1 in range(len(mat_free)):
            for ind2 in range(len(mat_free)):

                eps1_vec = np.zeros_like(mat_free)
                eps2_vec = np.zeros_like(mat_free)
                eps1_vec[ind1] = eps
                eps2_vec[ind2] = eps

                num_hess = MatFun(mat_free + eps1_vec + eps2_vec) - \
                           MatFun(mat_free + eps2_vec) - \
                           (MatFun(mat_free + eps1_vec) - MatFun(mat_free))
                np_test.assert_array_almost_equal(
                    num_hess, (eps ** 2) * MatFunHess(mat_free)[:, :, ind1, ind2])

    def test_simplex_derivatives(self):
        k = 3
        free_param = np.arange(0., float(k), 1.)
        z = constrain_simplex_vector(free_param)

        get_constrain_hess = hessian(constrain_simplex_vector)
        target_hess = get_constrain_hess(free_param)
        hess = constrain_hess_from_moment(z)
        np_test.assert_array_almost_equal(target_hess, hess)

        get_constrain_jac = jacobian(constrain_simplex_vector)
        target_jac = get_constrain_jac(free_param)
        jac = constrain_grad_from_moment(z)

        np_test.assert_array_almost_equal(target_jac, jac)

    def test_sparse_free_hessians(self):
        k = 2

        mat = np.full(k ** 2, 0.2).reshape(k, k) + np.eye(k)
        vp_array = ArrayParam('array', shape=(4, 5, 7))
        vp_mat = PosDefMatrixParam('mat', k, val=mat)
        vp_simplex = SimplexParam('simplex', shape=(5, 3))

        mp = par_dict.ModelParamsDict()
        mp.push_param(vp_mat)
        mp.push_param(vp_simplex)
        mp.push_param(vp_array)

        def model(mp):
            mat = mp['mat'].get()
            array = mp['array'].get()
            simplex = mp['simplex'].get()

            return np.sum(mat)**2 * np.sum(array)**2 * np.sum(simplex)**2

        def model_wrap_free(free_param, mp):
            mp.set_free(free_param)
            return model_wrap_vec(mp.get_vector(), mp)

        def model_wrap_vec(vec_param, mp):
            mp.set_vector(vec_param)
            return model(mp)

        free_vec = np.random.random(mp.free_size())
        mp.set_free(free_vec)
        mp_vec = mp.get_vector()

        model_wrap_vec_jac = jacobian(model_wrap_vec)
        model_wrap_free_hess = hessian(model_wrap_free)
        model_wrap_vec_hess = hessian(model_wrap_vec)

        vec_jac_model = model_wrap_vec_jac(mp_vec, mp)
        vec_hess_model = model_wrap_vec_hess(mp_vec, mp)
        free_hess_model = model_wrap_free_hess(free_vec, mp)

        free_hess_sparse = Parameters.convert_vector_to_free_hessian(
            mp, free_vec, vec_jac_model, vec_hess_model)

        np_test.assert_array_almost_equal(free_hess_model, free_hess_sparse)


if __name__ == '__main__':
    unittest.main()
