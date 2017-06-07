
import autograd.numpy as np
from autograd import grad, jacobian, hessian
from autograd.util import quick_grad_check
import copy
from itertools import product
import numpy.testing as np_test
from VariationalBayes import Parameters
from Parameters import \
    VectorParam, ScalarParam, PosDefMatrixParam, ModelParamsDict
from VariationalBayes.NormalParams import MVNParam, UVNParam, UVNParamVector
from VariationalBayes.GammaParams import GammaParam
from VariationalBayes.ExponentialFamilies import \
    UnivariateNormalEntropy, MultivariateNormalEntropy, GammaEntropy
import unittest
import scipy as sp

# Lower and upper bounds for unit tests.
lbs = [ 0., -2., 1.2, -float("inf")]
ubs = [ 0., -1., 2.1, float("inf")]

class TestParameters(unittest.TestCase):
    def test_scalar(self):
        for lb, ub in product(lbs, ubs):
            if ub > lb:
                val = lb + 0.2
                free_val = Parameters.unconstrain(val, lb, ub)
                new_val = Parameters.constrain(free_val, lb, ub)
                self.assertAlmostEqual(new_val, val)

    def test_vector(self):
        for lb, ub in product(lbs, ubs):
            if ub > lb:
                val = np.array([ lb + 0.1, lb + 0.2 ])
                free_val = Parameters.unconstrain(val, lb, ub)
                new_val = Parameters.constrain(free_val, lb, ub)
                np_test.assert_array_almost_equal(new_val, val)

    def test_VectorParam(self):
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
        self.assertRaises(ValueError, vp.set, bad_val_ub)
        self.assertRaises(ValueError, vp.set, bad_val_lb)
        vp.set(val)

        # Check size.
        self.assertEqual(k, vp.size())
        self.assertEqual(k, vp.free_size())

        # Check getting and free parameters.
        np_test.assert_array_almost_equal(val, vp.get())
        val_free = vp.get_free()
        vp.set(np.full(k, 0.))
        vp.set_free(val_free)
        np_test.assert_array_almost_equal(val, vp.get())

        val_vec = vp.get_vector()
        vp.set(np.full(k, 0.))
        vp.set_vector(val_vec)
        np_test.assert_array_almost_equal(val, vp.get())

        # Just make sure these run without error.
        vp.names()
        str(vp)
        vp.dictval()

    def test_ScalarParam(self):
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
        self.assertRaises(ValueError, vp.set, lb - abs(val))
        self.assertRaises(ValueError, vp.set, ub + abs(val))
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

        # Just make sure these run without error.
        vp.names()
        str(vp)
        vp.dictval()

    def test_LDMatrix_helpers(self):
        mat = np.full(4, 0.2).reshape(2, 2) + np.eye(2)
        mat_chol = np.linalg.cholesky(mat)
        vec = Parameters.VectorizeLDMatrix(mat_chol)
        np_test.assert_array_almost_equal(
            mat_chol, Parameters.UnvectorizeLDMatrix(vec))

        mat_vec = Parameters.pack_posdef_matrix(mat)
        np_test.assert_array_almost_equal(
            mat, Parameters.unpack_posdef_matrix(mat_vec))

    def test_PosDefMatrixParam(self):
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

        # Just make sure these run without error.
        vp.names()
        str(vp)
        vp.dictval()

    def test_ModelParamsDict(self):
        k = 2
        mat = np.full(k ** 2, 0.2).reshape(k, k) + np.eye(k)

        lb = -0.1
        ub = 5.2
        val = 0.5 * (ub - lb) + lb
        vec = np.linspace(lb, ub, k)

        vp_scalar = ScalarParam('scalar', lb=lb - 0.1, ub=ub + 0.1)
        vp_mat = PosDefMatrixParam('matrix', k)
        vp_vec = VectorParam('vector', k, lb=lb - 0.1, ub=ub + 0.1)

        mp = ModelParamsDict()
        mp.push_param(vp_scalar)
        mp.push_param(vp_vec)
        mp.push_param(vp_mat)

        mp['scalar'].set(val)
        mp['vector'].set(vec)
        mp['matrix'].set(mat)
        self.assertAlmostEqual(val, mp['scalar'].get())
        np_test.assert_array_almost_equal(vec, mp['vector'].get())
        np_test.assert_array_almost_equal(mat, mp['matrix'].get())

        free_vec = mp.get_free()
        mp['scalar'].set(0.)
        mp['vector'].set(np.full(k, 0.))
        mp['matrix'].set(np.full((k, k), 0.))
        mp.set_free(free_vec)
        self.assertAlmostEqual(val, mp['scalar'].get())
        np_test.assert_array_almost_equal(vec, mp['vector'].get())
        np_test.assert_array_almost_equal(mat, mp['matrix'].get())
        self.assertEqual(len(free_vec), mp.free_size())

        param_vec = mp.get_vector()
        mp['scalar'].set(0.)
        mp['vector'].set(np.full(k, 0.))
        mp['matrix'].set(np.full((k, k), 0.))
        mp.set_vector(param_vec)
        self.assertAlmostEqual(val, mp['scalar'].get())
        np_test.assert_array_almost_equal(vec, mp['vector'].get())
        np_test.assert_array_almost_equal(mat, mp['matrix'].get())
        self.assertEqual(len(param_vec), mp.vector_size())

        # Just check that these run.
        mp.names()
        str(mp)
        mp.dictval()


    def test_MVNParam(self):
        k = 2
        vec = np.full(2, 0.2)
        mat = np.full(k ** 2, 0.2).reshape(k, k) + np.eye(k)

        vp = MVNParam('test', k)

        # Check setting.
        self.assertRaises(ValueError, vp.mean.set, vec[-1])
        self.assertRaises(ValueError, vp.info.set, np.eye(k + 1))
        vp.mean.set(vec)
        vp.info.set(mat)

        # Check size.
        free_par = vp.get_free()
        self.assertEqual(len(free_par), vp.free_size())
        self.assertEqual(k, vp.dim())

        # Check getting and free parameters.
        vp.mean.set(np.full(k, 0.))
        vp.info.set(np.full((k, k), 0.))
        vp.set_free(free_par)
        np_test.assert_array_almost_equal(mat, vp.info.get())
        np_test.assert_array_almost_equal(vec, vp.mean.get())

        # Just make sure these run without error.
        vp.names()
        str(vp)
        vp.dictval()

    def test_UVNParam(self):
        vp_mean = 0.2
        vp_info = 1.2

        vp = UVNParam('test', min_info=0.1)

        # Check setting.
        vp.mean.set(vp_mean)
        vp.info.set(vp_info)

        # Check size.
        free_par = vp.get_free()
        self.assertEqual(len(free_par), vp.free_size())
        vec_par = vp.get_vector()
        self.assertEqual(len(vec_par), vp.vector_size())

        # Check getting and free parameters.
        vp.mean.set(0.)
        vp.info.set(1.0)
        vp.set_free(free_par)
        self.assertAlmostEqual(vp_mean, vp.mean.get())
        self.assertAlmostEqual(vp_info, vp.info.get())

        # Check getting and free parameters.
        vp.mean.set(0.)
        vp.info.set(1.0)
        vp.set_vector(vec_par)
        self.assertAlmostEqual(vp_mean, vp.mean.get())
        self.assertAlmostEqual(vp_info, vp.info.get())

        # Just make sure these run without error.
        vp.names()
        str(vp)
        vp.dictval()

    def test_UVNParamVector(self):
        k = 2
        vp_mean = np.array([ 0.2, 0.5 ])
        vp_info = np.array([ 1.2, 2.1 ])

        vp = UVNParamVector('test', k, min_info=0.1)

        # Check setting.
        vp.mean.set(vp_mean)
        vp.info.set(vp_info)

        # Check size.
        free_par = vp.get_free()
        self.assertEqual(len(free_par), vp.free_size())

        # Check getting and free parameters.
        vp.mean.set(np.full(k, 0.))
        vp.info.set(np.full(k, 1.))
        vp.set_free(free_par)
        np_test.assert_array_almost_equal(vp_mean, vp.mean.get())
        np_test.assert_array_almost_equal(vp_info, vp.info.get())

        # Just make sure these run without error.
        vp.names()
        str(vp)
        vp.dictval()

    def test_GammaParam(self):
        shape = 0.2
        rate = 0.4

        vp = GammaParam('test', min_rate=0.1)

        # Check setting.
        vp.shape.set(shape)
        vp.rate.set(rate)

        # Check size.
        free_par = vp.get_free()
        self.assertEqual(len(free_par), vp.free_size())

        # Check getting and free parameters.
        vp.shape.set(1.0)
        vp.rate.set(1.0)
        vp.set_free(free_par)
        self.assertAlmostEqual(shape, vp.shape.get())
        self.assertAlmostEqual(rate, vp.rate.get())

        # Just make sure these run without error.
        vp.names()
        str(vp)
        vp.dictval()


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

        mp = ModelParamsDict()
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

        mp = ModelParamsDict()
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


class TestEntropy(unittest.TestCase):
    def test_uvn_entropy(self):
        mean_par = 2.0
        info_par = 1.5
        num_draws = 10000
        norm_dist = sp.stats.norm(loc=mean_par, scale=np.sqrt(1 / info_par))
        self.assertAlmostEqual(norm_dist.entropy(), UnivariateNormalEntropy(info_par))

    def test_mvn_entropy(self):
        mean_par = np.array([1., 2.])
        info_par = np.eye(2) + np.full((2, 2), 0.1)
        norm_dist = sp.stats.multivariate_normal(
            mean=mean_par, cov=np.linalg.inv(info_par))
        self.assertAlmostEqual(
            norm_dist.entropy(), MultivariateNormalEntropy(info_par))

    def test_gamma_entropy(self):
        shape = 3.0
        rate = 2.4
        gamma_dist = sp.stats.gamma(a=shape, scale=1 / rate)
        self.assertAlmostEqual(gamma_dist.entropy(), GammaEntropy(shape, rate))


if __name__ == '__main__':
    unittest.main()
