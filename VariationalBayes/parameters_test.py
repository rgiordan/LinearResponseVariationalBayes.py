
import autograd.numpy as np
from autograd.util import quick_grad_check
import copy
from itertools import product
import numpy.testing as np_test
import Parameters
from Parameters import \
    VectorParam, ScalarParam, PosDefMatrixParam, ModelParamsDict
import unittest

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

        # Just make sure these run without error.
        vp.names()
        str(vp)

    def test_ScalarParam(self):
        lb = -0.1
        ub = 5.2
        val = 0.5 * (ub - lb) + lb
        vp = ScalarParam('test', lb=lb - 0.1, ub=ub + 0.1)

        # Check setting.

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

        # Just make sure these run without error.
        vp.names()
        str(vp)

    def test_LDMatrix_helpers(self):
        mat = np.full(4, 0.2).reshape(2, 2) + np.eye(2)
        mat_chol = np.linalg.cholesky(mat)
        vec = Parameters.VectorizeLDMatrix(mat_chol)
        np_test.assert_array_almost_equal(
            mat_chol, Parameters.UnvectorizeLDMatrix(vec))

        mat_vec = Parameters.pack_posdef_matrix(mat)
        np_test.assert_array_almost_equal(
            mat, Parameters.unpack_posdef_matrix(mat_vec))

    def test_LDMatrixParam(self):
        k = 2
        mat = np.full(k ** 2, 0.2).reshape(k, k) + np.eye(k)

        # not symmetric
        bad_mat = copy.deepcopy(mat)
        bad_mat[1, 0] += + 1

        vp = PosDefMatrixParam('test', k)

        # Check setting.
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

        # Just make sure these run without error.
        vp.names()
        str(vp)

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

        # Just check that these run.
        mp.names()
        str(mp)


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
        # each parameter.
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


if __name__ == '__main__':
    unittest.main()
