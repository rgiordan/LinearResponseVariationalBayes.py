import Parameters
import unittest
from itertools import product
import numpy as np
import numpy.testing as np_test

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


if __name__ == '__main__':
    unittest.main()
