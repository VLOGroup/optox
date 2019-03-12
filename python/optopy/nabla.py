import numpy as np
import unittest

import _py_ext.py_nabla_operator

float_2d = _py_ext.py_nabla_operator.Nabla_2d_float
double_2d = _py_ext.py_nabla_operator.Nabla_2d_double

float_3d = _py_ext.py_nabla_operator.Nabla_3d_float
double_3d = _py_ext.py_nabla_operator.Nabla_3d_double


# to run execute: python -m unittest [-v] optopy.nabla
class TestNablaFunction(unittest.TestCase):
    
    def _get_nabla_op(self, dtype, dim):
        if dtype == np.float32:
            if dim == 2:
                return float_2d()
            elif dim == 3:
                return float_3d()
            else:
                raise RuntimeError('Invalid number of dimensions!')
        elif dtype == np.float64:
            if dim == 2:
                return double_2d()
            elif dim == 3:
                return double_3d()
            else:
                raise RuntimeError('Invalid number of dimensions!')
        else:
            raise RuntimeError('Invalid dtype!')
            
    def _test_adjointness(self, dtype, dim):
        # get the corresponding operator
        op = self._get_nabla_op(dtype, dim)
        # setup the vaiables
        shape = [10 for i in range(dim)]
        np_x = np.random.randn(*shape).astype(dtype)
        shape.insert(0, dim)
        np_p = np.random.randn(*shape).astype(dtype)

        np_nabla_x = op.forward(np_x)
        np_nablaT_p = op.adjoint(np_p)

        lhs = (np_nabla_x * np_p).sum()
        rhs = (np_x * np_nablaT_p).sum()

        print('dtype: {} dim: {} diff: {}'.format(dtype, dim, np.abs(lhs - rhs)))
        self.assertTrue(np.abs(lhs - rhs) < 1e-3)

    def test_float2_gradient(self):
        self._test_adjointness(np.float32, 2)

    def test_float3_gradient(self):
        self._test_adjointness(np.float32, 3)

    def test_double2_gradient(self):
        self._test_adjointness(np.float64, 2)

    def test_double3_gradient(self):
        self._test_adjointness(np.float64, 3)


if __name__ == "__main__":
    unittest.test()
