import numpy as np
import unittest

import _ext.py_nabla_operator

__all__ = ['float_2d', 'double_2d', 'float_3d', 'double_3d', 'float_4d', 'double_4d']

float_2d = _ext.py_nabla_operator.Nabla_2d_float
double_2d = _ext.py_nabla_operator.Nabla_2d_double

float_3d = _ext.py_nabla_operator.Nabla_3d_float
double_3d = _ext.py_nabla_operator.Nabla_3d_double

float_4d = _ext.py_nabla_operator.Nabla_4d_float
double_4d = _ext.py_nabla_operator.Nabla_4d_double

class Nabla(object):
    def __init__(self, dim, hx=1, hy=1, hz=1, ht=1):
        self.dim = dim
        self.hx = hx
        self.hy = hy
        self.hz = hz
        self.ht = ht
    
    def _get_op(self, dtype, dim, hx, hy, hz=1, ht=1):
        if dtype == np.float32 or dtype == np.complex64:
            if dim == 2:
                return float_2d(hx, hy)
            elif dim == 3:
                return float_3d(hx, hy, hz)
            elif dim == 4:
                return float_4d(hx, hy, hz, ht)
            else:
                raise RuntimeError('Invalid number of dimensions!')
        elif dtype == np.float64 or dtype == np.complex128:
            if dim == 2:
                return double_2d(hx, hy)
            elif dim == 3:
                return double_3d(hx, hy, hz)
            elif dim == 4:
                return double_4d(hx, hy, hz, ht)
            else:
                raise RuntimeError('Invalid number of dimensions!')
        else:
            raise RuntimeError('Invalid dtype!')
        
    def forward(self, x):
        op = self._get_op(x.dtype, self.dim, self.hx, self.hy, self.hz, self.ht)
        
        if np.iscomplexobj(x):
            x_re = np.ascontiguousarray(np.real(x))
            x_im = np.ascontiguousarray(np.imag(x))
            return op.forward(x_re) + 1j * op.forward(x_im)
        else:
            return op.forward(x)

    def adjoint(self, x):
        op = self._get_op(x.dtype, self.dim, self.hx, self.hy, self.hz, self.ht)
        
        if np.iscomplexobj(x):
            x_re = np.ascontiguousarray(np.real(x))
            x_im = np.ascontiguousarray(np.imag(x))
            return op.adjoint(x_re) + 1j * op.adjoint(x_im)
        else:
            return op.adjoint(x)

# to run execute: python -m unittest [-v] optopy.nabla
class TestNablaFunction(unittest.TestCase):           
    def _test_adjointness(self, dtype, dim, hx, hy, hz=1, ht=1):
        assert dim in [2, 3, 4]
        # get the corresponding operator
        op = Nabla(dim, hx, hy, hz)
        # setup the vaiables
        shape = [10 for i in range(dim)]
        np_x = np.random.randn(*shape).astype(dtype)
        shape.insert(0, dim)
        np_p = np.random.randn(*shape).astype(dtype)

        np_nabla_x = op.forward(np_x)
        np_nablaT_p = op.adjoint(np_p)

        lhs = (np_nabla_x * np_p).sum()
        rhs = (np_x * np_nablaT_p).sum()
        
        if dim == 2:
            print('dtype: {} dim: {} hx: {} hy: {} diff: {}'.format(dtype, dim, hx, hy, np.abs(lhs - rhs)))
        elif dim == 3:
            print('dtype: {} dim: {} hx: {} hy: {} hz: {} diff: {}'.format(dtype, dim, hx, hy, hz, np.abs(lhs - rhs)))
        else: # dim == 4:
            print('dtype: {} dim: {} hx: {} hy: {} hz: {} ht: {} diff: {}'.format(dtype, dim, hx, hy, hz, ht, np.abs(lhs - rhs)))

        self.assertTrue(np.abs(lhs - rhs) < 1e-3)

    def test_float2_gradient(self):
        self._test_adjointness(np.float32, 2, 1, 1)

    def test_float3_gradient(self):
        self._test_adjointness(np.float32, 3, 1, 1, 1)

    def test_float4_gradient(self):
        self._test_adjointness(np.float32, 4, 1, 1, 1, 1)

    def test_double2_gradient(self):
        self._test_adjointness(np.float64, 2, 1, 1)

    def test_double3_gradient(self):
        self._test_adjointness(np.float64, 3, 1, 1, 1)

    def test_double4_gradient(self):
        self._test_adjointness(np.float64, 4, 1, 1, 1, 1)

    # anisotropic
    def test_float3_aniso_gradient(self):
        self._test_adjointness(np.float32, 3, 1, 1, 2)

    def test_double3_aniso_gradient(self):
        self._test_adjointness(np.float64, 3, 1, 1, 2)

    def test_float4_aniso_gradient(self):
        self._test_adjointness(np.float32, 4, 1, 1, 2, 4)

    def test_double4_aniso_gradient(self):
        self._test_adjointness(np.float64, 4, 1, 1, 2, 4)

class TestComplexNablaFunction(unittest.TestCase):           
    def _test_adjointness(self, dtype, dim, hx, hy, hz=1, ht=1):
        # get the corresponding operator
        op = Nabla(dim, hx, hy, hz, ht)
        # setup the vaiables
        shape = [10 for i in range(dim)]
        shape[0] *= 2
        np_x = np.random.randn(*shape).astype(dtype) + 1j * np.random.randn(*shape).astype(dtype)
        shape.insert(0, dim)
        np_p = np.random.randn(*shape).astype(dtype) + 1j * np.random.randn(*shape).astype(dtype)

        np_nabla_x = op.forward(np_x)
        np_nablaT_p = op.adjoint(np_p)

        lhs = (np_nabla_x * np.conj(np_p)).sum()
        rhs = (np_x * np.conj(np_nablaT_p)).sum()

        if dim == 2:
            print('dtype: {} dim: {} hx: {} hy: {} diff: {}'.format(dtype, dim, hx, hy, np.abs(lhs - rhs)))
        elif dim == 3:
            print('dtype: {} dim: {} hx: {} hy: {} hz: {} diff: {}'.format(dtype, dim, hx, hy, hz, np.abs(lhs - rhs)))
        else: # dim == 4:
            print('dtype: {} dim: {} hx: {} hy: {} hz: {} ht: {} diff: {}'.format(dtype, dim, hx, hy, hz, ht, np.abs(lhs - rhs)))

        self.assertTrue(np.abs(lhs - rhs) < 1e-3)

    def test_float2_gradient(self):
        self._test_adjointness(np.float32, 2, 1, 1)

    def test_float3_gradient(self):
        self._test_adjointness(np.float32, 3, 1, 1, 1)

    def test_float4_gradient(self):
        self._test_adjointness(np.float32, 4, 1, 1, 1, 1)

    def test_double2_gradient(self):
        self._test_adjointness(np.float64, 2, 1, 1)

    def test_double3_gradient(self):
        self._test_adjointness(np.float64, 3, 1, 1, 1)

    def test_double4_gradient(self):
        self._test_adjointness(np.float64, 4, 1, 1, 1, 1)

    # anisotropic
    def test_float3_aniso_gradient(self):
        self._test_adjointness(np.float32, 3, 1, 1, 2)

    def test_double3_aniso_gradient(self):
        self._test_adjointness(np.float64, 3, 1, 1, 2)

    def test_float4_aniso_gradient(self):
        self._test_adjointness(np.float32, 4, 1, 1, 2, 4)

    def test_double4_aniso_gradient(self):
        self._test_adjointness(np.float64, 4, 1, 1, 2, 4)

if __name__ == "__main__":
    unittest.test()
