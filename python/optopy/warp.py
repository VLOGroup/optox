import numpy as np
import unittest

import _ext.py_warp_operator

__all__ = ['float_2d', 'double_2d']

float_2d = _ext.py_warp_operator.Warp_2d_float
double_2d = _ext.py_warp_operator.Warp_2d_double

class Warp(object):
    def __init__(self):
        self.u = None
        self.x = None

    def _get_op(self, dtype):
        if dtype == np.float32 or dtype == np.complex64:
            return float_2d()
        elif dtype == np.float64 or dtype == np.complex128:
            return double_2d()
        else:
            raise RuntimeError('Invalid dtype!')

    def forward(self, x, u):
        # save for backward
        self.u = u

        op = self._get_op(x.dtype)
        
        if np.iscomplexobj(x):
            x_re = np.ascontiguousarray(np.real(x))
            x_im = np.ascontiguousarray(np.imag(x))
            return op.forward(x_re, u) + 1j * op.forward(x_im, u)
        else:
            return op.forward(x, u)

    def adjoint(self, grad_out, u = None):
        # check if all inputs are available
        if self.u is None and u is None:
            raise RuntimeError("u is not defined. Forward not called or u not passed in function call.")
        
        op = self._get_op(grad_out.dtype)

        # use passed variables or saved ones from forward path
        u = self.u if u is None else u

        if np.iscomplexobj(grad_out):
            grad_out_re = np.ascontiguousarray(np.real(grad_out))
            grad_out_im = np.ascontiguousarray(np.imag(grad_out))
            return op.adjoint(grad_out_re, u) + 1j * op.adjoint(grad_out_im, u)
        else:
            return op.adjoint(grad_out, u)

class TestWarpOperator(unittest.TestCase):
    def _test_adjointness(self, dtype):
        # get the corresponding operator
        op = Warp()

        # setup the vaiables
        shape_x = (10, 5, 20, 20)
        shape_u = (10, 20, 20, 2)

        np_x = np.random.randn(*shape_x).astype(dtype)
        np_u = np.random.randn(*shape_u).astype(dtype) * 10.0

        np_p = np.random.randn(*shape_x).astype(dtype)
        np_warp_x = op.forward(np_x, np_u)
        np_warpT_p = op.adjoint(np_p, np_u)

        lhs = (np_warp_x * np_p).sum()
        rhs = (np_x * np_warpT_p).sum()

        print('dtype: {} diff: {}'.format(dtype, np.abs(lhs - rhs)))
        self.assertTrue(np.abs(lhs - rhs) < 1e-3)

    def test_float2_adjointness(self):
        self._test_adjointness(np.float32)

    def test_double2_adjointness(self):
        self._test_adjointness(np.float64)

class TestComplexWarpOperator(unittest.TestCase):
    def _test_adjointness(self, dtype):
        # get the corresponding operator
        op = Warp()

        # setup the vaiables
        shape_x = (10, 5, 20, 20)
        shape_u = (10, 20, 20, 2)

        np_x = np.random.randn(*shape_x).astype(dtype) + 1j * np.random.randn(*shape_x).astype(dtype) 
        np_u = np.random.randn(*shape_u).astype(dtype) * 10.0

        np_p = np.random.randn(*shape_x).astype(dtype) + 1j * np.random.randn(*shape_x).astype(dtype) 
        np_warp_x = op.forward(np_x, np_u)
        np_warpT_p = op.adjoint(np_p, np_u)

        lhs = (np_warp_x * np.conj(np_p)).sum()
        rhs = (np_x * np.conj(np_warpT_p)).sum()

        print('dtype: {} diff: {}'.format(dtype, np.abs(lhs - rhs)))
        self.assertTrue(np.abs(lhs - rhs) < 1e-3)

    def test_float2_adjointness(self):
        self._test_adjointness(np.float32)

    def test_double2_adjointness(self):
        self._test_adjointness(np.float64)

if __name__ == "__main__":
    unittest.test()
