import numpy as np
import unittest

import _ext.py_pad_operator

__all__ = ['float_2d', 'double_2d', 'float_3d', 'double_3d']

float_2d = _ext.py_pad_operator.Pad2d_float
double_2d = _ext.py_pad_operator.Pad2d_double

float_3d = _ext.py_pad_operator.Pad3d_float
double_3d = _ext.py_pad_operator.Pad3d_double

class Pad2d(object):
    def __init__(self, padding, mode):
        self.padding = padding
        self.mode = mode

    def _get_op(self, dtype):
        if dtype == np.float32 or dtype == np.complex64:
            return float_2d(*self.padding, self.mode) 
        elif dtype == np.float64 or dtype == np.complex128:
            return double_2d(*self.padding, self.mode) 
        else:
            raise RuntimeError('Invalid dtype!')

    def forward(self, x):
        op = self._get_op(x.dtype)
        
        if np.iscomplexobj(x):
            x_re = np.ascontiguousarray(np.real(x))
            x_im = np.ascontiguousarray(np.imag(x))
            return op.forward(x_re) + 1j * op.forward(x_im)
        else:
            return op.forward(x)

    def adjoint(self, x):
        op = self._get_op(x.dtype)
        
        if np.iscomplexobj(x):
            x_re = np.ascontiguousarray(np.real(x))
            x_im = np.ascontiguousarray(np.imag(x))
            return op.adjoint(x_re) + 1j * op.adjoint(x_im)
        else:
            return op.adjoint(x)

class Pad3d(Pad2d):
    def _get_op(self, dtype):
        if dtype == np.float32 or dtype == np.complex64:
            return float_3d(*self.padding, self.mode) 
        elif dtype == np.float64 or dtype == np.complex128:
            return double_3d(*self.padding, self.mode) 
        else:
            raise RuntimeError('Invalid dtype!')

# to run execute: python -m unittest [-v] optopy.nabla
class TestPad2dFunction(unittest.TestCase):           
    def _test_adjointness(self, dtype, mode):
        # get the corresponding operator
        padding = [3,3,2,2]
        op = Pad2d(padding, mode)
        # setup the vaiables
        shape = [4, 32, 32]
        np_x = np.random.randn(*shape).astype(dtype)
        padded_shape = shape
        padded_shape[1] += padding[2] + padding[3]
        padded_shape[2] += padding[0] + padding[1]
        np_p = np.random.randn(*padded_shape).astype(dtype)

        np_K_x = op.forward(np_x)
        np_KT_p = op.adjoint(np_p)

        lhs = (np_K_x * np_p).sum()
        rhs = (np_x * np_KT_p).sum()

        print('forward: dtype={} mode={} diff: {}'.format(dtype, mode, np.abs(lhs - rhs)))
        self.assertTrue(np.abs(lhs - rhs) < 1e-3)

    def test_float_symmetric(self):
        self._test_adjointness(np.float32, 'symmetric')

    def test_float_replicate(self):
        self._test_adjointness(np.float32, 'replicate')
    
    def test_float_reflect(self):
        self._test_adjointness(np.float32, 'reflect')
    
    def test_double_symmetric(self):
        self._test_adjointness(np.float64, 'symmetric')

    def test_double_replicate(self):
        self._test_adjointness(np.float64, 'replicate')
    
    def test_double_reflect(self):
        self._test_adjointness(np.float64, 'reflect')

class TestComplexPad2dFunction(unittest.TestCase):           
    def _test_adjointness(self, dtype, mode):
        # get the corresponding operator
        padding = [3,3,2,2]
        op = Pad2d(padding, mode)

        # setup the vaiables
        shape = [4, 32, 32]
        np_x = np.random.randn(*shape).astype(dtype) + 1j * np.random.randn(*shape).astype(dtype)
        padded_shape = shape
        padded_shape[1] += padding[2] + padding[3]
        padded_shape[2] += padding[0] + padding[1]
        np_p = np.random.randn(*padded_shape).astype(dtype) + 1j * np.random.randn(*padded_shape).astype(dtype)

        np_K_x = op.forward(np_x)
        np_KT_p = op.adjoint(np_p)

        lhs = (np_K_x * np.conj(np_p)).sum()
        rhs = (np_x * np.conj(np_KT_p)).sum()

        print('adjoint: dtype={} mode={} diff: {}'.format(dtype, mode, np.abs(lhs - rhs)))
        self.assertTrue(np.abs(lhs - rhs) < 1e-3)

    def test_float_symmetric(self):
        self._test_adjointness(np.float32, 'symmetric')

    def test_float_replicate(self):
        self._test_adjointness(np.float32, 'replicate')
    
    def test_float_reflect(self):
        self._test_adjointness(np.float32, 'reflect')
    
    def test_double_symmetric(self):
        self._test_adjointness(np.float64, 'symmetric')

    def test_double_replicate(self):
        self._test_adjointness(np.float64, 'replicate')
    
    def test_double_reflect(self):
        self._test_adjointness(np.float64, 'reflect')

class TestPad3dFunction(unittest.TestCase):           
    def _test_adjointness(self, dtype, mode):
        # get the corresponding operator
        padding = [3,3,2,2,1,1]
        op = Pad3d(padding, mode)
        # setup the vaiables
        shape = [4, 32, 32, 32]
        np_x = np.random.randn(*shape).astype(dtype)
        padded_shape = shape
        padded_shape[1] += padding[4] + padding[5]
        padded_shape[2] += padding[2] + padding[3]
        padded_shape[3] += padding[0] + padding[1]
        np_p = np.random.randn(*padded_shape).astype(dtype)

        np_K_x = op.forward(np_x)
        np_KT_p = op.adjoint(np_p)

        lhs = (np_K_x * np_p).sum()
        rhs = (np_x * np_KT_p).sum()

        print('forward: dtype={} mode={} diff: {}'.format(dtype, mode, np.abs(lhs - rhs)))
        self.assertTrue(np.abs(lhs - rhs) < 1e-3)

    def test_float_symmetric(self):
        self._test_adjointness(np.float32, 'symmetric')

    def test_float_replicate(self):
        self._test_adjointness(np.float32, 'replicate')
    
    def test_float_reflect(self):
        self._test_adjointness(np.float32, 'reflect')
    
    def test_double_symmetric(self):
        self._test_adjointness(np.float64, 'symmetric')

    def test_double_replicate(self):
        self._test_adjointness(np.float64, 'replicate')
    
    def test_double_reflect(self):
        self._test_adjointness(np.float64, 'reflect')

class TestComplexPad3dFunction(unittest.TestCase):           
    def _test_adjointness(self, dtype, mode):
        # get the corresponding operator
        padding = [3,3,2,2,1,1]
        op = Pad3d(padding, mode)

        # setup the vaiables
        shape = [4, 32, 32, 32]
        np_x = np.random.randn(*shape).astype(dtype) + 1j * np.random.randn(*shape).astype(dtype)
        padded_shape = shape
        padded_shape[1] += padding[4] + padding[5]
        padded_shape[2] += padding[2] + padding[3]
        padded_shape[3] += padding[0] + padding[1]
        np_p = np.random.randn(*padded_shape).astype(dtype) + 1j * np.random.randn(*padded_shape).astype(dtype)

        np_K_x = op.forward(np_x)
        np_KT_p = op.adjoint(np_p)

        lhs = (np_K_x * np.conj(np_p)).sum()
        rhs = (np_x * np.conj(np_KT_p)).sum()

        print('adjoint: dtype={} mode={} diff: {}'.format(dtype, mode, np.abs(lhs - rhs)))
        self.assertTrue(np.abs(lhs - rhs) < 1e-3)

    def test_float_symmetric(self):
        self._test_adjointness(np.float32, 'symmetric')

    def test_float_replicate(self):
        self._test_adjointness(np.float32, 'replicate')
    
    def test_float_reflect(self):
        self._test_adjointness(np.float32, 'reflect')
    
    def test_double_symmetric(self):
        self._test_adjointness(np.float64, 'symmetric')

    def test_double_replicate(self):
        self._test_adjointness(np.float64, 'replicate')
    
    def test_double_reflect(self):
        self._test_adjointness(np.float64, 'reflect')

if __name__ == "__main__":
    unittest.test()
