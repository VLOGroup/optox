import numpy as np
import torch
import unittest

import _ext.th_nabla_operator

__all__ = ['float_2d', 'double_2d', 'float_3d', 'double_3d', 'float_4d', 'double_4d']

float_2d = _ext.th_nabla_operator.Nabla2_2d_float
double_2d = _ext.th_nabla_operator.Nabla2_2d_double

float_3d = _ext.th_nabla_operator.Nabla2_3d_float
double_3d = _ext.th_nabla_operator.Nabla2_3d_double

float_4d = _ext.th_nabla_operator.Nabla2_4d_float
double_4d = _ext.th_nabla_operator.Nabla2_4d_double

# to run execute: python -m unittest [-v] optoth.nabla2
class TestNabla2Function(unittest.TestCase):
    
    def _get_nabla_op(self, dtype, dim):
        if dtype == torch.float32:
            if dim == 2:
                return float_2d()
            elif dim == 3:
                return float_3d()
            elif dim == 4:
                return float_4d()
            else:
                raise RuntimeError('Invalid number of dimensions!')
        elif dtype == torch.float64:
            if dim == 2:
                return double_2d()
            elif dim == 3:
                return double_3d()
            elif dim == 4:
                return double_4d()
            else:
                raise RuntimeError('Invalid number of dimensions!')
        else:
            raise RuntimeError('Invalid dtype!')
            
    def _test_adjointness(self, dtype, dim):
        # get the corresponding operator
        op = self._get_nabla_op(dtype, dim)
        # setup the vaiables
        cuda = torch.device('cuda')
        shape = [dim,] + [30 for i in range(dim)]
        th_x = torch.randn(*shape, dtype=dtype, device=cuda)
        shape[0] = dim**2
        th_p = torch.randn(*shape, dtype=dtype, device=cuda)

        th_nabla_x = op.forward(th_x)
        th_nablaT_p = op.adjoint(th_p)

        lhs = (th_nabla_x * th_p).sum().cpu().numpy()
        rhs = (th_x * th_nablaT_p).sum().cpu().numpy()

        print('dtype: {} dim: {} diff: {}'.format(dtype, dim, np.abs(lhs - rhs)))
        self.assertTrue(np.abs(lhs - rhs) < 1e-3)

    def test_float2_gradient(self):
        self._test_adjointness(torch.float32, 2)

    def test_float3_gradient(self):
        self._test_adjointness(torch.float32, 3)

    def test_float4_gradient(self):
        self._test_adjointness(torch.float32, 4)

    def test_double2_gradient(self):
        self._test_adjointness(torch.float64, 2)

    def test_double3_gradient(self):
        self._test_adjointness(torch.float64, 3)

    def test_double4_gradient(self):
        self._test_adjointness(torch.float64, 4)


if __name__ == "__main__":
    unittest.test()
