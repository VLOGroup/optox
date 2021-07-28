import numpy as np
import torch
import unittest

import _ext.th_nabla_operator

__all__ = ['float_2d', 'double_2d', 'float_3d', 'double_3d', 'float_4d', 'double_4d']

float_2d = _ext.th_nabla_operator.Nabla_2d_float
double_2d = _ext.th_nabla_operator.Nabla_2d_double

float_3d = _ext.th_nabla_operator.Nabla_3d_float
double_3d = _ext.th_nabla_operator.Nabla_3d_double

float_4d = _ext.th_nabla_operator.Nabla_4d_float
double_4d = _ext.th_nabla_operator.Nabla_4d_double

class Nabla(torch.nn.Module):
    def __init__(self, dim, hx=1, hy=1, hz=1, ht=1, iscomplex=False):
        super().__init__()
        self.dim = dim
        self.hx = hx
        self.hy = hy
        self.hz = hz
        self.ht = ht
        self.iscomplex = iscomplex
    
    def _get_op(self, dtype, dim, hx, hy, hz=1, ht=1):
        if dtype == torch.float32:
            if dim == 2:
                return float_2d(hx, hy)
            elif dim == 3:
                return float_3d(hx, hy, hz)
            elif dim == 4:
                return float_4d(hx, hy, hz, ht)
            else:
                raise RuntimeError('Invalid number of dimensions!')
        elif dtype == torch.float64:
            if dim == 2:
                return double_2d(hx, hy)
            elif dim == 3:
                return double_3d(hx, hy, hz)
            elif dim == 4:
                return double_4d(hx, hy, hz, ht)
            else:
                raise RuntimeError('Invalid number of dimensions!')
        else:
            raise RuntimeError("Invalid number of dimensions!")

    def forward(self, x):
        op = self._get_op(x.dtype, self.dim, self.hx, self.hy, self.hz, self.ht)
        
        if self.iscomplex:
            Kx_re = op.forward(x[...,0].contiguous())
            Kx_im = op.forward(x[...,1].contiguous())
            return torch.cat([Kx_re.unsqueeze_(-1), Kx_im.unsqueeze_(-1)], -1)
        else:
            return op.forward(x)

    def adjoint(self, x):
        op = self._get_op(x.dtype, self.dim, self.hx, self.hy, self.hz, self.ht)
        
        if self.iscomplex:
            Kx_re = op.adjoint(x[...,0].contiguous())
            Kx_im = op.adjoint(x[...,1].contiguous())
            return torch.cat([Kx_re.unsqueeze_(-1), Kx_im.unsqueeze_(-1)], -1)
        else:
            return op.adjoint(x)

# to run execute: python -m unittest [-v] optoth.nabla
class TestNablaFunction(unittest.TestCase):
    def _test_adjointness(self, dtype, dim, hx, hy, hz=1, ht=1):
        assert dim in [2, 3, 4]
        # get the corresponding operator
        op = Nabla(dim, hx, hy, hz, ht)
        # setup the vaiables
        cuda = torch.device('cuda')
        shape = [30 for i in range(dim)]
        th_x = torch.randn(*shape, dtype=dtype, device=cuda)
        shape.insert(0, dim)
        th_p = torch.randn(*shape, dtype=dtype, device=cuda)

        th_nabla_x = op.forward(th_x)
        th_nablaT_p = op.adjoint(th_p)

        lhs = (th_nabla_x * th_p).sum().cpu().numpy()
        rhs = (th_x * th_nablaT_p).sum().cpu().numpy()

        if dim == 2:
            print('dtype: {} dim: {} hx: {} hy: {} diff: {}'.format(dtype, dim, hx, hy, np.abs(lhs - rhs)))
        elif dim == 3:
            print('dtype: {} dim: {} hx: {} hy: {} hz: {} diff: {}'.format(dtype, dim, hx, hy, hz, np.abs(lhs - rhs)))
        else: # dim == 4:
            print('dtype: {} dim: {} hx: {} hy: {} hz: {} ht: {} diff: {}'.format(dtype, dim, hx, hy, hz, ht, np.abs(lhs - rhs)))

        self.assertTrue(np.abs(lhs - rhs) < 1e-3)

    def test_float2_gradient(self):
        self._test_adjointness(torch.float32, 2, 1, 1)

    def test_float3_gradient(self):
        self._test_adjointness(torch.float32, 3, 1, 1, 1)

    def test_float4_gradient(self):
        self._test_adjointness(torch.float32, 4, 1, 1, 1, 1)

    def test_double2_gradient(self):
        self._test_adjointness(torch.float64, 2, 1, 1)

    def test_double3_gradient(self):
        self._test_adjointness(torch.float64, 3, 1, 1, 1)

    def test_double4_gradient(self):
        self._test_adjointness(torch.float64, 4, 1, 1, 1, 1)

    # anisotropic
    def test_float3_aniso_gradient(self):
        self._test_adjointness(torch.float32, 3, 1, 1, 2)

    def test_double3_aniso_gradient(self):
        self._test_adjointness(torch.float64, 3, 1, 1, 2)

    def test_float4_aniso_gradient(self):
        self._test_adjointness(torch.float32, 4, 1, 1, 2, 4)

    def test_double4_aniso_gradient(self):
        self._test_adjointness(torch.float64, 4, 1, 1, 2, 4)

class TestComplexNablaFunction(unittest.TestCase):            
    def _test_adjointness(self, dtype, dim, hx, hy, hz=1, ht=1):
        assert dim in [2, 3, 4]
        # get the corresponding operator
        op = Nabla(dim, hx, hy, hz, iscomplex=True)
        # setup the vaiables
        cuda = torch.device('cuda')
        shape = [30 for i in range(dim)] + [2,]
        shape[0] *= 2
        th_x = torch.randn(*shape, dtype=dtype, device=cuda)
        shape.insert(0, dim)
        th_p = torch.randn(*shape, dtype=dtype, device=cuda)

        th_nabla_x = op.forward(th_x)
        th_nablaT_p = op.adjoint(th_p)

        # conjugate
        th_p_conj = th_p.clone()
        th_p_conj[...,1] *= -1

        th_nablaT_p_conj = th_nablaT_p.clone()
        th_nablaT_p_conj[...,1] *= -1

        lhs = (th_nabla_x * th_p_conj).sum().cpu().numpy()
        rhs = (th_x * th_nablaT_p_conj).sum().cpu().numpy()

        if dim == 2:
            print('dtype: {} dim: {} hx: {} hy: {} diff: {}'.format(dtype, dim, hx, hy, np.abs(lhs - rhs)))
        elif dim == 3:
            print('dtype: {} dim: {} hx: {} hy: {} hz: {} diff: {}'.format(dtype, dim, hx, hy, hz, np.abs(lhs - rhs)))
        else: # dim == 4:
            print('dtype: {} dim: {} hx: {} hy: {} hz: {} ht: {} diff: {}'.format(dtype, dim, hx, hy, hz, ht, np.abs(lhs - rhs)))

        self.assertTrue(np.abs(lhs - rhs) < 1e-3)

    def test_float2_gradient(self):
        self._test_adjointness(torch.float32, 2, 1, 1)

    def test_float3_gradient(self):
        self._test_adjointness(torch.float32, 3, 1, 1, 1)

    def test_float4_gradient(self):
        self._test_adjointness(torch.float32, 3, 1, 1, 1, 1)

    def test_double2_gradient(self):
        self._test_adjointness(torch.float64, 2, 1, 1)

    def test_double3_gradient(self):
        self._test_adjointness(torch.float64, 3, 1, 1, 1)

    def test_double4_gradient(self):
        self._test_adjointness(torch.float64, 3, 1, 1, 1, 1)

    # anisotropic
    def test_float3_aniso_gradient(self):
        self._test_adjointness(torch.float32, 3, 1, 1, 2)

    def test_double3_aniso_gradient(self):
        self._test_adjointness(torch.float64, 3, 1, 1, 2)

    def test_float4_aniso_gradient(self):
        self._test_adjointness(torch.float32, 4, 1, 1, 2, 4)

    def test_double4_aniso_gradient(self):
        self._test_adjointness(torch.float64, 4, 1, 1, 2, 4)

if __name__ == "__main__":
    unittest.test()
