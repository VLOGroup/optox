import numpy as np
import torch
import unittest

import _ext.th_pad3d_operator

__all__ = ['pad3d', 'pad3d_tranpose', 'pad3d_symmetric', 'pad3d_symmetric_transpose']


def pad3d(x, padding, mode):
    return PadFunction().apply(x, padding, mode)


def pad3d_transpose(x, padding, mode):
    return PadFunctionTranspose().apply(x, padding, mode)


# legacy
def pad3d_symmetric(x, padding):
    return PadFunction().apply(x, padding, 'symmetric')


def pad3d_symmetric_transpose(x, padding):
    return PadFunctionTranspose().apply(x, padding, 'symmetric')


def get_operator(dtype, pad, mode):
    if dtype == torch.float32:
        return _ext.th_pad3d_operator.Pad3d_float(pad[0], pad[1], pad[2], pad[3], pad[4], pad[5], mode)
    elif dtype == torch.float64:
        return _ext.th_pad3d_operator.Pad3d_double(pad[0], pad[1], pad[2], pad[3], pad[4], pad[5], mode)
    else:
        raise RuntimeError('Invalid dtype!')


class PadFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, pad, mode):
        assert len(pad) == 6
        assert type(mode) == str
        ctx.op = get_operator(x.dtype, pad, mode)
        ctx.shape = x.shape
        pad_shape = list(x.shape)
        pad_shape[-3] += pad[4]+pad[5]
        pad_shape[-2] += pad[2]+pad[3]
        pad_shape[-1] += pad[0]+pad[1]
        out = ctx.op.forward(x.reshape(-1, x.shape[-3], x.shape[-2], x.shape[-1])).view(pad_shape)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        grad_x = ctx.op.adjoint(grad_out.reshape(-1, grad_out.shape[-3], grad_out.shape[-2], grad_out.shape[-1]))
        return grad_x.view(ctx.shape), None, None


class PadFunctionTranspose(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, pad, mode):
        assert len(pad) == 6
        ctx.op = get_operator(x.dtype, pad, mode)
        ctx.shape = x.shape
        padT_shape = list(x.shape)
        padT_shape[-3] += -pad[4]-pad[5]
        padT_shape[-2] += -pad[2]-pad[3]
        padT_shape[-1] += -pad[0]-pad[1]
        out = ctx.op.adjoint(x.reshape(-1, x.shape[-3], x.shape[-2], x.shape[-1])).view(padT_shape)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        grad_x = ctx.op.forward(grad_out.reshape(-1, grad_out.shape[-3], grad_out.shape[-2], grad_out.shape[-1]))
        return grad_x.view(ctx.shape), None, None

# to run execute: python -m unittest [-v] optoth.pad3d
class Testpad3dFunction(unittest.TestCase):
    
    def _test_adjointness(self, dtype, mode):                   
        # setup the hyper parameters for each test
        S, C, D, M, N =4, 3, 16, 32, 32

        pad = [3,3,2,2,1,1]

        # transfer to torch
        cuda = torch.device('cuda')
        x = torch.randn(S, C, D, M, N, dtype=dtype, device=cuda).requires_grad_(True)
        p = torch.randn(S, C, D+pad[4]+pad[5], M+pad[2]+pad[3], N+pad[0]+pad[1], dtype=dtype, device=cuda).requires_grad_(True)

        Ax = pad3d(x, pad, mode)
        ATp = torch.autograd.grad(Ax, x, p)[0]

        lhs = (Ax * p).sum().item()
        rhs = (x * ATp).sum().item()

        print('forward: dtype={} diff: {}'.format(dtype, np.abs(lhs - rhs)))
        self.assertTrue(np.abs(lhs - rhs) < 1e-3)

        Ap = pad3d_transpose(p, pad, mode)
        ATx = torch.autograd.grad(Ap, p, x)[0]

        lhs = (Ap * x).sum().item()
        rhs = (p * ATx).sum().item()

        print('adjoint: dtype={} diff: {}'.format(dtype, np.abs(lhs - rhs)))
        self.assertTrue(np.abs(lhs - rhs) < 1e-3)

    def test_float_symmetric(self):
        self._test_adjointness(torch.float32, 'symmetric')

    def test_float_replicate(self):
        self._test_adjointness(torch.float32, 'replicate')
    
    def test_float_reflect(self):
        self._test_adjointness(torch.float32, 'reflect')
    
    def test_double_symmetric(self):
        self._test_adjointness(torch.float64, 'symmetric')

    def test_double_replicate(self):
        self._test_adjointness(torch.float64, 'replicate')
    
    def test_double_reflect(self):
        self._test_adjointness(torch.float64, 'reflect')

if __name__ == "__main__":
    unittest.test()
