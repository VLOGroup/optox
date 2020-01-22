import numpy as np
import torch
import unittest

import _ext.th_pad2d_operator

__all__ = ['PadFunction', 'PadFunctionTranspose']


class PadFunction(torch.autograd.Function):
    @staticmethod
    def _get_operator(dtype, pad):
        if dtype == torch.float32:
            return _ext.th_pad2d_operator.Pad2d_float(pad[0], pad[1], pad[2], pad[3])
        elif dtype == torch.float64:
            return _ext.th_pad2d_operator.Pad2d_double(pad[0], pad[1], pad[2], pad[3])
        else:
            raise RuntimeError('Invalid dtype!')

    @staticmethod
    def forward(ctx, x, pad):
        assert len(pad) == 4
        ctx.op = PadFunction._get_operator(x.dtype, pad)
        ctx.shape = x.shape
        out = ctx.op.forward(x.view(-1, x.shape[2], x.shape[3])).view(x.shape[0], x.shape[1], x.shape[2]+pad[2]+pad[3], x.shape[3]+pad[0]+pad[1])
        return out

    @staticmethod
    def backward(ctx, grad_out):
        grad_x = ctx.op.adjoint(grad_out.view(-1, grad_out.shape[2], grad_out.shape[3]))
        return grad_x.view(ctx.shape), None


class PadFunctionTranspose(torch.autograd.Function):
    @staticmethod
    def _get_operator(dtype, pad):
        if dtype == torch.float32:
            return _ext.th_pad2d_operator.Pad2d_float(pad[0], pad[1], pad[2], pad[3])
        elif dtype == torch.float64:
            return _ext.th_pad2d_operator.Pad2d_double(pad[0], pad[1], pad[2], pad[3])
        else:
            raise RuntimeError('Invalid dtype!')

    @staticmethod
    def forward(ctx, x, pad):
        assert len(pad) == 4
        ctx.op = PadFunction._get_operator(x.dtype, pad)
        ctx.shape = x.shape
        out = ctx.op.adjoint(x.view(-1, x.shape[2], x.shape[3])).view(x.shape[0], x.shape[1], x.shape[2]-pad[2]-pad[3], x.shape[3]-pad[0]-pad[1])
        return out

    @staticmethod
    def backward(ctx, grad_out):
        grad_x = ctx.op.forward(grad_out.view(-1, grad_out.shape[2], grad_out.shape[3]))
        return grad_x.view(ctx.shape), None

# to run execute: python -m unittest [-v] optoth.pad2d
class Testpad2dFunction(unittest.TestCase):
    
    def _test_adjointness(self, dtype):
        # setup the hyper parameters for each test
        S, C, M, N =4, 3, 32, 32

        pad = [3,3,2,2]

        # transfer to torch
        cuda = torch.device('cuda')
        x = torch.randn(S, C, M, N, dtype=dtype, device=cuda).requires_grad_(True)
        p = torch.randn(S, C, M+pad[2]+pad[3], N+pad[0]+pad[1], dtype=dtype, device=cuda).requires_grad_(True)

        op = PadFunction()
        Ax = op.apply(x, pad)
        ATp = torch.autograd.grad(Ax, x, p)[0]


        lhs = (Ax * p).sum().item()
        rhs = (x * ATp).sum().item()

        print('forward: dtype={} diff: {}'.format(dtype, np.abs(lhs - rhs)))
        self.assertTrue(np.abs(lhs - rhs) < 1e-3)

        op = PadFunctionTranspose()
        Ap = op.apply(p, pad)
        ATx = torch.autograd.grad(Ap, p, x)[0]

        lhs = (Ap * x).sum().item()
        rhs = (p * ATx).sum().item()

        print('adjoint: dtype={} diff: {}'.format(dtype, np.abs(lhs - rhs)))
        self.assertTrue(np.abs(lhs - rhs) < 1e-3)

    def test_float(self):
        self._test_adjointness(torch.float32)
    
    def test_double(self):
        self._test_adjointness(torch.float64)

if __name__ == "__main__":
    unittest.test()
