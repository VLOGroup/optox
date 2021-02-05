import numpy as np
import torch
import unittest

import _ext.th_pad2d_operator

__all__ = ['pad2d', 'pad2d_transpose', 'pad2d_symmetric', 'pad2d_symmetric_transpose']


def pad2d(x, padding, mode):
    """Padding of a 2d tensor.
    
    This function pads a 2d tensor (rank 4). The tensorformat is [N, C, H, W]. The tensor is
    padded by values specified in `padding` [W0, W1, H0, H1] where 0 indicates the padding before and 1
    indicates the padding after. This functions supports the padding modes "reflect", "symmetric" and "replicate".

    Args:
        tensor: A `Tensor`.
        padding: A `Tensor` of type `int32`.
        mode: One of "reflect", "symmetric", or "replicate" (case-insensitive)
        channel_last: 
    Returns:
        A padded `Tensor`. Has the same type as `tensor`.
    """
    return PadFunction().apply(x, padding, mode)


def pad2d_transpose(x, padding, mode):
    """Transpose padding of a 2d tensor.
    
    This function transpose pads a 2d tensor (rank 4). The tensorformat is [N, C, D, H, W]. The tensor is
    padded by values specified in `padding` [W0, W1, H0, H1, D0, D1] where 0 indicates the padding before and 1
    indicates the padding after. This functions supports the padding modes "reflect", "symmetric" and "replicate".

    Args:
        tensor: A `Tensor`.
        padding: A `Tensor` of type `int32`.
        mode: One of "reflect", "symmetric", or "replicate" (case-insensitive)
        channel_last: 
    Returns:
        A transposed padded `Tensor`. Has the same type as `tensor`.
    """
    return PadFunctionTranspose().apply(x, padding, mode)


# legacy
def pad2d_symmetric(x, padding):
    return PadFunction().apply(x, padding, 'symmetric')


def pad2d_symmetric_transpose(x, padding):
    return PadFunctionTranspose().apply(x, padding, 'symmetric')


def get_operator(dtype, pad, mode):
    if dtype == torch.float32:
        return _ext.th_pad2d_operator.Pad2d_float(pad[0], pad[1], pad[2], pad[3], mode)
    elif dtype == torch.float64:
        return _ext.th_pad2d_operator.Pad2d_double(pad[0], pad[1], pad[2], pad[3], mode)
    else:
        raise RuntimeError('Invalid dtype!')


class PadFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, pad, mode):
        assert len(pad) == 4
        assert type(mode) == str
        ctx.op = get_operator(x.dtype, pad, mode)
        ctx.shape = x.shape
        out = ctx.op.forward(x.reshape(-1, x.shape[2], x.shape[3]).contiguous()).view(x.shape[0], x.shape[1], x.shape[2]+pad[2]+pad[3], x.shape[3]+pad[0]+pad[1])
        return out

    @staticmethod
    def backward(ctx, grad_out):
        grad_x = ctx.op.adjoint(grad_out.reshape(-1, grad_out.shape[2], grad_out.shape[3]).contiguous())
        return grad_x.view(ctx.shape), None, None


class PadFunctionTranspose(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, pad, mode):
        assert len(pad) == 4
        ctx.op = get_operator(x.dtype, pad, mode)
        ctx.shape = x.shape
        out = ctx.op.adjoint(x.reshape(-1, x.shape[2], x.shape[3]).contiguous()).view(x.shape[0], x.shape[1], x.shape[2]-pad[2]-pad[3], x.shape[3]-pad[0]-pad[1])
        return out

    @staticmethod
    def backward(ctx, grad_out):
        grad_x = ctx.op.forward(grad_out.reshape(-1, grad_out.shape[2], grad_out.shape[3]).contiguous())
        return grad_x.view(ctx.shape), None, None

# to run execute: python -m unittest [-v] optoth.pad2d
class Testpad2dFunction(unittest.TestCase):
    
    def _test_adjointness(self, dtype, mode):
        # setup the hyper parameters for each test
        S, C, M, N =4, 3, 32, 32

        pad = [3,3,2,2]

        # transfer to torch
        cuda = torch.device('cuda')
        x = torch.randn(S, C, M, N, dtype=dtype, device=cuda).requires_grad_(True)
        p = torch.randn(S, C, M+pad[2]+pad[3], N+pad[0]+pad[1], dtype=dtype, device=cuda).requires_grad_(True)

        Ax = pad2d(x, pad, mode)
        ATp = torch.autograd.grad(Ax, x, p)[0]


        lhs = (Ax * p).sum().item()
        rhs = (x * ATp).sum().item()

        print('forward: dtype={} diff: {}'.format(dtype, np.abs(lhs - rhs)))
        self.assertTrue(np.abs(lhs - rhs) < 1e-3)

        Ap = pad2d_transpose(p, pad, mode)
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
