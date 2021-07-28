import numpy as np
import torch
import unittest

import _ext.th_pad_operator

__all__ = ['pad2d',
           'pad2d_transpose',
           'pad2d_symmetric',
           'pad2d_symmetric_transpose',
           'pad3d',
           'pad3d_tranpose',
           'pad3d_symmetric',
           'pad3d_symmetric_transpose']


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
    return Pad2dFunction().apply(x, padding, mode)


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
    return Pad2dFunctionTranspose().apply(x, padding, mode)


# legacy
def pad2d_symmetric(x, padding):
    return Pad2dFunction().apply(x, padding, 'symmetric')


def pad2d_symmetric_transpose(x, padding):
    return Pad2dFunctionTranspose().apply(x, padding, 'symmetric')

# complex
def complex_pad2d(x, padding, mode='symmetric'):
    xp_re = pad2d(x[...,0].contiguous(), padding, mode=mode)
    xp_im = pad2d(x[...,1].contiguous(), padding, mode=mode)

    new_shape = list(xp_re.shape)
    new_shape.append(2)
    xp = torch.zeros(*new_shape, device=x.device, dtype=x.dtype)
    xp[...,0] = xp_re
    xp[...,1] = xp_im

    return xp

def complex_pad2d_transpose(x, padding, mode='symmetric'):
    xp_re = pad2d_transpose(x[...,0].contiguous(), padding, mode=mode)
    xp_im = pad2d_transpose(x[...,1].contiguous(), padding, mode=mode)

    new_shape = list(xp_re.shape)
    new_shape.append(2)
    xp = torch.zeros(*new_shape, device=x.device, dtype=x.dtype)
    xp[...,0] = xp_re
    xp[...,1] = xp_im

    return xp

def get_operator_2d(dtype, pad, mode):
    if dtype == torch.float32:
        return _ext.th_pad_operator.Pad2d_float(pad[0], pad[1], pad[2], pad[3], mode)
    elif dtype == torch.float64:
        return _ext.th_pad_operator.Pad2d_double(pad[0], pad[1], pad[2], pad[3], mode)
    else:
        raise RuntimeError('Invalid dtype!')


class Pad2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, pad, mode):
        assert len(pad) == 4
        assert type(mode) == str
        ctx.op = get_operator_2d(x.dtype, pad, mode)
        ctx.shape = x.shape
        out = ctx.op.forward(x.reshape(-1, x.shape[2], x.shape[3])).view(x.shape[0], x.shape[1], x.shape[2]+pad[2]+pad[3], x.shape[3]+pad[0]+pad[1])
        return out

    @staticmethod
    def backward(ctx, grad_out):
        grad_x = ctx.op.adjoint(grad_out.reshape(-1, grad_out.shape[2], grad_out.shape[3]))
        return grad_x.view(ctx.shape), None, None


class Pad2dFunctionTranspose(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, pad, mode):
        assert len(pad) == 4
        ctx.op = get_operator_2d(x.dtype, pad, mode)
        ctx.shape = x.shape
        out = ctx.op.adjoint(x.reshape(-1, x.shape[2], x.shape[3])).view(x.shape[0], x.shape[1], x.shape[2]-pad[2]-pad[3], x.shape[3]-pad[0]-pad[1])
        return out

    @staticmethod
    def backward(ctx, grad_out):
        grad_x = ctx.op.forward(grad_out.reshape(-1, grad_out.shape[2], grad_out.shape[3]))
        return grad_x.view(ctx.shape), None, None

def pad3d(x, padding, mode):
    """Padding of a 3d tensor.
    
    This function pads a 3d tensor (rank 5). The tensorformat is [N, C, D, H, W]. The tensor is
    padded by values specified in `padding` [W0, W1, H0, H1, D0, D1] where 0 indicates the padding before and 1
    indicates the padding after. This functions supports the padding modes "reflect", "symmetric" and "replicate".

    Args:
        tensor: A `Tensor`.
        padding: A `Tensor` of type `int32`.
        mode: One of "reflect", "symmetric", or "replicate" (case-insensitive)
        channel_last: 
    Returns:
        A padded `Tensor`. Has the same type as `tensor`.
    """
    return Pad3dFunction().apply(x, padding, mode)


def pad3d_transpose(x, padding, mode):
    """Transpose padding of a 3d tensor.
    
    This function transpose pads a 3d tensor (rank 5). The tensorformat is [N, C, D, H, W]. The tensor is
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
    return Pad3dFunctionTranspose().apply(x, padding, mode)


# legacy
def pad3d_symmetric(x, padding):
    return Pad3dFunction().apply(x, padding, 'symmetric')


def pad3d_symmetric_transpose(x, padding):
    return Pad3dFunctionTranspose().apply(x, padding, 'symmetric')

# complex
def complex_pad3d(x, padding, mode='symmetric'):
    xp_re = pad3d(x[...,0].contiguous(), padding, mode=mode)
    xp_im = pad3d(x[...,1].contiguous(), padding, mode=mode)

    new_shape = list(xp_re.shape)
    new_shape.append(2)
    xp = torch.zeros(*new_shape, device=x.device, dtype=x.dtype)
    xp[...,0] = xp_re
    xp[...,1] = xp_im

    return xp

def complex_pad3d_transpose(x, padding, mode='symmetric'):
    xp_re = pad3d_transpose(x[...,0].contiguous(), padding, mode=mode)
    xp_im = pad3d_transpose(x[...,1].contiguous(), padding, mode=mode)

    new_shape = list(xp_re.shape)
    new_shape.append(2)
    xp = torch.zeros(*new_shape, device=x.device, dtype=x.dtype)
    xp[...,0] = xp_re
    xp[...,1] = xp_im

    return xp

def get_operator_3d(dtype, pad, mode):
    if dtype == torch.float32:
        return _ext.th_pad_operator.Pad3d_float(pad[0], pad[1], pad[2], pad[3], pad[4], pad[5], mode)
    elif dtype == torch.float64:
        return _ext.th_pad_operator.Pad3d_double(pad[0], pad[1], pad[2], pad[3], pad[4], pad[5], mode)
    else:
        raise RuntimeError('Invalid dtype!')


class Pad3dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, pad, mode):
        assert len(pad) == 6
        assert type(mode) == str
        ctx.op = get_operator_3d(x.dtype, pad, mode)
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


class Pad3dFunctionTranspose(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, pad, mode):
        assert len(pad) == 6
        ctx.op = get_operator_3d(x.dtype, pad, mode)
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

        print(Ax.shape, x.shape)

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

class TestComplexpad2dFunction(unittest.TestCase):
    
    def _test_adjointness(self, dtype, mode):
        # setup the hyper parameters for each test
        S, C, M, N =4, 3, 32, 32

        pad = [3,3,2,2]

        # transfer to torch
        cuda = torch.device('cuda')
        x = torch.randn(S, C, M, N, 2, dtype=dtype, device=cuda).requires_grad_(True)
        p = torch.randn(S, C, M+pad[2]+pad[3], N+pad[0]+pad[1], 2, dtype=dtype, device=cuda).requires_grad_(True)

        Ax = complex_pad2d(x, pad, mode)
        ATp = torch.autograd.grad(Ax, x, p)[0]


        lhs = (Ax * p).sum().item()
        rhs = (x * ATp).sum().item()

        print('forward: dtype={} diff: {}'.format(dtype, np.abs(lhs - rhs)))
        self.assertTrue(np.abs(lhs - rhs) < 1e-3)

        Ap = complex_pad2d_transpose(p, pad, mode)
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
class TestComplexpad3dFunction(unittest.TestCase):
    
    def _test_adjointness(self, dtype, mode):                   
        # setup the hyper parameters for each test
        S, C, D, M, N =4, 3, 16, 32, 32

        pad = [3,3,2,2,1,1]

        # transfer to torch
        cuda = torch.device('cuda')
        x = torch.randn(S, C, D, M, N, 2, dtype=dtype, device=cuda).requires_grad_(True)
        p = torch.randn(S, C, D+pad[4]+pad[5], M+pad[2]+pad[3], N+pad[0]+pad[1], 2, dtype=dtype, device=cuda).requires_grad_(True)

        Ax = complex_pad3d(x, pad, mode)
        ATp = torch.autograd.grad(Ax, x, p)[0]

        lhs = (Ax * p).sum().item()
        rhs = (x * ATp).sum().item()

        print(Ax.shape, x.shape)

        print('forward: dtype={} diff: {}'.format(dtype, np.abs(lhs - rhs)))
        self.assertTrue(np.abs(lhs - rhs) < 1e-3)

        Ap = complex_pad3d_transpose(p, pad, mode)
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
