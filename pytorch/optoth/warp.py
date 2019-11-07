import numpy as np
import torch
import unittest

import _ext.th_warp_operator

__all__ = ['WarpFunction']


class WarpFunction(torch.autograd.Function):
    @staticmethod
    def _get_operator(dtype):
        if dtype == torch.float32:
            return _ext.th_warp_operator.Warp_float()
        elif dtype == torch.float64:
            return _ext.th_warp_operator.Warp_double()
        else:
            raise RuntimeError('Invalid dtype!')

    @staticmethod
    def forward(ctx, x, u):
        ctx.save_for_backward(x, u)
        ctx.op = WarpFunction._get_operator(x.dtype)
        return ctx.op.forward(x, u)

    @staticmethod
    def backward(ctx, grad_out):
        x, u = ctx.saved_tensors
        grad_x = ctx.op.adjoint(x, u, grad_out)
        return grad_x, None

# to run execute: python -m unittest [-v] optoth.warp
class TestWarpFunction(unittest.TestCase):
    
    def _run_gradient_test(self, dtype):
        # setup the hyper parameters for each test
        M, N = 32, 32
        C = 3
        S = 4

        # perform a gradient check:
        epsilon = 1e-3

        # prefactors
        a = 1.0

        # transfer to torch
        cuda = torch.device('cuda')
        th_x = torch.randn(S, C, M, N, dtype=dtype, device=cuda)
        th_u = torch.randn(S, M, N, 2, dtype=dtype, device=cuda) * 0.1
        th_a = torch.tensor(a, requires_grad=True, dtype=th_x.dtype, device=cuda)
        op = WarpFunction()

        # setup the model
        compute_loss = lambda a: 0.5 * torch.sum(op.apply(th_x*a, th_u)**2)
        th_loss = compute_loss(th_a)

        # backpropagate the gradient
        th_loss.backward()
        grad_a = th_a.grad.item()

        # numerical gradient w.r.t. the input
        with torch.no_grad():
            l_ap = compute_loss(th_a+epsilon).item()
            l_an = compute_loss(th_a-epsilon).item()
            grad_a_num = (l_ap - l_an) / (2 * epsilon)

        print("grad_x: {:.7f} num_grad_x {:.7f} success: {}".format(
            grad_a, grad_a_num, np.abs(grad_a - grad_a_num) < 1e-4))
        self.assertTrue(np.abs(grad_a - grad_a_num) < 1e-4)
        
    def test_float_gradient(self):
        self._run_gradient_test(torch.float32)
    
    def test_double_gradient(self):
        self._run_gradient_test(torch.float64)

if __name__ == "__main__":
    unittest.test()
