import os

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

import _ext.th_act_operators

import unittest


class ActivationFunction(torch.autograd.Function):
    @staticmethod
    def _get_operator(dtype, base_type, vmin, vmax):
        if base_type == 'rbf':
            if dtype == torch.float32:
                return _ext.th_act_operators.RbfAct_float(vmin, vmax)
            elif dtype == torch.float64:
                return _ext.th_act_operators.RbfAct_double(vmin, vmax)
            else:
                raise RuntimeError('Unsupported dtype!')
        elif base_type == 'linear':
            if dtype == torch.float32:
                return _ext.th_act_operators.LinearAct_float(vmin, vmax)
            elif dtype == torch.float64:
                return _ext.th_act_operators.LinearAct_double(vmin, vmax)
            else:
                raise RuntimeError('Unsupported dtype!')
        else:
            raise RuntimeError('Unsupported operator type!')

    @staticmethod
    def forward(ctx, x, weights, base_type, vmin, vmax):
        ctx.save_for_backward(x, weights)
        ctx.op = ActivationFunction._get_operator(x.dtype, base_type, vmin, vmax)
        return ctx.op.forward(x, weights)

    @staticmethod
    def backward(ctx, grad_in):
        x, weights = ctx.saved_tensors
        grad_x, grad_weights = ctx.op.adjoint(x, weights, grad_in)
        return grad_x, grad_weights, None, None, None

    @staticmethod
    def draw(weights, base_type, vmin, vmax):
        x = torch.linspace(2*vmin, 2*vmax, 1001, dtype=ctx.dtype).unsqueeze_(0)
        x = x.repeat(weights.shape[0], 1)
        f_x = ctx.op.forward(x.to(weights.device), weights)
        return x, f_x


class TrainableActivation(nn.Module):
    def __init__(self, num_channels, vmin, vmax, num_weights, base_type="rbf", init="linear", init_scale=1.0):
        super(TrainableActivation, self).__init__()

        self.num_channels = num_channels
        self.vmin = vmin
        self.vmax = vmax
        self.num_weights = num_weights
        self.base_type = base_type
        self.init = init
        self.init_scale = init_scale

        # setup the parameters of the layer
        self.weight = nn.Parameter(torch.Tensor(self.num_channels, self.num_weights))
        self.reset_parameters()

        # define the reduction index
        self.weight.reduction_dim = (1, )
        # possibly add a projection function
        # self.weight.proj = lambda _: pass

        # determine the operator
        if self.base_type in ["rbf", "linear"]:
            self.op = ActivationFunction
        else:
            raise RuntimeError("Unsupported base type '{}'!".format(base_type))

    def reset_parameters(self):
        # define the bins
        np_x = np.linspace(self.vmin, self.vmax, self.num_weights, dtype=np.float32)[np.newaxis, :]
        # initialize the weights
        if self.init == "linear":
            np_w = np_x * self.init_scale
        elif self.init == "student-t":
            alpha = 100
            np_w = self.init_scale * np.sqrt(alpha) * np_x / (1 + 0.5 * alpha * np_x ** 2)
        else:
            raise RuntimeError("Unsupported init type '{}'!".format(self.init))
        # tile to proper size
        np_w = np.tile(np_w, (self.num_channels, 1))

        self.weight.data = torch.from_numpy(np_w)

    def forward(self, x):
        # first reshape the input
        x = x.transpose(0, 1).contiguous()
        x_r = x.view(x.shape[0], -1)
        # compute the output
        x_r = self.op.apply(x_r, self.weight, self.base_type, self.vmin, self.vmax)
        return x_r.view(x.shape).transpose_(0, 1)

    def draw(self):
        return self.op.draw(self.weight, self.base_type, self.vmin, self.vmax)

    def extra_repr(self):
        s = "num_channels={num_channels}, num_weights={num_weights}, type={base_type}, vmin={vmin}, vmax={vmax}, init={init}"
        return s.format(**self.__dict__)


# to run execute: python -m unittest [-v] optoth.activations.act
class TestActivationFunction(unittest.TestCase):
    
    def _run_gradient_test(self, base_type):
        # setup the hyper parameters for each test
        Nw = 31
        vmin = -1.0
        vmax = 1
        dtype = np.float64
        C = 3

        np_x = np.linspace(vmin, vmax, Nw, dtype=dtype)
        np_w = np.tile(np_x[np.newaxis, :], (C, 1))

        # specify the functions
        np_w[0, :] = np_x
        np_w[1, :] = np_x ** 2
        np_w[2, :] = np.abs(np_x)

        np_x = np.linspace(2 * vmin, 2 * vmax, 1001, dtype=dtype)[np.newaxis, :]
        np_x = np.tile(np_x, (C, 1))

        # perform a gradient check:
        epsilon = 1e-4

        # prefactors
        a = 1.1
        b = 1.1

        # transfer to torch
        cuda = torch.device('cuda')
        th_x = torch.tensor(np_x, device=cuda)
        th_w = torch.tensor(np_w, device=cuda)
        th_a = torch.tensor(a, requires_grad=True, dtype=th_x.dtype, device=cuda)
        th_b = torch.tensor(b, requires_grad=True, dtype=th_w.dtype, device=cuda)
        op = ActivationFunction()

        # setup the model
        compute_loss = lambda a, b: 0.5 * torch.sum(op.apply(th_x*a, th_w*b, base_type, vmin, vmax)**2)
        th_loss = compute_loss(th_a, th_b)

        # backpropagate the gradient
        th_loss.backward()
        grad_a = th_a.grad.cpu().numpy()
        grad_b = th_b.grad.cpu().numpy()

        # numerical gradient w.r.t. the input
        with torch.no_grad():
            l_ap = compute_loss(th_a+epsilon, th_b).cpu().numpy()
            l_an = compute_loss(th_a-epsilon, th_b).cpu().numpy()
            grad_a_num = (l_ap - l_an) / (2 * epsilon)

        print("grad_x: {:.7f} num_grad_x {:.7f} success: {}".format(
            grad_a, grad_a_num, np.abs(grad_a - grad_a_num) < 1e-4))
        self.assertTrue(np.abs(grad_a - grad_a_num) < 1e-4)

        # numerical gradient w.r.t. the weights
        with torch.no_grad():
            l_bp = compute_loss(th_a, th_b+epsilon).cpu().numpy()
            l_bn = compute_loss(th_a, th_b-epsilon).cpu().numpy()
            grad_b_num = (l_bp - l_bn) / (2 * epsilon)

        print("grad_w: {:.7f} num_grad_w {:.7f} success: {}".format(
            grad_b, grad_b_num, np.abs(grad_b - grad_b_num) < 1e-4))
        self.assertTrue(np.abs(grad_b - grad_b_num) < 1e-4)
        
    def test_rbf_gradient(self):
        self._run_gradient_test("rbf")

    def test_linear_gradient(self):
        self._run_gradient_test("linear")


if __name__ == "__main__":
    unittest.test()
