import os

import numpy as np

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
    def draw(weights, base_type, vmin, vmax, scale=2):
        x = torch.linspace(scale*vmin, scale*vmax, 1001, dtype=weights.dtype).unsqueeze_(0)
        x = x.repeat(weights.shape[0], 1)
        op = ActivationFunction._get_operator(x.dtype, base_type, vmin, vmax)
        f_x = op.forward(x.to(weights.device), weights)
        return x, f_x


class Activation2Function(torch.autograd.Function):
    @staticmethod
    def _get_operator(dtype, base_type, vmin, vmax):
        if base_type == 'rbf':
            if dtype == torch.float32:
                return _ext.th_act_operators.RbfAct2_float(vmin, vmax)
            elif dtype == torch.float64:
                return _ext.th_act_operators.RbfAct2_double(vmin, vmax)
            else:
                raise RuntimeError('Unsupported dtype!')
        else:
            raise RuntimeError('Unsupported operator type!')

    @staticmethod
    def forward(ctx, x, weights, base_type, vmin, vmax):
        ctx.save_for_backward(x, weights)
        ctx.op = Activation2Function._get_operator(x.dtype, base_type, vmin, vmax)
        outputs = ctx.op.forward(x, weights)
        f, f_prime = outputs
        return f, f_prime

    @staticmethod
    def backward(ctx, grad_in, grad_in_prime):
        x, weights = ctx.saved_tensors
        grad_x, grad_weights = ctx.op.adjoint(x, weights, grad_in, grad_in_prime)
        return grad_x, grad_weights, None, None, None

    @staticmethod
    def draw(weights, base_type, vmin, vmax):
        x = torch.linspace(2*vmin, 2*vmax, 1001, dtype=weights.dtype).unsqueeze_(0)
        x = x.repeat(weights.shape[0], 1)
        op = Activation2Function._get_operator(x.dtype, base_type, vmin, vmax)
        f_x, f_prime_x = op.forward(x.to(weights.device), weights)
        return x, f_x, f_prime_x


def projection_onto_grad_bound(x, A, gmin, gmax, num_max_iter=3000, stopping_value=1e-8):
    # compute the Lipschitz constant
    L = np.linalg.norm(np.dot(A, A.T), 2)

    # transform the input to numpy
    np_x = x.detach().cpu().numpy().T

    # define the Lagrange duals
    np_l = np.zeros((A.shape[0], np_x.shape[1]), dtype=np.float32)
    # use Fista
    np_l_old = np_l.copy()

    for k in range(1,num_max_iter+1):
        beta = 0.#(k - 1) / (k + 2)
        np_l_hat = np_l + beta * (np_l - np_l_old)
        np_l_old = np_l.copy()

        np_x_old = np_x.copy()
        np_x = np_x - np.dot(A.T, np_l_hat)
        grad_l = np.dot(A, -np_x)
        np_l_hat = np_l_hat - grad_l/L
        np_l = np_l_hat - 1./L * np.maximum(gmin, np.minimum(gmax, np_l_hat * L))

        np_diff = np.sqrt(np.mean((np_x_old - np_x) ** 2))

        if k > 1 and np_diff < stopping_value:
            break

    if np_diff > stopping_value:
        print('   Projection onto linear constraints: %d/%d iterations' % (k, num_max_iter))

    x.data = torch.as_tensor(np_x.T, dtype=x.dtype, device=x.device)

class TrainableActivation(nn.Module):
    def __init__(self, num_channels, vmin, vmax, num_weights, base_type="rbf", init="linear", init_scale=1.0,
                 group=1, bmin=None, bmax=None, gmin=None, gmax=None, symmetric=False):
        super(TrainableActivation, self).__init__()

        self.num_channels = num_channels
        self.vmin = vmin
        self.vmax = vmax
        self.num_weights = num_weights
        self.base_type = base_type
        self.init = init
        self.init_scale = init_scale
        self.group = group
        self.bmin = bmin
        self.bmax = bmax
        self.gmin = gmin
        self.gmax = gmax
        self.symmetric = symmetric

        # setup the parameters of the layer
        self.weight = nn.Parameter(torch.Tensor(self.num_channels, self.num_weights))
        self.reset_parameters()

        # define the reduction index
        self.weight.reduction_dim = (1, )
        # possibly add a projection function
        # self.weight.proj = lambda _: pass

        if self.symmetric:
            # first construct the symmetry matrix
            I = np.eye(self.num_weights // 2, dtype=np.float32)
            I_a = I[:, ::-1]
            M_t = np.concatenate([I, np.zeros((self.num_weights // 2, 1), dtype=np.float32), I_a], axis=1)
            M_b = np.concatenate([I_a, np.zeros((self.num_weights // 2, 1), dtype=np.float32), I], axis=1)
            M = np.concatenate([M_t, np.zeros((1, self.num_weights), dtype=np.float32), M_b])
            M[self.num_weights // 2, self.num_weights // 2] = 2
            self.register_buffer('M', torch.tensor(M))

            # apply the symmetry constraint
            self.weight.proj = lambda: self.weight.data.sub_(self.weight.data @ self.M / 2)

        elif bmin is not None or bmax is not None or gmin is not None or gmax is not None:
            if bmin is None:
                bmin = -np.Inf
            if bmax is None:
                bmax = np.Inf
            if gmin is None:
                gmin = -np.Inf
            if gmax is None:
                gmax = np.Inf

            if self.base_type == 'linear':
                # define the constraint
                delta_x = (self.vmax - self.vmin) / (self.num_weights - 1)
                eye = np.eye(self.num_weights)
                forward_differences = (np.diag(-np.ones((self.num_weights,)), k=0)[:-1, :] +
                                       np.diag(np.ones(self.num_weights - 1, ), k=1)[:-1, :]) / delta_x
                A = np.vstack((eye, forward_differences))
                lower_bound = np.vstack((
                    bmin * np.ones((self.num_weights, self.num_channels), dtype=np.float32),
                    gmin * np.ones((self.num_weights-1, self.num_channels), dtype=np.float32)
                ))
                upper_bound = np.vstack((
                    bmax * np.ones((self.num_weights, self.num_channels), dtype=np.float32),
                    gmax * np.ones((self.num_weights-1, self.num_channels), dtype=np.float32)
                ))
                self.weight.proj = lambda: projection_onto_grad_bound(self.weight, A, 
                                lower_bound, upper_bound)
            else:
                raise RuntimeError("Gradient bound not supported for base type: '{}'".format(self.base_type))

        # determine the operator
        if self.base_type in ["rbf", "linear"]:
            self.op = ActivationFunction
        else:
            raise RuntimeError("Unsupported base type '{}'!".format(base_type))

    def reset_parameters(self):
        # define the bins
        np_x = np.linspace(self.vmin, self.vmax, self.num_weights, dtype=np.float32)[np.newaxis, :]
        # initialize the weights
        if self.init == "constant":
            np_w = np.ones_like(np_x) * self.init_scale
        elif self.init == "linear":
            np_w = np_x * self.init_scale
        elif self.init == "quadratic":
            np_w = np_x**2 * self.init_scale
        elif self.init == "student-t":
            alpha = 100
            np_w = self.init_scale * np.sqrt(alpha) * np_x / (1 + 0.5 * alpha * np_x ** 2)
        elif self.init == "invert":
            np_w = self.init_scale / np_x
            if not np.all(np.isfinite(np_w)):
                raise RuntimeError("Invalid value encountered in weight init!")
        else:
            raise RuntimeError("Unsupported init type '{}'!".format(self.init))
        # tile to proper size
        np_w = np.tile(np_w, (self.num_channels, 1))

        self.weight.data = torch.from_numpy(np_w)

    def forward(self, x):
        # first reshape the input
        x = x.transpose(0, 1).contiguous()
        if x.shape[0] % self.group != 0:
            raise RuntimeError("Input shape must be a multiple of group!") 
        x_r = x.view(x.shape[0]//self.group, -1)
        # compute the output
        x_r = self.op.apply(x_r, self.weight, self.base_type, self.vmin, self.vmax)
        return x_r.view(x.shape).transpose_(0, 1)

    def draw(self, scale=2):
        return self.op.draw(self.weight, self.base_type, self.vmin, self.vmax, scale=scale)

    def extra_repr(self):
        s = "num_channels={num_channels}, num_weights={num_weights}, type={base_type}, vmin={vmin}, vmax={vmax}, init={init}, init_scale={init_scale}"
        s += " group={group}, bmin={bmin}, bmax={bmax}, gmin={gmin}, gmax={gmax}, symmetric={symmetric}"
        return s.format(**self.__dict__)


class TrainableActivation2(TrainableActivation):
    def __init__(self, num_channels, vmin, vmax, num_weights, base_type="rbf", init="linear", init_scale=1.0):
        super(TrainableActivation2, self).__init__(num_channels, vmin, vmax, num_weights, base_type, init, init_scale)

        # determine the operator
        if self.base_type in ["rbf"]:
            self.op = Activation2Function
        else:
            raise RuntimeError("Unsupported base type '{}'!".format(base_type))

    def forward(self, x):
        # first reshape the input
        x = x.transpose(0, 1).contiguous()
        x_r = x.view(x.shape[0], -1)
        # compute the output
        ret = self.op.apply(x_r, self.weight, self.base_type, self.vmin, self.vmax)
        f_r = ret[0]
        f_prime_r = ret[1]
        return f_r.view(x.shape).transpose_(0, 1), f_prime_r.view(x.shape).transpose_(0, 1)

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


class TestActivation2Function(unittest.TestCase):
    
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
        np_w[2, :] = np.sin(np_x*np.pi)

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
        op = Activation2Function()

        # setup the model
        def compute_loss(a, b):
            f, f_prime = op.apply(th_x*a, th_w*b, base_type, vmin, vmax)
            return 0.5 * torch.sum(f**2 + f_prime**2)
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


if __name__ == "__main__":
    unittest.test()
