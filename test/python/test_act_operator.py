import sys
sys.path.insert(0, '../../lib/python')

import PyActOperator

import numpy as np
import matplotlib.pyplot as plt

Nw = 31
vmin = -1. 
vmax = 1
dtype = np.float64
C = 3

x = np.linspace(vmin, vmax, Nw, dtype=dtype)
w = np.tile(x[np.newaxis, :], (C, 1))

w[0, :] = x
w[1, :] = x**2
w[2,: ] = np.abs(x)

x = np.linspace(2*vmin, 2*vmax, 1000, dtype=dtype)[np.newaxis, :]
x = np.tile(x, (C, 1))

# test the forward operator
if dtype == np.float32:
    op = PyActOperator.RBFActOperator_float(vmin, vmax)
else:
    op = PyActOperator.RBFActOperator_double(vmin, vmax)
f_x = op.forward(x, w)

plt.plot(x[0], f_x.T)
plt.savefig("./f.pdf")

# test the adjoint operator
grad_x, grad_w = op.adjoint(x, w, np.ones_like(x))

plt.figure()
plt.plot(x[0], grad_x.T)
plt.savefig("./grad_f.pdf")

# perform a gradient check:
epsilon = 1e-3
s_x = 1.4
s_w = 1.2

op_fwd = lambda x, s_x, w, s_w: op.forward(x*s_x, w*s_w)
loss = lambda x: 0.5 * np.sum(x**2)

# compute the loss
f_x = op_fwd(x, s_x, w, s_w)
l = loss(f_x)
# backpropagate the gradient
grad_l = f_x
grad_x, grad_w = op.adjoint(x*s_x, w*s_w, grad_l)
grad_s_x = np.sum(grad_x * x)
grad_s_w = np.sum(grad_w * w)

# first w.r.t. the input
l_xp = loss(op_fwd(x, s_x+epsilon, w, s_w))
l_xn = loss(op_fwd(x, s_x-epsilon, w, s_w))
grad_s_x_num = (l_xp - l_xn) / (2*epsilon)

# first w.r.t. the weights
l_xp = loss(op_fwd(x, s_x, w, s_w+epsilon))
l_xn = loss(op_fwd(x, s_x, w, s_w-epsilon))
grad_s_w_num = (l_xp - l_xn) / (2*epsilon)

print("grad_x: {:.7f} num_grad_x {:.7f} success: {}".format(grad_s_x, grad_s_x_num, np.abs(grad_s_x - grad_s_x_num) < 1e-4))
print("grad_w: {:.7f} num_grad_w {:.7f} success: {}".format(grad_s_w, grad_s_w_num, np.abs(grad_s_w - grad_s_w_num) < 1e-4))
