import sys
sys.path.insert(0, '../../lib/python')

import PyActOperator

import numpy as np
import matplotlib.pyplot as plt

Nw = 63
vmin = -1. 
vmax = 1
dtype = np.float32
C = 3

x = np.linspace(vmin, vmax, Nw, dtype=dtype)
w = np.tile(x[np.newaxis, :], (C, 1))

w[0, :] = x**2
w[1, :] = x
w[2,: ] = np.abs(x)

x_test = np.linspace(2*vmin, 2*vmax, 1000, dtype=dtype)[np.newaxis, :]
x_test = np.tile(x_test, (C, 1))

# test the forward operator
op = PyActOperator.RBFActOperator_float(vmin, vmax)
f_x = op.forward(x_test, w)

plt.plot(x_test[0], f_x.T)
plt.savefig("./f.pdf")

# test the adjoint operator
grad_x, grad_w = op.adjoint(x_test, w, np.ones_like(x_test))
print(grad_x, grad_w)

plt.figure()
plt.plot(x_test[0], grad_x.T)
plt.savefig("./grad_f.pdf")
