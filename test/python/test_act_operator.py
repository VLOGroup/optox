import sys
sys.path.insert(0, '../../lib/python')

import PyActOperator

import numpy as np

Nw = 31
vmin = -1. 
vmax = 1
dtype = np.float32
C = 3

x = np.linspace(vmin, vmax, Nw, dtype=dtype)
w = np.tile(x[np.newaxis, :], (C, 1))

w[0, :] = x**2
w[1, :] = x
w[2,: ] = np.abs(x)

x_test = np.linspace(2*vmin, 2*vmax, 100, dtype=dtype)[np.newaxis, :]

print(x.shape, w.shape, x_test.shape)

op = PyActOperator.RBFActOperator_float(vmin, vmax)
print(op)
f_x = op.forward(x_test, w)
print(f_x.shape)

# op = PyAddOperator.AddOperator()
