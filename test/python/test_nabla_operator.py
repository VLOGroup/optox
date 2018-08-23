import sys
sys.path.insert(0, '../../lib/python')

import PyNablaOperator

import numpy as np

nabla_op = PyNablaOperator.NablaOperator()

x_rand = np.random.randn(64, 64).astype(np.float32)
y_rand = np.random.randn(2, 64, 64).astype(np.float32)

Ax = nabla_op.forward(x_rand)
ATy = nabla_op.adjoint(y_rand)
lhs = np.sum(Ax * y_rand)
rhs = np.sum(x_rand * ATy)
print(lhs, rhs, np.abs(lhs - rhs))
