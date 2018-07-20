import sys
sys.path.insert(0, '../../lib/python')

import PyAddOperator

import numpy as np

a = np.ones((10,), dtype=np.float32)
b = np.ones((10,), dtype=np.float32)

op = PyAddOperator.AddOperator()
op.set_config({"w_1": 1.0, "w_2": 1.0})

print(op.forward(a, b))
print(op.adjoint(a))

