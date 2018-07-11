import sys
sys.path.insert(0, '../../lib/python')

import PyAddOperator

import numpy as np

a = np.ones((10,), dtype=np.float32)
b = np.ones((10,), dtype=np.float32)

c = np.ones((10,), dtype=np.float32)

op = PyAddOperator.AddOperator()
op.append_input(a)
op.append_input(b)
op.append_output(c)

op.set_config({"w_1": 1.0, "w_2": 1.0})

#op.set_input(0, a)

op.apply()

print(op.get_output(0))

