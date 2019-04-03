from __future__ import print_function

import os as _os
import sys as _sys
import numpy as np
import tensorflow as tf
import unittest
from tensorflow.python.framework import ops as _ops

_ext = tf.load_op_library(tf.resource_loader.get_path_to_datafile("tf_nabla_operator.so"))

__all__ = ['nabla_2d', 'nabla_2d_adjoint']

nabla_2d = _ext.nabla_operator
nabla_2d_adjoint = _ext.nabla_operator_adjoint

# to run execute: python -m unittest [-v] optotf.nabla
class TestNablaFunction(unittest.TestCase):
            
    def _test_adjointness(self, dtype, dim):
        # do not allocate entire memory for testing
        conf = tf.ConfigProto()
        conf.gpu_options.allow_growth=True
        tf.enable_eager_execution(config=conf)

        # setup the vaiables
        shape = [30 for i in range(dim)]
        tf_x = tf.random.normal(shape, dtype=dtype)
        shape.insert(0, dim)
        tf_p = tf.random.normal(shape, dtype=dtype)

        tf_nabla_x = nabla_2d(tf_x)
        tf_nablaT_p = nabla_2d_adjoint(tf_p)

        lhs = tf.reduce_sum(tf_nabla_x * tf_p)
        rhs = tf.reduce_sum(tf_x * tf_nablaT_p)

        print('dtype: {} dim: {} diff: {}'.format(dtype, dim, np.abs(lhs - rhs)))
        self.assertTrue(np.abs(lhs - rhs) < 1e-3)

    def test_float2_gradient(self):
        self._test_adjointness(tf.float32, 2)

    def test_double2_gradient(self):
        self._test_adjointness(tf.float64, 2)


if __name__ == "__main__":
    unittest.test()
