from __future__ import print_function

import os as _os
import sys as _sys
import numpy as np
import tensorflow as tf
import unittest
from tensorflow.python.framework import ops as _ops

_ext = tf.load_op_library(tf.compat.v1.resource_loader.get_path_to_datafile("tf_warp_operator.so"))

__all__ = ['warp_2d', 'warp_2d_transpose']

warp_2d = _ext.warp
warp_2d_transpose = _ext.warp_transpose

@_ops.RegisterGradient("Warp")
def _warp_forward_grad(op, grad):
    grad_in = warp_2d_transpose(grad,
                                op.inputs[1])
    return [grad_in, None]

@_ops.RegisterGradient("WarpTranspose")
def _warp_transpose_grad(op, grad):
    grad_in = warp_2d(grad,
                      op.inputs[1])
    return [grad_in, None]

class TestWarpFunction(unittest.TestCase):            
    def _test_adjointness(self, dtype):
        # do not allocate entire memory for testing
        # for gpu_device in tf.config.experimental.list_physical_devices('GPU'):
        #     tf.config.experimental.set_memory_growth(gpu_device, True)

        # setup the vaiables
        tf_x = tf.random.normal([10, 5, 20, 20,], dtype=dtype)
        tf_u = tf.random.normal([10, 20, 20, 2,],  dtype=dtype)*2
        tf_p = tf.random.normal([10, 5, 20, 20,], dtype=dtype)

        A = warp_2d
        AH = warp_2d_transpose

        tf_warp_x = A(tf_x, tf_u)
        tf_warpT_p = AH(tf_p, tf_u)

        lhs = tf.reduce_sum(tf_warp_x * tf_p)
        rhs = tf.reduce_sum(tf_x * tf_warpT_p)

        print('dtype: {} diff: {}'.format(dtype, np.abs(lhs - rhs)))
        self.assertTrue(np.abs(lhs - rhs) < 1e-3)

    def test_float2_gradient(self):
        self._test_adjointness(tf.float32)

    def test_double2_gradient(self):
        self._test_adjointness(tf.float64)


if __name__ == "__main__":
    unittest.test()
