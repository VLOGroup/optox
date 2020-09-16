from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import ops as _ops
import unittest
import numpy as np

__all__ = ['pad3d', 'pad3d_transpose']

# load operators from the library
_ext = tf.load_op_library(tf.compat.v1.resource_loader.get_path_to_datafile("tf_pad3d_operator.so"))

def pad3d(x, pad_x, pad_y, pad_z, mode):
    #assert tf.rank(x) == 5

    # first reshape the input
    #if channel_last:
    x = tf.transpose(x, [0, 4, 1, 2, 3])
        
    shape = list(x.shape)
    x_r = tf.reshape(x, (-1, *shape[2:]))

    # compute the output
    x_r = _ext.pad3d(x_r,  pad_x=pad_x, pad_y=pad_y, pad_z=pad_z, mode=mode)

    padded_shape = shape
    padded_shape[2] += 2*pad_z
    padded_shape[3] += 2*pad_y
    padded_shape[4] += 2*pad_x

    #if channel_last:
    return tf.transpose(tf.reshape(x_r, padded_shape), [0, 2, 3, 4, 1])
    #else:
    #    return tf.reshape(x_r, padded_shape)

def pad3d_transpose(x, pad_x, pad_y, pad_z, mode):
    #assert tf.rank(x) == 5

    # first reshape the input
    #if channel_last:
    x = tf.transpose(x, [0, 4, 1, 2, 3])

    shape = list(x.shape)
    x_r = tf.reshape(x, (-1, *shape[2:]))

    # compute the output
    x_r = _ext.pad3d_transpose(x_r,  pad_x=pad_x, pad_y=pad_y, pad_z=pad_z, mode=mode)

    paddedT_shape = shape
    paddedT_shape[2] -= 2*pad_z
    paddedT_shape[3] -= 2*pad_y
    paddedT_shape[4] -= 2*pad_x

    #if channel_last:
    return tf.transpose(tf.reshape(x_r, paddedT_shape), [0, 2, 3, 4, 1])
    #else:
    #    return tf.reshape(x_r, paddedT_shape)

@_ops.RegisterGradient("Pad3d")
def _pad3d_grad(op, grad):
    grad_x = _ext.pad3d_transpose(
        grad,
        pad_x=op.get_attr("pad_x"), 
        pad_y=op.get_attr("pad_y"),
        pad_z=op.get_attr("pad_z"),
        mode=op.get_attr("mode"))
    return [grad_x]  

@_ops.RegisterGradient("Pad3dTranspose")
def _pad3d_transpose_grad(op, grad):
    grad_x = _ext.pad3d(
        grad,
        pad_x=op.get_attr("pad_x"), 
        pad_y=op.get_attr("pad_y"),
        pad_z=op.get_attr("pad_z"),
        mode=op.get_attr("mode"))
    return [grad_x]  

# to run execute: python -m unittest [-v] optotf.pad3d
class TestFunction(unittest.TestCase):
    def _test_adjointness(self, base_type):
        tf_dtype = tf.float64

        pad_x = 4
        pad_y = 3
        pad_z = 2

        N = 5
        D = 10
        H = 20
        W = 40

        # determine the operator
        A = _ext.pad3d
        AH = _ext.pad3d_transpose

        np_x = np.random.randn(N, D, H, W)
        np_y = np.random.randn(N, D+2*pad_z, H+2*pad_y, W+2*pad_x)

        # transfer to tensorflow
        tf_x = tf.convert_to_tensor(np_x, tf_dtype)
        tf_y = tf.convert_to_tensor(np_y, tf_dtype)

        # perform fwd/adj
        tf_Ax = A(tf_x, pad_x=pad_x, pad_y=pad_y, pad_z=pad_z, mode=base_type)
        tf_AHy = AH(tf_y, pad_x=pad_x, pad_y=pad_y, pad_z=pad_z, mode=base_type)

        # adjointness check
        lhs = tf.reduce_sum(tf_Ax * tf_y)
        rhs = tf.reduce_sum(tf_AHy * tf_x)
        
        print('adjointness diff: {}'.format(np.abs(lhs - rhs)))
        self.assertTrue(np.abs(lhs - rhs) < 1e-5)

    def test_symmetric_adjointness(self):
        self._test_adjointness("symmetric")

    def test_reflect_adjointness(self):
        self._test_adjointness("reflect")

    def test_replicate_adjointness(self):
        self._test_adjointness("replicate")


if __name__ == "__main__":
    unittest.test()