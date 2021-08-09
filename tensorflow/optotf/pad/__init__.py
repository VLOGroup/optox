from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import ops as _ops
import unittest
import numpy as np

__all__ = ['pad2d', 'pad2d_transpose', 'pad3d', 'pad3d_transpose']

# load operators from the library
_ext = tf.load_op_library(tf.compat.v1.resource_loader.get_path_to_datafile("tf_pad_operator.so"))

def pad2d(x, padding, mode, channel_last=True):
    """Padding of a 2d tensor.
    
    This function pads a 2d tensor (rank 4). The tensorformat is either
    [N, C, H, W] for `channel_last=False` or [N, H, W, C] for `channel_last=True`. The tensor is
    padded by values specified in `padding` [W0, W1, H0, H1] where 0 indicates the padding before and 1
    indicates the padding after. This functions supports the padding modes "reflect", "symmetric" and "replicate".

    Args:
        tensor: A `Tensor`.
        padding: A `Tensor` of type `int32`.
        mode: One of "reflect", "symmetric", or "replicate" (case-insensitive)
        channel_last: 
    Returns:
        A padded `Tensor`. Has the same type as `tensor`.
    """

    # first reshape the input
    if channel_last:
        x = tf.transpose(x, [0, 3, 1, 2])
        
    shape = tf.unstack(tf.shape(x))
    new_shape = [-1, *shape[2:]]
    new_shape = tf.stack(new_shape)
    x_r = tf.reshape(x, new_shape)

    # compute the output
    x_r = _ext.pad2d(x_r, left=padding[0], right=padding[1], bottom=padding[2], top=padding[3], mode=mode)

    padded_shape = shape
    padded_shape[-2] += padding[2] + padding[3]
    padded_shape[-1] += padding[0] + padding[1]
    padded_shape = tf.stack(padded_shape)

    if channel_last:
        return tf.transpose(tf.reshape(x_r, padded_shape), [0, 2, 3, 1])
    else:
        return tf.reshape(x_r, padded_shape)

def pad2d_transpose(x, padding, mode, channel_last=True):
    """Transpose padding of a 2d tensor.
    
    This function transpose pads a 2d tensor (rank 4). The tensorformat is either
    [N, C, D, H, W] for `channel_last=False` or [N, D, H, W, C] for `channel_last=True`. The tensor is
    padded by values specified in `padding` [W0, W1, H0, H1, D0, D1] where 0 indicates the padding before and 1
    indicates the padding after. This functions supports the padding modes "reflect", "symmetric" and "replicate".

    Args:
        tensor: A `Tensor`.
        padding: A `Tensor` of type `int32`.
        mode: One of "reflect", "symmetric", or "replicate" (case-insensitive)
        channel_last: 
    Returns:
        A transposed padded `Tensor`. Has the same type as `tensor`.
    """
    # first reshape the input
    if channel_last:
        x = tf.transpose(x, [0, 3, 1, 2])

    shape = tf.unstack(tf.shape(x))
    new_shape = [-1, *shape[2:]]
    new_shape = tf.stack(new_shape)
    x_r = tf.reshape(x, new_shape)

    # compute the output
    x_r = _ext.pad2d_transpose(x_r, left=padding[0], right=padding[1], bottom=padding[2], top=padding[3], mode=mode)

    paddedT_shape = shape
    paddedT_shape[-2] -= padding[2] + padding[3]
    paddedT_shape[-1] -= padding[0] + padding[1]
    paddedT_shape = tf.stack(paddedT_shape)

    if channel_last:
        return tf.transpose(tf.reshape(x_r, paddedT_shape), [0, 2, 3, 1])
    else:
        return tf.reshape(x_r, paddedT_shape)

@_ops.RegisterGradient("Pad2d")
def _pad2d_grad(op, grad):
    grad_x = _ext.pad2d_transpose(
        grad,
        left=op.get_attr("left"), 
        right=op.get_attr("right"),
        bottom=op.get_attr("bottom"),
        top=op.get_attr("top"),
        mode=op.get_attr("mode"))
    return [grad_x]  

@_ops.RegisterGradient("Pad2dTranspose")
def _pad2d_transpose_grad(op, grad):
    grad_x = _ext.pad2d(
        grad,
        left=op.get_attr("left"), 
        right=op.get_attr("right"),
        bottom=op.get_attr("bottom"),
        top=op.get_attr("top"),
        mode=op.get_attr("mode"))
    return [grad_x]  


def pad3d(x, padding, mode, channel_last=True):
    """Padding of a 3d tensor.
    
    This function pads a 3d tensor (rank 5). The tensorformat is either
    [N, C, D, H, W] for `channel_last=False` or [N, D, H, W, C] for `channel_last=True`. The tensor is
    padded by values specified in `padding` [W0, W1, H0, H1, D0, D1] where 0 indicates the padding before and 1
    indicates the padding after. This functions supports the padding modes "reflect", "symmetric" and "replicate".

    Args:
        tensor: A `Tensor`.
        padding: A `Tensor` of type `int32`.
        mode: One of "reflect", "symmetric", or "replicate" (case-insensitive)
        channel_last: 
    Returns:
        A padded `Tensor`. Has the same type as `tensor`.
    """

    # first reshape the input
    if channel_last:
        x = tf.transpose(x, [0, 4, 1, 2, 3])
        
    shape = tf.unstack(tf.shape(x))
    new_shape = [-1, *shape[2:]]
    new_shape = tf.stack(new_shape)
    x_r = tf.reshape(x, new_shape)

    # compute the output
    x_r = _ext.pad3d(x_r, left=padding[0], right=padding[1], bottom=padding[2], top=padding[3], front=padding[4], back=padding[5], mode=mode)

    padded_shape = shape
    padded_shape[-3] += padding[4] + padding[5]
    padded_shape[-2] += padding[2] + padding[3]
    padded_shape[-1] += padding[0] + padding[1]
    padded_shape = tf.stack(padded_shape)

    if channel_last:
        return tf.transpose(tf.reshape(x_r, padded_shape), [0, 2, 3, 4, 1])
    else:
        return tf.reshape(x_r, padded_shape)

def pad3d_transpose(x, padding, mode, channel_last=True):
    """Transpose padding of a 3d tensor.
    
    This function transpose pads a 3d tensor (rank 5). The tensorformat is either
    [N, C, D, H, W] for `channel_last=False` or [N, D, H, W, C] for `channel_last=True`. The tensor is
    padded by values specified in `padding` [W0, W1, H0, H1, D0, D1] where 0 indicates the padding before and 1
    indicates the padding after. This functions supports the padding modes "reflect", "symmetric" and "replicate".

    Args:
        tensor: A `Tensor`.
        padding: A `Tensor` of type `int32`.
        mode: One of "reflect", "symmetric", or "replicate" (case-insensitive)
        channel_last: 
    Returns:
        A transposed padded `Tensor`. Has the same type as `tensor`.
    """
    # first reshape the input
    if channel_last:
        x = tf.transpose(x, [0, 4, 1, 2, 3])

    shape = tf.unstack(tf.shape(x))
    new_shape = [-1, *shape[2:]]
    new_shape = tf.stack(new_shape)
    x_r = tf.reshape(x, new_shape)

    # compute the output
    x_r = _ext.pad3d_transpose(x_r, left=padding[0], right=padding[1], bottom=padding[2], top=padding[3], front=padding[4], back=padding[5], mode=mode)

    paddedT_shape = shape
    paddedT_shape[-3] -= padding[4] + padding[5]
    paddedT_shape[-2] -= padding[2] + padding[3]
    paddedT_shape[-1] -= padding[0] + padding[1]
    paddedT_shape = tf.stack(paddedT_shape)

    if channel_last:
        return tf.transpose(tf.reshape(x_r, paddedT_shape), [0, 2, 3, 4, 1])
    else:
        return tf.reshape(x_r, paddedT_shape)

@_ops.RegisterGradient("Pad3d")
def _pad3d_grad(op, grad):
    grad_x = _ext.pad3d_transpose(
        grad,
        left=op.get_attr("left"), 
        right=op.get_attr("right"),
        bottom=op.get_attr("bottom"),
        top=op.get_attr("top"),
        front=op.get_attr("front"),
        back=op.get_attr("back"),
        mode=op.get_attr("mode"))
    return [grad_x]  

@_ops.RegisterGradient("Pad3dTranspose")
def _pad3d_transpose_grad(op, grad):
    grad_x = _ext.pad3d(
        grad,
        left=op.get_attr("left"), 
        right=op.get_attr("right"),
        bottom=op.get_attr("bottom"),
        top=op.get_attr("top"),
        front=op.get_attr("front"),
        back=op.get_attr("back"),
        mode=op.get_attr("mode"))
    return [grad_x]  

# to run execute: python -m unittest [-v] optotf.pad3d
class TestFunction2d(unittest.TestCase):
    def _test_adjointness(self, base_type):
        tf_dtype = tf.float64

        pad_x = 4
        pad_y = 3

        N = 5
        H = 20
        W = 40

        # determine the operator
        A = _ext.pad2d
        AH = _ext.pad2d_transpose

        np_x = np.random.randn(N, H, W)
        np_y = np.random.randn(N, H+2*pad_y, W+2*pad_x)

        # transfer to tensorflow
        tf_x = tf.convert_to_tensor(np_x, tf_dtype)
        tf_y = tf.convert_to_tensor(np_y, tf_dtype)

        # perform fwd/adj
        tf_Ax = A(tf_x, 
                  left=pad_x, 
                  right=pad_x,
                  bottom=pad_y,
                  top=pad_y,
                  mode=base_type)
        tf_AHy = AH(tf_y,
                  left=pad_x, 
                  right=pad_x,
                  bottom=pad_y,
                  top=pad_y,
                  mode=base_type)

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


class TestFunction3d(unittest.TestCase):
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
        tf_Ax = A(tf_x, 
                  left=pad_x, 
                  right=pad_x,
                  bottom=pad_y,
                  top=pad_y,
                  front=pad_z,
                  back=pad_z,
                  mode=base_type)
        tf_AHy = AH(tf_y,
                  left=pad_x, 
                  right=pad_x,
                  bottom=pad_y,
                  top=pad_y,
                  front=pad_z,
                  back=pad_z,
                  mode=base_type)

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