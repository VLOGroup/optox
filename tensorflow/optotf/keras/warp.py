import tensorflow as tf
import optotf.warp
import unittest

class Warp(tf.keras.layers.Layer):
    def __init__(self, channel_last=True):
        super().__init__()
        self.channel_last = channel_last
        self.op = optotf.warp.warp_2d
    
    def call(self, x, u):
        if self.channel_last:
            x = tf.transpose(x, [0, 3, 1, 2])

        if x.dtype == tf.complex64 or x.dtype == tf.complex128:
            out = tf.complex(self.op(tf.math.real(x), u), self.op(tf.math.imag(x), u))
        else:
            out = self.op(x, u)
        
        if self.channel_last:
            out = tf.transpose(out, [0, 2, 3, 1])

        return out

class WarpTranspose(tf.keras.layers.Layer):
    def __init__(self, channel_last=True):
        super().__init__()
        self.channel_last = channel_last
        self.op = optotf.warp.warp_2d_transpose
    
    def call(self, grad_out, u):
        if self.channel_last:
            grad_out = tf.transpose(grad_out, [0, 3, 1, 2])

        if grad_out.dtype == tf.complex64 or grad_out.dtype == tf.complex128:
            out = tf.complex(self.op(tf.math.real(grad_out), u), self.op(tf.math.imag(grad_out), u))
        else:
            out = self.op(grad_out, u)
        
        if self.channel_last:
            out = tf.transpose(out, [0, 2, 3, 1])
            
        return out

class TestWarp(unittest.TestCase):
    def test_warp_forward_channelfirst(self):
        x = tf.random.normal((10, 2, 20, 20))
        u = tf.random.normal((10, 20, 20, 2))*10.0
        op = Warp(channel_last=False)
        Kx = op(x, u)

    def test_warp_transpose_channelfirst(self):
        x = tf.random.normal((10, 2, 20, 20))
        u = tf.random.normal((10, 20, 20, 2))*10.0
        op = WarpTranspose(channel_last=False)
        Kx = op(x, u)

    def test_warp_forward(self):
        x = tf.random.normal((10, 20, 20, 2))
        u = tf.random.normal((10, 20, 20, 2))*10.0
        op = Warp(channel_last=True)
        Kx = op(x, u)

    def test_warp_transpose(self):
        x = tf.random.normal((10, 20, 20, 2))
        u = tf.random.normal((10, 20, 20, 2))*10.0
        op = WarpTranspose(channel_last=True)
        Kx = op(x, u)

    def test_warp_forward_complex(self):
        x = tf.complex(tf.random.normal((10, 20, 20, 2)), tf.random.normal((10, 20, 20, 2)))
        u = tf.random.normal((10, 20, 20, 2))*10.0
        op = Warp(channel_last=True)
        Kx = op(x, u)

    def test_warp_transpose_complex(self):
        x = tf.complex(tf.random.normal((10, 20, 20, 2)), tf.random.normal((10, 20, 20, 2)))
        u = tf.random.normal((10, 20, 20, 2))*10.0
        op = WarpTranspose(channel_last=True)
        Kx = op(x, u)