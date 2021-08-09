import tensorflow as tf
import optotf.pad
import unittest

class Pad2d(tf.keras.layers.Layer):
    def __init__(self, padding, mode, channel_last=True):
        super().__init__()
        self.padding = padding
        self.mode = mode
        self.channel_last = channel_last
        self.op = optotf.pad._ext.pad2d

    def build(self, input_shape):
        shape = tf.unstack(input_shape)

        if self.channel_last:
            shape = [shape[0], shape[-1], *shape[1:-1]]

        new_shape = [-1, *shape[2:]]
        new_shape = tf.stack(new_shape)

        padded_shape = shape
        padded_shape[-2] += self.padding[2] + self.padding[3]
        padded_shape[-1] += self.padding[0] + self.padding[1]
        padded_shape = tf.stack(padded_shape)

        self.pre_pad_shape = new_shape
        self.post_pad_shape = padded_shape


    def call(self, x):
        # first reshape the input
        if self.channel_last:
            x = tf.transpose(x, [0, 3, 1, 2])

        x_r = tf.reshape(x, self.pre_pad_shape, self.post_pad_shape)

        if x.dtype == tf.complex64 or x.dtype == tf.complex128:
            x_r = tf.complex(self.op(tf.math.real(x_r), left=self.padding[0], right=self.padding[1], bottom=self.padding[2], top=self.padding[3], mode=self.mode), 
                            self.op(tf.math.imag(x_r), left=self.padding[0], right=self.padding[1], bottom=self.padding[2], top=self.padding[3], mode=self.mode))
        else:
            x_r = self.op(x_r, left=self.padding[0], right=self.padding[1], bottom=self.padding[2], top=self.padding[3], mode=self.mode)

        if self.channel_last:
            return tf.transpose(tf.reshape(x_r, self.post_pad_shape), [0, 2, 3, 1])
        else:
            return tf.reshape(x_r, self.post_pad_shape)

class Pad2dTranspose(Pad2d):
    def __init__(self, padding, mode, channel_last=True):
        super().__init__(padding, mode, channel_last=channel_last)
        self.op = optotf.pad._ext.pad2d_transpose

    def build(self, input_shape):
        shape = tf.unstack(input_shape)

        if self.channel_last:
            shape = [shape[0], shape[-1], *shape[1:-1]]

        new_shape = [-1, *shape[2:]]
        new_shape = tf.stack(new_shape)

        padded_shape = shape
        padded_shape[-2] -= self.padding[2] + self.padding[3]
        padded_shape[-1] -= self.padding[0] + self.padding[1]
        padded_shape = tf.stack(padded_shape)

        self.pre_pad_shape = new_shape
        self.post_pad_shape = padded_shape
class Pad3d(tf.keras.layers.Layer):
    def __init__(self, padding, mode, channel_last=True):
        super().__init__()
        self.padding = padding
        self.mode = mode
        self.channel_last = channel_last
        self.op = optotf.pad._ext.pad3d

    def build(self, input_shape):
        shape = tf.unstack(input_shape)

        if self.channel_last:
            shape = [shape[0], shape[-1], *shape[1:-1]]

        new_shape = [-1, *shape[2:]]
        new_shape = tf.stack(new_shape)

        padded_shape = shape
        padded_shape[-3] += self.padding[4] + self.padding[5]
        padded_shape[-2] += self.padding[2] + self.padding[3]
        padded_shape[-1] += self.padding[0] + self.padding[1]
        padded_shape = tf.stack(padded_shape)

        self.pre_pad_shape = new_shape
        self.post_pad_shape = padded_shape


    def call(self, x):
        # first reshape the input
        if self.channel_last:
            x = tf.transpose(x, [0, 2, 3, 4, 1])

        x_r = tf.reshape(x, self.pre_pad_shape, self.post_pad_shape)

        if x.dtype == tf.complex64 or x.dtype == tf.complex128:
            x_r = tf.complex(self.op(tf.math.real(x_r), left=self.padding[0], right=self.padding[1], bottom=self.padding[2], top=self.padding[3], front=self.padding[4], back=self.padding[5], mode=self.mode), 
                            self.op(tf.math.imag(x_r), left=self.padding[0], right=self.padding[1], bottom=self.padding[2], top=self.padding[3], front=self.padding[4], back=self.padding[5], mode=self.mode))
        else:
            x_r = self.op(x_r, left=self.padding[0], right=self.padding[1], bottom=self.padding[2], top=self.padding[3], front=self.padding[4], back=self.padding[5], mode=self.mode)

        if self.channel_last:
            return tf.transpose(tf.reshape(x_r, self.post_pad_shape), [0, 2, 3, 4, 1])
        else:
            return tf.reshape(x_r, self.post_pad_shape)

class Pad3dTranspose(Pad3d):
    def __init__(self, padding, mode, channel_last=True):
        super().__init__(padding, mode, channel_last=channel_last)
        self.op = optotf.pad._ext.pad3d_transpose

    def build(self, input_shape):
        shape = tf.unstack(input_shape)

        if self.channel_last:
            shape = [shape[0], shape[-1], *shape[1:-1]]

        new_shape = [-1, *shape[2:]]
        new_shape = tf.stack(new_shape)

        padded_shape = shape
        padded_shape[-3] -= self.padding[4] + self.padding[5]
        padded_shape[-2] -= self.padding[2] + self.padding[3]
        padded_shape[-1] -= self.padding[0] + self.padding[1]
        padded_shape = tf.stack(padded_shape)

        self.pre_pad_shape = new_shape
        self.post_pad_shape = padded_shape
class TestPad(unittest.TestCase):
    def test2d(self):
        shape = (5, 2, 10, 10)
        x = tf.random.normal(shape)
        padding = [2, 2, 4, 4]
        op = Pad2d(padding=padding, mode='symmetric', channel_last=False)
        Kx = op(x)

        # manually construct new shape
        new_shape = list(x.shape)
        new_shape[-1] += padding[0] + padding[1]
        new_shape[-2] += padding[2] + padding[3]
        new_shape = tuple(new_shape)

        self.assertTrue(new_shape == Kx.shape)

    def test2d_complex(self):
        shape = (5, 2, 10, 10)
        x = tf.complex(tf.random.normal(shape), tf.random.normal(shape))
        padding = [2, 2, 4, 4]
        op = Pad2d(padding=padding, mode='symmetric', channel_last=False)
        Kx = op(x)

        # manually construct new shape
        new_shape = list(x.shape)
        new_shape[-1] += padding[0] + padding[1]
        new_shape[-2] += padding[2] + padding[3]
        new_shape = tuple(new_shape)

        self.assertTrue(new_shape == Kx.shape)

    def test2d_channel_last(self):
        shape = (5, 10, 10, 2)
        x = tf.random.normal(shape)
        padding = [2, 2, 4, 4]
        op = Pad2d(padding=padding, mode='symmetric', channel_last=True)
        Kx = op(x)

        # manually construct new shape
        new_shape = list(x.shape)
        new_shape[-2] += padding[0] + padding[1]
        new_shape[-3] += padding[2] + padding[3]
        new_shape = tuple(new_shape)

        self.assertTrue(new_shape == Kx.shape)

    def test3d(self):
        shape = (5, 2, 8, 10, 10)
        x = tf.random.normal(shape)
        padding = [2, 2, 4, 4, 1, 1]
        op = Pad3d(padding=padding, mode='symmetric', channel_last=False)
        Kx = op(x)

        # manually construct new shape
        new_shape = list(x.shape)
        new_shape[-1] += padding[0] + padding[1]
        new_shape[-2] += padding[2] + padding[3]
        new_shape[-3] += padding[4] + padding[5]
        new_shape = tuple(new_shape)

        self.assertTrue(new_shape == Kx.shape)

    def test3d_complex(self):
        shape = (5, 2, 8, 10, 10)
        x = tf.complex(tf.random.normal(shape), tf.random.normal(shape))
        padding = [2, 2, 4, 4, 1, 1]
        op = Pad3d(padding=padding, mode='symmetric', channel_last=False)
        Kx = op(x)

        # manually construct new shape
        new_shape = list(x.shape)
        new_shape[-1] += padding[0] + padding[1]
        new_shape[-2] += padding[2] + padding[3]
        new_shape[-3] += padding[4] + padding[5]
        new_shape = tuple(new_shape)

        self.assertTrue(new_shape == Kx.shape)

    def test3d_channel_last(self):
        shape = (5, 8, 10, 10, 2)
        x = tf.random.normal(shape)
        padding = [2, 2, 4, 4, 1, 2]
        op = Pad3d(padding=padding, mode='symmetric', channel_last=True)
        Kx = op(x)

        # manually construct new shape
        new_shape = list(x.shape)
        new_shape[-2] += padding[0] + padding[1]
        new_shape[-3] += padding[2] + padding[3]
        new_shape[-4] += padding[4] + padding[5]
        new_shape = tuple(new_shape)

        self.assertTrue(new_shape == Kx.shape)

    def test2d_transpose(self):
        shape = (5, 2, 10, 10)
        x = tf.random.normal(shape)
        padding = [2, 2, 4, 4]
        op = Pad2dTranspose(padding=padding, mode='symmetric', channel_last=False)
        Kx = op(x)

        # manually construct new shape
        new_shape = list(x.shape)
        new_shape[-1] -= padding[0] + padding[1]
        new_shape[-2] -= padding[2] + padding[3]
        new_shape = tuple(new_shape)

        self.assertTrue(new_shape == Kx.shape)

    def test2d_complex_transpose(self):
        shape = (5, 2, 10, 10)
        x = tf.complex(tf.random.normal(shape), tf.random.normal(shape))
        padding = [2, 2, 4, 4]
        op = Pad2dTranspose(padding=padding, mode='symmetric', channel_last=False)
        Kx = op(x)

        # manually construct new shape
        new_shape = list(x.shape)
        new_shape[-1] -= padding[0] + padding[1]
        new_shape[-2] -= padding[2] + padding[3]
        new_shape = tuple(new_shape)

        self.assertTrue(new_shape == Kx.shape)

    def test2d_channel_last_transpose(self):
        shape = (5, 10, 10, 2)
        x = tf.random.normal(shape)
        padding = [2, 2, 4, 4]
        op = Pad2dTranspose(padding=padding, mode='symmetric', channel_last=True)
        Kx = op(x)

        # manually construct new shape
        new_shape = list(x.shape)
        new_shape[-2] -= padding[0] + padding[1]
        new_shape[-3] -= padding[2] + padding[3]
        new_shape = tuple(new_shape)

        self.assertTrue(new_shape == Kx.shape)

    def test3d_transpose(self):
        shape = (5, 2, 8, 10, 10)
        x = tf.random.normal(shape)
        padding = [2, 2, 4, 4, 1, 1]
        op = Pad3dTranspose(padding=padding, mode='symmetric', channel_last=False)
        Kx = op(x)

        # manually construct new shape
        new_shape = list(x.shape)
        new_shape[-1] -= padding[0] + padding[1]
        new_shape[-2] -= padding[2] + padding[3]
        new_shape[-3] -= padding[4] + padding[5]
        new_shape = tuple(new_shape)

        self.assertTrue(new_shape == Kx.shape)

    def test3d_complex_transpose(self):
        shape = (5, 2, 8, 10, 10)
        x = tf.complex(tf.random.normal(shape), tf.random.normal(shape))
        padding = [2, 2, 4, 4, 1, 1]
        op = Pad3dTranspose(padding=padding, mode='symmetric', channel_last=False)
        Kx = op(x)

        # manually construct new shape
        new_shape = list(x.shape)
        new_shape[-1] -= padding[0] + padding[1]
        new_shape[-2] -= padding[2] + padding[3]
        new_shape[-3] -= padding[4] + padding[5]
        new_shape = tuple(new_shape)

        self.assertTrue(new_shape == Kx.shape)

    def test3d_channel_last_transpose(self):
        shape = (5, 8, 10, 10, 2)
        x = tf.random.normal(shape)
        padding = [2, 2, 4, 4, 1, 2]
        op = Pad3dTranspose(padding=padding, mode='symmetric', channel_last=True)
        Kx = op(x)

        # manually construct new shape
        new_shape = list(x.shape)
        new_shape[-2] -= padding[0] + padding[1]
        new_shape[-3] -= padding[2] + padding[3]
        new_shape[-4] -= padding[4] + padding[5]
        new_shape = tuple(new_shape)

        self.assertTrue(new_shape == Kx.shape)

if __name__ == "__main__":
    unittest.test()