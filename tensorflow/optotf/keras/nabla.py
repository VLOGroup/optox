import tensorflow as tf
import optotf.nabla
import unittest

class Nabla2d(tf.keras.layers.Layer):
    def __init__(self, hx=1, hy=1):
        super().__init__()
        self.op = lambda x: optotf.nabla.nabla_2d(x, hx=hx, hy=hy)

    def call(self, x):
        if x.dtype == tf.complex64 or x.dtype == tf.complex128:
            return tf.complex(self.op(tf.math.real(x)), 
                              self.op(tf.math.imag(x)))
        else:
            return self.op(x)

class Nabla3d(tf.keras.layers.Layer):
    def __init__(self, hx=1, hy=1, hz=1):
        super().__init__()
        self.op = lambda x: optotf.nabla.nabla_3d(x, hx=hx, hy=hy, hz=hz)

    def call(self, x):
        if x.dtype == tf.complex64 or x.dtype == tf.complex128:
            return tf.complex(self.op(tf.math.real(x)), 
                              self.op(tf.math.imag(x)))
        else:
            return self.op(x)

class Nabla4d(tf.keras.layers.Layer):
    def __init__(self, hx=1, hy=1, hz=1, ht=1):
        super().__init__()
        self.op = lambda x: optotf.nabla.nabla_4d(x, hx=hx, hy=hy, hz=hz, ht=ht)

    def call(self, x):
        if x.dtype == tf.complex64 or x.dtype == tf.complex128:
            return tf.complex(self.op(tf.math.real(x)), 
                              self.op(tf.math.imag(x)))
        else:
            return self.op(x)


class NablaT2d(tf.keras.layers.Layer):
    def __init__(self, hx=1, hy=1):
        super().__init__()
        self.op = lambda x: optotf.nabla.nabla_2d_adjoint(x, hx=hx, hy=hy)

    def call(self, x):
        if x.dtype == tf.complex64 or x.dtype == tf.complex128:
            return tf.complex(self.op(tf.math.real(x)), 
                              self.op(tf.math.imag(x)))
        else:
            return self.op(x)

class NablaT3d(tf.keras.layers.Layer):
    def __init__(self, hx=1, hy=1, hz=1):
        super().__init__()
        self.op = lambda x: optotf.nabla.nabla_3d_adjoint(x, hx=hx, hy=hy, hz=hz)

    def call(self, x):
        if x.dtype == tf.complex64 or x.dtype == tf.complex128:
            return tf.complex(self.op(tf.math.real(x)), 
                              self.op(tf.math.imag(x)))
        else:
            return self.op(x)

class NablaT4d(tf.keras.layers.Layer):
    def __init__(self, hx=1, hy=1, hz=1, ht=1):
        super().__init__()
        self.op = lambda x: optotf.nabla.nabla_4d_adjoint(x, hx=hx, hy=hy, hz=hz, ht=ht)

    def call(self, x):
        if x.dtype == tf.complex64 or x.dtype == tf.complex128:
            return tf.complex(self.op(tf.math.real(x)), 
                              self.op(tf.math.imag(x)))
        else:
            return self.op(x)

class TestNabla(unittest.TestCase):
    def test2d(self):
        x = tf.random.normal((10, 10))
        op = Nabla2d()
        Kx = op(x)
        self.assertTrue((2, *x.shape) == Kx.shape)

    def test2d_complex(self):
        x = tf.complex(tf.random.normal((10, 10)),
                       tf.random.normal((10, 10)))
        op = Nabla2d()
        Kx = op(x)
        self.assertTrue((2, *x.shape) == Kx.shape)

    def test2d_adjoint(self):
        x = tf.random.normal((2, 10, 10))
        op = NablaT2d()
        Kx = op(x)
        self.assertTrue(x.shape[1:] == Kx.shape)

    def test2d_adjoint_complex(self):
        x = tf.complex(tf.random.normal((2, 10, 10)),
                       tf.random.normal((2, 10, 10)))
        op = NablaT2d()
        Kx = op(x)
        self.assertTrue(x.shape[1:] == Kx.shape)

    def test3d(self):
        x = tf.random.normal((10, 10, 10))
        op = Nabla3d()
        Kx = op(x)
        self.assertTrue((3, *x.shape) == Kx.shape)

    def test3d_complex(self):
        x = tf.complex(tf.random.normal((10, 10, 10)),
                       tf.random.normal((10, 10, 10)))
        op = Nabla3d()
        Kx = op(x)
        self.assertTrue((3, *x.shape) == Kx.shape)

    def test3d_adjoint(self):
        x = tf.random.normal((3, 10, 10, 10))
        op = NablaT3d()
        Kx = op(x)
        self.assertTrue(x.shape[1:] == Kx.shape)

    def test3d_adjoint_complex(self):
        x = tf.complex(tf.random.normal((3, 10, 10, 10)),
                       tf.random.normal((3, 10, 10, 10)))
        op = NablaT3d()
        Kx = op(x)
        self.assertTrue(x.shape[1:] == Kx.shape)

    def test4d(self):
        x = tf.random.normal((10, 10, 10, 10))
        op = Nabla4d()
        Kx = op(x)
        self.assertTrue((4, *x.shape) == Kx.shape)

    def test4d_complex(self):
        x = tf.complex(tf.random.normal((10, 10, 10, 10)),
                       tf.random.normal((10, 10, 10, 10)))
        op = Nabla4d()
        Kx = op(x)
        self.assertTrue((4, *x.shape) == Kx.shape)

    def test4d_adjoint(self):
        x = tf.random.normal((4, 10, 10, 10, 10))
        op = NablaT4d()
        Kx = op(x)
        self.assertTrue(x.shape[1:] == Kx.shape)

    def test4d_adjoint_complex(self):
        x = tf.complex(tf.random.normal((4, 10, 10, 10, 10)),
                       tf.random.normal((4, 10, 10, 10, 10)))
        op = NablaT4d()
        Kx = op(x)
        self.assertTrue(x.shape[1:] == Kx.shape)

if __name__ == "__main__":
    unittest.test()