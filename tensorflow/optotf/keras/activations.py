import tensorflow as tf
import optotf.activations
import unittest
import numpy as np

class TrainableActivationInitializer(tf.keras.initializers.Initializer):
    def __init__(self, vmin, vmax, num_weights, init, init_scale, num_channels):
        self.initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
        self.vmin = vmin
        self.vmax = vmax
        self.num_weights = num_weights
        self.init = init
        self.init_scale = init_scale
        self.num_channels = num_channels
    
    def __call__(self, shape, dtype=None):
        # define the bins
        np_x = np.linspace(self.vmin, self.vmax, self.num_weights, dtype=np.float32)[np.newaxis, :]
        # initialize the weights
        if self.init == "constant":
            np_w = np.ones_like(np_x) * self.init_scale
        elif self.init == "linear":
            np_w = np_x * self.init_scale
        elif self.init == "quadratic":
            np_w = np_x**2 * self.init_scale
        elif self.init == "abs":
            np_w = np.abs(np_x) * self.init_scale
        elif self.init == "student-t":
            alpha = 100
            np_w = self.init_scale * np.sqrt(alpha) * np_x / (1 + 0.5 * alpha * np_x ** 2)
        elif self.init == "invert":
            np_w = self.init_scale / np_x
            if not np.all(np.isfinite(np_w)):
                raise RuntimeError("Invalid value encountered in weight init!")
        else:
            raise RuntimeError("Unsupported init type '{}'!".format(self.init))
        # tile to proper size
        np_w = np.tile(np_w, (self.num_channels, 1))

        return np_w

class TrainableActivation(tf.keras.layers.Layer):
    def __init__(self, vmin, vmax, num_weights, base_type="rbf", init="linear", init_scale=1.0,
                 group=1, **kwargs):
        super(TrainableActivation, self).__init__()

        self.vmin = vmin
        self.vmax = vmax
        self.num_weights = num_weights
        self.base_type = base_type
        self.init = init
        self.init_scale = init_scale
        self.group = group

        # determine the operator
        if self.base_type in ["rbf", "linear", "spline"]:
            self.op = optotf.activations._get_operator(self.base_type)
        else:
            raise RuntimeError("Unsupported base type '{}'!".format(base_type))

    def build(self, input_shape):
        super().build(input_shape)

        self.num_channels = input_shape[-1]
        
        # setup the parameters of the layer
        initializer = TrainableActivationInitializer(self.vmin, self.vmax, self.num_weights, self.init, self.init_scale, self.num_channels)
        self.weight = self.add_weight('weight', shape=(self.num_channels, self.num_weights), initializer=initializer)
        # define the reduction index
        self.weight.reduction_dim = (1, )


    def call(self, x):
        # first reshape the input
        shape = tf.shape(x)
        x = tf.transpose(tf.reshape(x, (-1, shape[-1])), [1, 0])
        # if tf.shape(x)[0] % self.group != 0: # TODO move this to cpp code!
        #     raise RuntimeError("Input shape must be a multiple of group!") 
        x_r = tf.reshape(x, (tf.shape(x)[0]//self.group, -1))
        # compute the output
        x_r = self.op(x_r, self.weight, vmin=self.vmin, vmax=self.vmax)
        return tf.reshape(tf.transpose(tf.reshape(x_r, tf.shape(x)), [1, 0]), shape)

    def extra_repr(self):
        s = "num_channels={num_channels}, num_weights={num_weights}, type={base_type}, vmin={vmin}, vmax={vmax}, init={init}, init_scale={init_scale}"
        s += " group={group}"
        return s.format(**self.__dict__)

class TestActivations(unittest.TestCase):
    def test(self):
        x = np.random.normal((10, 5))
        op = TrainableActivation(-0.5, 0.5, 31)
        y = op(x)
        self.assertTrue(x.shape == y.shape)
        
if __name__ == "__main__":
    unittest.test()