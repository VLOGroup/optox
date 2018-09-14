import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.client import timeline

import numpy as np

_tutorial = tf.load_op_library('../../lib/tf/TFDemosaicingOperator.so')
demosaicing_operator_forward = _tutorial.demosaicing_operator_forward
demosaicing_operator_adjoint = _tutorial.demosaicing_operator_adjoint

in_shape = (2, 12, 12, 3)
out_shape = (2, 12, 12, 1)

np_in = np.random.random(in_shape)
np_out = np.random.random(out_shape)

tf_in = tf.constant(np_in, dtype=tf.float32, name="a")
tf_out = tf.constant(np_out, dtype=tf.float32, name="b")
tf_pattern = tf.placeholder(dtype=tf.int32, shape=(), name="pattern")

tf_op_out = demosaicing_operator_forward(tf_in, tf_pattern)
tf_op_adjoint_out = demosaicing_operator_adjoint(tf_out, tf_pattern)

tf_lhs = tf.reduce_sum(tf_in * tf_op_adjoint_out)
tf_rhs = tf.reduce_sum(tf_out * tf_op_out)

with tf.Session() as sess:
    lhs, rhs = sess.run([tf_lhs, tf_rhs], {tf_pattern: 0})
    print(lhs, rhs, np.abs(lhs - rhs))

