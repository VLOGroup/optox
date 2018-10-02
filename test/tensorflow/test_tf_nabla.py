import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import optotf

M = N = 10
tf_u = tf.constant((np.random.randn(M, N)), dtype=tf.float32)
tf_p = tf.constant((np.random.randn(2, M, N)), dtype=tf.float32)

tf_Au = optotf.nabla(tf_u)
tf_ATp = optotf.div(tf_p)
print(tf_p, tf_Au)
print(tf_u, tf_ATp)

lhs = tf.reduce_sum(tf_p * tf_Au)
rhs = tf.reduce_sum(tf_u * tf_ATp)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    print(sess.run([lhs, rhs]))
