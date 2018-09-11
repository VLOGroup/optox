import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import optotf.activations as act

Nw = 3
vmin = -1
vmax = 1

np_x = np.linspace(vmin, vmax, Nw)[np.newaxis, :]
np_w = np.abs(np_x)

tf_w = tf.Variable(initial_value=np_w, dtype=tf.float32)

tf_x = tf.placeholder(dtype=tf.float32)

#tf_phi = act.interpolate_linear(tf_x, tf_w, v_min=vmin, v_max=vmax, num_weights=Nw, feature_stride=1)
#tf_phi_extrapolate = act.interpolate_linear_extrapolate(tf_x, tf_w, v_min=vmin, v_max=vmax, num_weights=Nw, feature_stride=1)
tf_phi = act.int_interpolate_linear(tf_x, tf_w, v_min=vmin, v_max=vmax, num_weights=Nw, feature_stride=1)
tf_phi_extrapolate = act.int_interpolate_linear_extrapolate(tf_x, tf_w, v_min=vmin, v_max=vmax, num_weights=Nw, feature_stride=1)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    np_x = np.linspace(3*vmin, 3*vmax, 1001)[:, np.newaxis]

    np_phi = sess.run(tf_phi, {tf_x: np_x})
    np_phi_extrapolate = sess.run(tf_phi_extrapolate, {tf_x: np_x})

    plt.figure(1)
    plt.subplot(1,2,1)
    plt.plot(np_x, np_phi)
    plt.title('zero')
    plt.subplot(1,2,2)
    plt.plot(np_x, np_phi_extrapolate)
    plt.title('extrapolate')
    plt.show()
