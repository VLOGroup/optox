import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import optotf.activations as act

Nw = 9
vmin = -1
vmax = 1

nptype = np.float64
tftype = tf.float64

np_x_0 = np.linspace(-3, 3, 1001, dtype=nptype)[:, np.newaxis]
np_x = np.linspace(vmin, vmax, Nw, dtype=nptype)[np.newaxis, :]
np_w =np_x

tf_w = tf.placeholder(dtype=tftype, shape=np_w.shape)
tf_x = tf.placeholder(dtype=tftype)

tf_s = tf.placeholder(dtype=tftype, name="s")

phi = act.prime_quad_b_spline

tf_phi = phi(tf_x * tf_s, tf_w, v_min=vmin, v_max=vmax, num_weights=Nw, feature_stride=1)

tf_loss = tf.reduce_sum(tf_phi**2)/2

init = tf.global_variables_initializer()

np_s = 1

with tf.Session() as sess:
    sess.run(init)

    np_phi = sess.run(tf_phi, {tf_x: np_x_0, tf_w: np_w, tf_s: np_s})

    plt.figure(1)
    plt.plot(np_x_0, np_phi)

tf_grad_phi = tf.placeholder(shape=np_x_0.shape, dtype=tftype, name='grad_y')
tf_grad_x = tf.gradients(tf_phi, tf_x, tf_grad_phi)
tf_grad_w = tf.gradients(tf_loss, tf_w)
tf_grad_s = tf.gradients(tf_loss, tf_s)

with tf.Session() as sess:
    sess.run(init)

    # compute the gradient
    np_grad_x, np_grad_w, np_grad_s = sess.run([tf_grad_x[0], tf_grad_w[0], tf_grad_s], feed_dict={tf_x: np_x_0, tf_w : np_w, tf_grad_phi: np.ones_like(np_x_0), tf_s: np_s})

    plt.figure(2)
    for i in range(np_x_0.shape[1]):
        plt.plot(np_x_0[:,i], np_grad_x[:,i])

    # compute it numerically
    epsilon = 1e-6
    np_w_flat = np_w.copy().reshape([-1])
    np_grad_w_num = np.zeros_like(np_w_flat)
    for c in range(np_w_flat.shape[0]):
        np_w_flat[c] -= epsilon
        L_n = sess.run(tf_loss, feed_dict={tf_x: np_x_0, tf_w:  np_w_flat.reshape(np_w.shape), tf_s: np_s})
        np_w_flat[c] += 2*epsilon
        L_p = sess.run(tf_loss, feed_dict={tf_x: np_x_0, tf_w:  np_w_flat.reshape(np_w.shape), tf_s: np_s})
        np_grad_w_num[c] = (L_p - L_n) / (2*epsilon)
        np_w_flat[c] += epsilon

    print("Check grad_w:", np.allclose(np_grad_w.reshape([-1]), np_grad_w_num, rtol=1e-4, atol=1e-4))
    i = np.argmax(np.abs(np_grad_w_num - np_grad_w.reshape([-1])))
    print("tf:", np_grad_w.reshape([-1])[i], " num:", np_grad_w_num[i], " diff:", np_grad_w.reshape([-1])[i] - np_grad_w_num[i])

    plt.figure(3)
    plt.plot(np_grad_w.reshape([-1]))
    plt.plot(np_grad_w_num)
    plt.legend(['grad_w', 'grad_w_num'])

    # compute it numerically
    epsilon = 1e-6
    np_s_flat = np_s
    L_n = sess.run(tf_loss, feed_dict={tf_x: np_x_0, tf_w:  np_w_flat.reshape(np_w.shape), tf_s: np_s - epsilon})
    L_p = sess.run(tf_loss, feed_dict={tf_x: np_x_0, tf_w:  np_w_flat.reshape(np_w.shape), tf_s: np_s + epsilon})
    np_grad_s_num = (L_p - L_n) / (2*epsilon)

    print("Check grad_s:", np.allclose(np_grad_s, np_grad_s_num, rtol=1e-4, atol=1e-4))
    print("tf:", np_grad_s, " num:", np_grad_s_num, " diff:", np_grad_s - np_grad_s_num)


plt.show()
