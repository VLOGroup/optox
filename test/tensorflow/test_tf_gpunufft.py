import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.client import timeline

import numpy as np
import matplotlib.pyplot as plt

_tutorial = tf.load_op_library('../../lib/tf/TfGpuNufftOperator.so')
gpu_nufft_forward = _tutorial.gpu_nufft_forward
gpu_nufft_adjoint = _tutorial.gpu_nufft_adjoint

# load data
k = np.load('../../../pd_toolbox/data/brain_trajectory.npy')
sensitivities = np.load('../../../pd_toolbox/data/brain_sensitivities.npy')
rawdata = np.load('../../../pd_toolbox/data/brain_rawdata.npy')

# setup trajectory k and dcf weighting w
[nCh, nFE, nSpokes] = rawdata.shape
rawdata = np.reshape(rawdata, [nCh, nFE*nSpokes])
k_col = k.flatten()
k_col = np.array([np.imag(k_col), np.real(k_col)])
w = np.sqrt(np.abs(k.flatten()))

# init operator
config = {'osf' : 2,
          'sector_width' : 8,
          'kernel_width' : 3,
          'img_dim' : nFE/2}

print('Trajectory: {}'.format(k_col.shape))
print('Sensitivities: {}'.format(sensitivities.shape))
print('Rawdata: {}'.format(rawdata.shape))
print('DCF: {}'.format(w.shape))
print(config)

rawdata_tf = tf.placeholder(tf.complex64, name="rawdata")
sensitivities_tf = tf.placeholder(tf.complex64, name="sensitivities")
trajectory_tf = tf.placeholder(tf.float32, name="trajectory")
dcf_tf = tf.placeholder(tf.float32, name="dcf")

img_tf = gpu_nufft_adjoint(rawdata_tf,
                           sensitivities_tf,
                           trajectory_tf,
                           dcf_tf,
                           osf=config['osf'],
                           sector_width=config['sector_width'],
                           kernel_width=config['kernel_width'],
                           img_dim=config['img_dim'])

forward_tf = gpu_nufft_forward(img_tf,
                                sensitivities_tf,
                                trajectory_tf,
                                dcf_tf,
                                osf=config['osf'],
                                sector_width=config['sector_width'],
                                kernel_width=config['kernel_width'],
                                img_dim=config['img_dim'])

config = tf.ConfigProto(log_device_placement = True)
config.gpu_options.allow_growth = True
config.graph_options.optimizer_options.opt_level = -1

options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()

rawdata = np.array([rawdata, rawdata])
sensitivities = np.array([sensitivities, sensitivities])
k_col = np.array([k_col, k_col])
w = np.array([w, np.ones_like(w)])

with tf.Session(config=config) as sess:
    feed = {rawdata_tf : rawdata,
            sensitivities_tf : sensitivities,
            trajectory_tf : k_col,
            dcf_tf : w}

    result = sess.run(img_tf, feed_dict=feed, options=options, run_metadata=run_metadata)
    kspace = sess.run(forward_tf, feed_dict=feed)
    print(kspace.shape)
    print(result.shape)

    # Create the Timeline object, and write it to a json file
    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
    chrome_trace = fetched_timeline.generate_chrome_trace_format()
    with open('timeline_01.json', 'w') as f:
        f.write(chrome_trace)
    plt.figure(1)
    plt.imshow(np.abs(result[1]),cmap='gray')
    plt.figure(0)
    plt.imshow(np.abs(result[0]),cmap='gray')
    plt.show()