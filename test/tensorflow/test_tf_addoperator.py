import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.client import timeline

import numpy as np

_tutorial = tf.load_op_library('../../lib/tf/TfAddOperator.so')
custom_add = _tutorial.custom_add

shape = (1,10,10,1)

a_data = np.random.random(shape)
b_data = np.random.random(shape)

a_data = np.ones_like(a_data)
b_data = np.ones_like(b_data) * 2

a = tf.placeholder(tf.float32, shape=shape, name="a")
b = tf.placeholder(tf.float32, shape=shape, name="b")

c_cust = custom_add(a,b)
print(c_cust)
c =tf.log(tf.exp(a + b))

config = tf.ConfigProto(log_device_placement = True)
config.gpu_options.allow_growth = True
config.graph_options.optimizer_options.opt_level = -1

options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()

with tf.Session(config=config) as sess:
    feed = {a: a_data, b: b_data}
    _, result = sess.run([c, c_cust], feed)
#    _, result = sess.run([c, c_cust], feed_dict=feed, options=options, run_metadata=run_metadata)
    expected = sess.run(c, feed_dict=feed)
    print(np.array_equal(expected,result))

    # Create the Timeline object, and write it to a json file
#    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
#    chrome_trace = fetched_timeline.generate_chrome_trace_format()
#    with open('timeline_01.json', 'w') as f:
#        f.write(chrome_trace)

