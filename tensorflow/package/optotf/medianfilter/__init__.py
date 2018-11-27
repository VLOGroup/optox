"""Python layer for filterexamples_ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os


dir_path = os.path.dirname(os.path.realpath(__file__))
lib_path = os.path.join(dir_path, "TFMedianfilterOperator.so")
assert os.path.isfile(lib_path), "Build mapcoordinates from source first!"
_median_filter_4D_so = tf.load_op_library(lib_path)

#median_filter =  _filterexamples_ops_so.filterexamples_median_filter
#median_filter_gradient =  _filterexamples_ops_so.filterexamples_median_filter_gradient

median_filter4d =  _median_filter_4D_so.filterexamples_median_filter4d
median_filter4d_gradient =  _median_filter_4D_so.filterexamples_median_filter4d_gradient

@tf.RegisterGradient("FilterexamplesMedianFilter4d")
def _MedianFilter4dGrad(op, grad):
    grad_med = median_filter4d_gradient(op.inputs[0], grad,filtersize= op.get_attr("filtersize"),filtertype= op.get_attr("filtertype"), debug_indices =op.get_attr("debug_indices"))
    return [grad_med]

#@tf.RegisterGradient("FilterexamplesMedianFilter")
#def _MedianFilterGrad(op, grad):
#    grad_med = median_filter_gradient(op.inputs[0], grad,filtersize= op.get_attr("filtersize"),filtertype= op.get_attr("filtertype"), debug_indices =op.get_attr("debug_indices"))
#    return [grad_med]
#
