from __future__ import print_function

import os as _os
import sys as _sys
import tensorflow as _tf
from tensorflow.python.framework import ops as _ops


# load operators from the library
_filters_lib = _tf.load_op_library(
    _tf.resource_loader.get_path_to_datafile("TfRotateFiltersOperator.so"))

rotate_filter = _filters_lib.rotate_filter

@_ops.RegisterGradient("RotateFilter")
def _RotateFilterGrad(op, grad):
    in_grad = _filters_lib.rotate_filter_grad(op.inputs[1], grad, op.get_attr("interpolation"))
    return [in_grad, None]

# load operators from the library
_meta_lib = _tf.load_op_library(
    _tf.resource_loader.get_path_to_datafile("TfMetamorphosisOperator.so"))

metamorphosis_warp = _meta_lib.metamorphosis_warp
warp = _meta_lib.warp

@_ops.RegisterGradient("MetamorphosisWarp")
def _RotateFilterGrad(op, grad):
    in_grad, phi_grad = _meta_lib.metamorphosis_warp_grad(op.inputs[0], op.inputs[1], grad, op.get_attr("interpolation"))
    return [in_grad, phi_grad]

@_ops.RegisterGradient("Warp")
def _WarpGrad(op, grad):
    in_grad, phi_grad = _meta_lib.warp_grad(op.inputs[0], op.inputs[1], grad, op.get_attr("interpolation"))
    return [in_grad, phi_grad]