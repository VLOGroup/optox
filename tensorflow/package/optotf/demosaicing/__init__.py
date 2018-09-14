from __future__ import print_function

import os as _os
import sys as _sys
import tensorflow as _tf
from tensorflow.python.framework import ops as _ops


# load operators from the library
_lib = _tf.load_op_library(
    _tf.resource_loader.get_path_to_datafile("TFDemosaicingOperator.so"))

forward = _lib.demosaicing_operator_forward
adjoint = _lib.demosaicing_operator_adjoint

@_ops.RegisterGradient("DemosaicingOperatorForward")
def _demosaicing_forward_grad(op, grad):
    grad_in = adjoint(grad, op.inputs[1])
    return [grad_in, None]

@_ops.RegisterGradient("DemosaicingOperatorAdjoint")
def _demosaicing_forward_grad(op, grad):
    grad_in = forward(grad, op.inputs[1])
    return [grad_in, None]