from __future__ import print_function

import tensorflow as _tf
from tensorflow.python.framework import ops as _ops

__all__ = ['forward', 'adjoint']

# load operators from the library
_ext = _tf.load_op_library(_tf.compat.v1.resource_loader.get_path_to_datafile("tf_demosaicing_operator.so"))

forward = _ext.demosaicing_operator_forward
adjoint = _ext.demosaicing_operator_adjoint

@_ops.RegisterGradient("DemosaicingOperatorForward")
def _demosaicing_forward_grad(op, grad):
    grad_in = adjoint(grad, op.inputs[1])
    return [grad_in, None]

@_ops.RegisterGradient("DemosaicingOperatorAdjoint")
def _demosaicing_adjoint_grad(op, grad):
    grad_in = forward(grad, op.inputs[1])
    return [grad_in, None]
