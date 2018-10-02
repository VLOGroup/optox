from __future__ import print_function

import os as _os
import sys as _sys
import tensorflow as _tf
from tensorflow.python.framework import ops as _ops


# load operators from the library
_nabla_lib = _tf.load_op_library(
    _tf.resource_loader.get_path_to_datafile("TfNablaOperator.so"))

nabla = _nabla_lib.nabla_operator
div = _nabla_lib.nabla_operator_adjoint

@_ops.RegisterGradient("NablaOperator")
def _NablaGrad(op, grad):
    in_grad = div(grad)
    return [in_grad]

@_ops.RegisterGradient("NablaOperatorAdjoint")
def _DivGrad(op, grad):
    in_grad = nabla(grad)
    return [in_grad]
