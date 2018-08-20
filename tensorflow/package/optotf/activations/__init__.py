from __future__ import print_function

import os as _os
import sys as _sys
import tensorflow as _tf
from tensorflow.python.framework import ops as _ops


# load operators from the library
_activation_lib = _tf.load_op_library(
    _tf.resource_loader.get_path_to_datafile("TfActivationOperators.so"))

# define the notations
rbf = _activation_lib.activation_rbf
prime_rbf = _activation_lib.activation_prime_rbf


@_ops.RegisterGradient("ActivationRBF")
def _activation_rbf_grad(op, grad):
    # gradient w.r.t. the input
    rbf_prime = prime_rbf(op.inputs[0], op.inputs[1], op.get_attr("v_min"), op.get_attr(
        "v_max"), op.get_attr("num_weights"), op.get_attr("feature_stride"))
    grad_x = rbf_prime * grad
    # gradient w.r.t. the weights
    grad_w = _activation_lib.activation_rbf_grad_w(op.inputs[0], grad, op.get_attr("v_min"), op.get_attr(
        "v_max"), op.inputs[1].shape[0], op.get_attr("num_weights"), op.get_attr("feature_stride"))
    return [grad_x, grad_w]


@_ops.RegisterGradient("ActivationPrimeRBF")
def _activation_rbf_grad(op, grad):
    # gradient w.r.t. the input
    rbf_prime = _activation_lib.activation_double_prime_rbf(op.inputs[0], op.inputs[1], op.get_attr(
        "v_min"), op.get_attr("v_max"), op.get_attr("num_weights"), op.get_attr("feature_stride"))
    grad_x = rbf_prime * grad
    # gradient w.r.t. the weights
    grad_w = _activation_lib.activation_prime_rbf_grad_w(op.inputs[0], grad, op.get_attr("v_min"), op.get_attr(
        "v_max"), op.inputs[1].shape[0], op.get_attr("num_weights"), op.get_attr("feature_stride"))
    return [grad_x, grad_w]
