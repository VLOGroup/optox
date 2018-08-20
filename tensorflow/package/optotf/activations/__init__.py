from __future__ import print_function

import os as _os
import sys as _sys
import tensorflow as _tf
from tensorflow.python.framework import ops as _ops


# load operators from the library
_activation_lib = _tf.load_op_library(
    _tf.resource_loader.get_path_to_datafile("TfActivationOperators.so"))

# rbf
rbf = _activation_lib.activation_rbf
prime_rbf = _activation_lib.activation_prime_rbf
int_rbf = _activation_lib.activation_int_rbf

# linear interpolation
interpolate_linear = _activation_lib.activation_interpolate_linear
int_interpolate_linear = _activation_lib.activation_integral_interpolate_linear

# spline
b_spline = _activation_lib.activation_b_spline
quad_b_spline = _activation_lib.activation_quad_b_spline
cubic_b_spline = _activation_lib.activation_cubic_b_spline
prime_cubic_b_spline = _activation_lib.activation_prime_cubic_b_spline


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
    rbf_prime = _activation_lib.activation_double_prime_rbf(op.inputs[0], op.inputs[1], op.get_attr(
        "v_min"), op.get_attr("v_max"), op.get_attr("num_weights"), op.get_attr("feature_stride"))
    grad_x = rbf_prime * grad
    grad_w = _activation_lib.activation_prime_rbf_grad_w(op.inputs[0], grad, op.get_attr("v_min"), op.get_attr(
        "v_max"), op.inputs[1].shape[0], op.get_attr("num_weights"), op.get_attr("feature_stride"))
    return [grad_x, grad_w]


@_ops.RegisterGradient("ActivationInterpolateLinear")
def _activation_interpolate_linear_grad(op, grad):
    act_prime = _activation_lib.activation_prime_interpolate_linear(op.inputs[0], op.inputs[1], op.get_attr(
        "v_min"), op.get_attr("v_max"), op.get_attr("num_weights"), op.get_attr("feature_stride"))
    grad_x = act_prime * grad
    grad_w = _activation_lib.activation_interpolate_linear_grad_w(op.inputs[0], grad, op.get_attr(
        "v_min"), op.get_attr("v_max"), op.inputs[1].shape[0], op.get_attr("num_weights"), op.get_attr("feature_stride"))
    return [grad_x, grad_w]


@_ops.RegisterGradient("ActivationIntegralInterpolateLinear")
def _activation_int_interpolate_linear_grad(op, grad):
    act = _activation_lib.activation_interpolate_linear(op.inputs[0], op.inputs[1], op.get_attr(
        "v_min"), op.get_attr("v_max"), op.get_attr("num_weights"), op.get_attr("feature_stride"))
    grad_x = act * grad
    grad_w = _activation_lib.activation_integral_interpolate_linear_grad_w(op.inputs[0], grad, op.get_attr(
        "v_min"), op.get_attr("v_max"), op.inputs[1].shape[0], op.get_attr("num_weights"), op.get_attr("feature_stride"))
    return [grad_x, grad_w]


@_ops.RegisterGradient("ActivationBSpline")
def _activation_bubic_b_spline_grad(op, grad):
    rbf_prime = _activation_lib.activation_prime_b_spline(op.inputs[0], op.inputs[1], op.get_attr(
        "v_min"), op.get_attr("v_max"), op.get_attr("num_weights"), op.get_attr("feature_stride"))
    grad_x = rbf_prime * grad
    grad_w = _activation_lib.activation_b_spline_grad_w(op.inputs[0], grad, op.get_attr("v_min"), op.get_attr(
        "v_max"), op.inputs[1].shape[0], op.get_attr("num_weights"), op.get_attr("feature_stride"))
    return [grad_x, grad_w]


@_ops.RegisterGradient("ActivationQuadBSpline")
def _activation_cubic_b_spline_grad(op, grad):
    rbf_prime = _activation_lib.activation_prime_quad_b_spline(op.inputs[0], op.inputs[1], op.get_attr(
        "v_min"), op.get_attr("v_max"), op.get_attr("num_weights"), op.get_attr("feature_stride"))
    grad_x = rbf_prime * grad
    grad_w = _activation_lib.activation_quad_b_spline_grad_w(op.inputs[0], grad, op.get_attr("v_min"), op.get_attr(
        "v_max"), op.inputs[1].shape[0], op.get_attr("num_weights"), op.get_attr("feature_stride"))
    return [grad_x, grad_w]


@_ops.RegisterGradient("ActivationCubicBSpline")
def _activation_cubic_b_spline_grad(op, grad):
    rbf_prime = prime_cubic_b_spline(op.inputs[0], op.inputs[1], op.get_attr(
        "v_min"), op.get_attr("v_max"), op.get_attr("num_weights"), op.get_attr("feature_stride"))
    grad_x = rbf_prime * grad
    grad_w = _activation_lib.activation_cubic_b_spline_grad_w(op.inputs[0], grad, op.get_attr("v_min"), op.get_attr(
        "v_max"), op.inputs[1].shape[0], op.get_attr("num_weights"), op.get_attr("feature_stride"))
    return [grad_x, grad_w]


@_ops.RegisterGradient("ActivationPrimeCubicBSpline")
def _activation_prime_cubic_b_spline_grad(op, grad):
    rbf_double_prime = _activation_lib.activation_double_prime_cubic_b_spline(op.inputs[0], op.inputs[1], op.get_attr(
        "v_min"), op.get_attr("v_max"), op.get_attr("num_weights"), op.get_attr("feature_stride"))
    grad_x = rbf_double_prime * grad
    grad_w = _activation_lib.activation_prime_cubic_b_spline_grad_w(op.inputs[0], grad, op.get_attr(
        "v_min"), op.get_attr("v_max"), op.inputs[1].shape[0], op.get_attr("num_weights"), op.get_attr("feature_stride"))
    return [grad_x, grad_w]
