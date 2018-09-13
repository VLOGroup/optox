from __future__ import print_function

import os as _os
import sys as _sys
import tensorflow as _tf
from tensorflow.python.framework import ops as _ops

# load operators from the library
_fftutils_lib = _tf.load_op_library(
    _tf.resource_loader.get_path_to_datafile("TfFftOperators.so"))

fftshift2d = _fftutils_lib.fftshift2d
ifftshift2d = _fftutils_lib.ifftshift2d

@_ops.RegisterGradient("Fftshift2d")
def _Fftshift2dGrad(op, grad):
    in_grad = _fftutils_lib.ifftshift2d(grad)
    return [in_grad]

@_ops.RegisterGradient("Ifftshift2d")
def _Iftshift2dGrad(op, grad):
    in_grad = _fftutils_lib.fftshift2d(grad)
    return [in_grad]

def conv2d_complex(u, k, strides=[1,1,1,1], padding='SAME', data_format='NHWC'):
    """ Complex 2d convolution with the same interface as `conv2d`.
    """
    conv_rr = _tf.nn.conv2d(_tf.real(u), _tf.real(k),  strides=strides, padding=padding,
                                     data_format=data_format)
    conv_ii = _tf.nn.conv2d(_tf.imag(u), _tf.imag(k),  strides=strides, padding=padding,
                                     data_format=data_format)
    conv_ri = _tf.nn.conv2d(_tf.real(u), _tf.imag(k), strides=strides, padding=padding,
                                     data_format=data_format)
    conv_ir = _tf.nn.conv2d(_tf.imag(u), _tf.real(k), strides=strides, padding=padding,
                                     data_format=data_format)
    return _tf.complex(conv_rr-conv_ii, conv_ri+conv_ir)

def conv2d_transpose_complex(u, k, output_shape, strides=[1,1,1,1], padding='SAME', data_format='NHWC'):
    """ Complex 2d transposed convolution with the same interface as `conv2d_transpose`.
    """
    convT_rr = _tf.nn.conv2d_transpose(_tf.real(u), _tf.real(k), output_shape, strides=strides, padding=padding,
                                     data_format=data_format)
    convT_ii = _tf.nn.conv2d_transpose(_tf.imag(u), _tf.imag(k), output_shape, strides=strides, padding=padding,
                                     data_format=data_format)
    convT_ri = _tf.nn.conv2d_transpose(_tf.real(u), _tf.imag(k), output_shape, strides=strides, padding=padding,
                                     data_format=data_format)
    convT_ir = _tf.nn.conv2d_transpose(_tf.imag(u), _tf.real(k), output_shape, strides=strides, padding=padding,
                                     data_format=data_format)
    return _tf.complex(convT_rr+convT_ii, convT_ir-convT_ri)

def ifftc2d(inp):
    """ Centered inverse 2d Fourier transform, performed on axis (-1,-2).
    """
    shape = _tf.shape(inp)
    numel = shape[-2]*shape[-1]
    scale = _tf.sqrt(_tf.cast(numel, _tf.float32))

    out = fftshift2d(_tf.ifft2d(ifftshift2d(inp)))
    out = _tf.complex(_tf.real(out)*scale, _tf.imag(out)*scale)
    return out

def fftc2d(inp):
    """ Centered 2d Fourier transform, performed on axis (-1,-2).
    """
    shape = _tf.shape(inp)
    numel = shape[-2]*shape[-1]
    scale = 1.0 / _tf.sqrt(_tf.cast(numel, _tf.float32))

    out = fftshift2d(_tf.fft2d(ifftshift2d(inp)))
    out = _tf.complex(_tf.real(out) * scale, _tf.imag(out) * scale)
    return out
