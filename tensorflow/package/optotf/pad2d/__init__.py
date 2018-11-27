#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 20:03:00 2018

@author: hofinger
"""

import tensorflow as tf
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
lib_path = os.path.join(dir_path, "TFpad2dOperator.so")
assert os.path.isfile(lib_path), "Build pad2d from source first!"
  


if "pad2d" in locals():
  print("re-using already loaded map_coordinates function")
else:
  _pad2d_op_so = tf.load_op_library( lib_path )
  


  pad2d = _pad2d_op_so.pad2d
  pad2d_transpose = _pad2d_op_so.pad2d_transpose
  
  
@tf.RegisterGradient("Pad2d")
def _Pad2dGrad(op, grad):
    in_grad = _pad2d_op_so.pad2d_transpose(grad, op.get_attr("mode"), op.get_attr("pad"))
    return [in_grad]

@tf.RegisterGradient("Pad2dTranspose")
def _Pad2dTransposeGrad(op, grad):
    in_grad = _pad2d_op_so.pad2d(grad, op.get_attr("mode"), op.get_attr("pad"))
    return [in_grad]