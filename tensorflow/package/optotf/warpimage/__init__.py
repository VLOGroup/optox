#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 20:03:00 2018

@author: hofinger
"""

import tensorflow as tf
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
lib_path = os.path.join(dir_path, "TFWarpimageOperator.so")
assert os.path.isfile(lib_path), "Build warpimage from source first!"
  


if "_warpimage_op_so" in locals():
  print("re-using already loaded warpimage function")
else:
  _warpimage_op_so = tf.load_op_library( lib_path )
  _warpimage_grad = _warpimage_op_so.warpimage_gradients

  warpimage = _warpimage_op_so.warpimage


  
  
  
  @tf.RegisterGradient("Warpimage")
  def _WarpimageGrad(op, grad):
    grad_img,grad_coords = _warpimage_grad(op.inputs[0], op.inputs[1], grad,interp_type= op.get_attr("interp_type"))
    return [grad_img,grad_coords]
    