#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 20:03:00 2018

@author: hofinger
"""

import tensorflow as tf
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
lib_path = os.path.join(dir_path, "TFMapcoordinatesOperator.so")
assert os.path.isfile(lib_path), "Build mapcoordinates from source first!"
  


if "_mapcoordinates_op_so" in locals():
  print("re-using already loaded map_coordinates function")
else:
  _mapcoordinates_op_so = tf.load_op_library( lib_path )
  _mapcoordinates_grad = _mapcoordinates_op_so.mapcoordinates_gradients

  mapcoordinates = _mapcoordinates_op_so.mapcoordinates


  
  
  
  @tf.RegisterGradient("Mapcoordinates")
  def _MapcoordinatesGrad(op, grad):
    grad_img,grad_coords = _mapcoordinates_grad(op.inputs[0], op.inputs[1], grad,interp_type= op.get_attr("interp_type"))
    return [grad_img,grad_coords]
    