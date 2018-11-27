#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 20:03:00 2018

@author: hofinger
"""
import tensorflow as tf
import os

from tensorflow.contrib.util import loader
from tensorflow.python.platform import resource_loader

lib = "TFWarpimageOperator"
print ("using locally built %s function..." % lib)
dir_path = os.path.dirname(os.path.realpath(__file__))
lib_path = os.path.join(dir_path,"../../lib/tf" ,lib + ".so")
print (lib_path)
if os.path.isfile(lib_path):
    if "lib_so" in locals():
        print("re-using already loaded %s function" % lib)
    else:
        lib_so = loader.load_op_library(lib_path)
        print( "Found Ops:",[name for name in dir (lib_so) if name[0] != "_"])

        ###########################
        # CUSTOM LOAD code      
        _warpimage_op_so = lib_so
        my_warpimage = _warpimage_op_so.warpimage
        mgrad = _warpimage_op_so.warpimage_gradients
        
        from tensorflow.python.framework import ops 
        
        @ops.RegisterGradient("Warpimage")
        def _warpimageGrad(op, grad):
          grad_img,grad_coords = mgrad(op.inputs[0], op.inputs[1], grad,interp_type= op.get_attr("interp_type"))
          return [grad_img,grad_coords]
  
elif hasattr (tf.contrib,"icg") and hasattr (tf.contrib.icg,"warpimage"):
  print ("found Tensorflow ICG version built from source :) -> use map coordinates from there..")
  import tensorflow.contrib.icg as tficg  
  my_warpimage = tficg.warpimage
  mgrad = tficg.python.ops.icg_ops.warpimage_gradients
else:
   assert False, "Needs %s built from source or ICG version of Tensorflow!"