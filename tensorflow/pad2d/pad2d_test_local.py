#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 13:04:56 2018

@author: hofinger
"""

#%% load dependencies and prepare session
import os
import tensorflow as tf
import matplotlib
if os.name == 'posix' and "DISPLAY" not in os.environ:
    print ("no display found using Agg backend")
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

######
config = tf.ConfigProto()
### Allow memory growth => Use memory as needed, easy but not as fast
config.gpu_options.allow_growth=True
if not "sess" in locals():
  sess = tf.InteractiveSession(config=config)    
  #sess = tf.Session(config=config)    


#%% load library
import numpy as np
#%% load library
import numpy as np
  
print ("Attention pad2D is NCHW only!")


lib = "TFpad2dOperator"
dir_path = os.path.dirname(os.path.realpath(__file__))
lib_path = os.path.join(dir_path,"../../lib/tf",lib + ".so")
print (lib_path)
if os.path.isfile(lib_path):
    print ("using locally built %s function..." % lib)
    if "lib_so" in locals():
        print("re-using already loaded %s function" % lib)
    else:
        lib_so = tf.load_op_library(lib_path)
        print( "Found Ops:",[name for name in dir (lib_so) if name[0] != "_"])

        ###########################
        # CUSTOM LOAD code 

        _pad2d_op_so = lib_so
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
elif hasattr (tf.contrib,"icg") and hasattr (tf.contrib.icg,lib):
  print ("found Tensorflow ICG version built from source :) -> use %s from there.."%lib)
  import tensorflow.contrib.icg as tficg  
  pad2d = tficg.pad2d
  pad2d_transpose = tficg.pad2d_transpose
else:
   assert False, "Needs %s built from source or ICG version of Tensorflow!"%lib
  
#%%
eps = 1e-5
padding = 6
modes = ["SYMMETRIC","REPLICATE"]

for sz in [20,int(1.5*padding), padding]:
  szT = sz+2*padding
  for wdc in [1,3]:
    for wdn in [1,3]:
  #    print ("Simply python example of transposed operators")
  
      # Simple example of pad operator and transpose operator: 
      # <K@u,v> = <u,K^T @ v> 
      # 0Padding is transpose to Cropping
      I1 = np.random.random( [wdn,wdc,sz,sz])#*0+1
      I2 = np.random.random( [wdn,wdc,szT,szT])#*0+2
      wdn,wdc ,wdy,wdx = I2.shape
      
  #    I1p = np.pad( I1,((0,0),(0,0),(padding,padding),(padding,padding)),"constant")
  #    s1 =np.sum(I1p*I2)
  #    I2c = I2[:,:,padding:wdy-padding,padding:wdx-padding]
  #    s2 =np.sum(I1*I2c)
  #    d = s1-s2
  #    print("numpy example, Delta :",d)
  #    if d <eps:
  #      print(" Test OK!")
  #    else:
  #      assert False, "Test Failed!"
      
  
      
      I1_4D = I1 # Batch,Color,Height,Width
      I2_4D = I2 # Batch,Color,Height,Width
      
      for mode in modes:
        # Standard operator
        I1p_4D = pad2d( I1_4D, mode, pad=padding).eval()
        # Transpose operator
        I2pT_4D = pad2d_transpose( I2_4D, mode, pad=padding)
        I2pT_4D = I2pT_4D.eval()
        
        ####################################################
        # Testing Good Cases:
        s1 = np.sum(I1p_4D * I2_4D)
        s2 = np.sum(I1_4D * I2pT_4D)
        d = s1-s2
        print("Tensorflow Pad2D example batches=%i,colours=%i (%s), Delta: "%(wdn,wdc,mode), d)
        if d <eps:
          print(" Test OK!")
        else:
          assert False, "Test Failed! with mode=" + mode + ",size="+ str(sz) + ",pad=" +str(padding)
  
        
        ####################################################
        # Testing Graph Building Failure Cases:
        print("Running Test for Exception Mechanism on 'pad2d'")
        found = False
        try:
          pad2d( np.zeros([wdn,wdc,padding-1,padding-1]), mode, pad=padding)
        except ValueError as e:
          found = True
        assert found, "Expected error message did not pop up!"
        print(" Test OK!")
        
        print("Running Test for Exception Mechanism  on 'pad2d_transpose'")
        found = False
        try:
          pad2d_transpose( np.zeros([wdn,wdc,3*padding-1,3*padding-1]), mode, pad=padding)
        except ValueError as e:
          found = True
        assert found, "Expected error message did not pop up!"
        print(" Test OK!")
        
        ####################################################
        # Testing Graph Runnin Failure Cases:
        print("Running Test for Exception Mechanism on 'pad2d'")
        found = False
        feed = tf.placeholder(shape=[wdn,wdc,None,None],dtype= I1_4D.dtype)
        op = pad2d( feed, mode, pad=padding)
        try:
          sess.run(op, feed_dict={feed:np.zeros([wdn,wdc,padding-1,padding-1]) })
        except tf.errors.InvalidArgumentError as e:
          found = True
        assert found, "Expected error message did not pop up!"
        print(" Test OK!")
        
        print("Running Test with FailureCase on 'pad2d_transpose'")
        found = False
        feed = tf.placeholder(shape=[wdn,wdc,None,None],dtype= I1_4D.dtype)
        op = pad2d_transpose( feed, mode, pad=padding)
        try:
          sess.run(op, feed_dict={feed: np.zeros([wdn,wdc,3*padding-1,3*padding-1]) })
        except tf.errors.InvalidArgumentError as e:
          found = True
        assert found, "Expected error message did not pop up!"
        print(" Test OK!")