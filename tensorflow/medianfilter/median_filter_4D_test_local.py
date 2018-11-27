#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 12:46:08 2017

@author: max
"""
import tensorflow as tf 
import scipy as sp
import numpy as np
import os
import matplotlib
if os.name == 'posix' and "DISPLAY" not in os.environ:
    print ("no display found using Agg backend")
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tensorflow.contrib.util import loader
from tensorflow.python.platform import resource_loader


filtertype="SHAREDMEMORY"
filtertype="SIMPLE"
MEDSIZE = 3
if tf.test.is_built_with_cuda():
  lib = "TFMedianfilterOperator"
  print ("using locally built %s function..." % lib)
  dir_path = os.path.dirname(os.path.realpath(__file__))
  lib_path = os.path.join(dir_path,"../../lib/tf",lib + ".so")
  print (lib_path)
  if os.path.isfile(lib_path):
      if "lib_so" in locals():
          print("re-using already loaded %s function" % lib)
      else:
          lib_so = loader.load_op_library(lib_path)
          print( "Found Ops:",[name for name in dir (lib_so) if name[0] != "_"])

          ###########################
          # CUSTOM LOAD code    
          median_filter_4D = lib_so.filterexamples_median_filter4d
          median_filter_4D_gradient = lib_so.filterexamples_median_filter4d_gradient
          from tensorflow.python.framework import ops 

          @tf.RegisterGradient("FilterexamplesMedianFilter4d")
          def _MedianFilter4dGrad(op, grad):
              grad_med = median_filter_4D_gradient(op.inputs[0], grad,filtersize= op.get_attr("filtersize"),filtertype= op.get_attr("filtertype"), debug_indices =op.get_attr("debug_indices"))
              return [grad_med]

  elif hasattr (tf.contrib,"icg"):
      print ("found Tensorflow ICG version built from source :) -> use median filter from there..")
      import tensorflow.contrib.icg as tficg  
      median_filter_4D = tficg.median_filter_4D

  else:
      assert False, lib + "_lib.so not found"

  #
  tf.reset_default_graph() # Start with an empty graph
  ###### Create a new session that uses minimal RAM
  config = tf.ConfigProto()
  ### Allow memory growth => Use memory as needed, easy but not as fast
  config.gpu_options.allow_growth=True
  sess = tf.InteractiveSession(config=config)    
  #sess = tf.Session(config=config)    

  wn=1
  wx=640
  wy=480
  wc = 3
  if wn == 0 and wc == 0:
     inp = (10+100*np.random.rand(wy,wx)).astype(np.int32)
#     inp = (np.ones((wy,wx))).astype(np.int32)
  else:
     inp = np.random.rand(wn,wy,wx,wc).astype(np.int32)
  
  #result = median_filter_4D([[5, 4, 3, 2, 1],[5, 4, 3, 2, 1]]).eval()
  #  expectedOut = [[6, 5, 4, 3, 2],[6, 5, 4, 3, 2]]
  debug = True
  debug = False
  idx = median_filter_4D(inp,filtersize=str(MEDSIZE),filtertype=filtertype,debug_indices = True).eval()
#  idx = np.clip(idx,0,10000)
  result = median_filter_4D(inp,filtersize=str(MEDSIZE),filtertype=filtertype).eval()
#  result = median_filter_4D(inp.T,debug).eval().T
  
  doGradientCheck = False
  doGradientCheck = True
  if doGradientCheck:
    #    for idd ,dev in enumerate( ["/cpu:0"]):#,"/gpu:0"]):    
#    for idd ,dev in enumerate( ["/gpu:0"]):#,"/gpu:0"]):
#    for idd ,dev in enumerate( ["/cpu:0","/gpu:0"]):
#    for idd ,dev in enumerate( ["/cpu:0","/cpu:0","/cpu:0","/cpu:0"]):
#    for idd ,dev in enumerate( ["/gpu:0","/gpu:0","/gpu:0","/gpu:0"]):
    wn1=1
    wx1=10
    wy1=10
    wc1=3
    if not "inp1" in locals():
      print ("regenerating data")
      #inp1 = (10+100*np.random.rand(wy1,wx1)).astype(np.float64)
      inp1 = np.reshape(np.random.choice(1000000, wn1*wc1*wx1*wy1, replace=False)*10.0,(wn1,wy1,wx1,wc1))
    inp1_tf = tf.constant(inp1)    
    # for idd ,dev in enumerate( ["/cpu:0","/gpu:0","/cpu:0","/gpu:0","/cpu:0","/gpu:0","/cpu:0","/gpu:0"]):    
#    for idd ,dev in enumerate( ["/cpu:0","/cpu:0","/cpu:0","/cpu:0","/cpu:0","/cpu:0","/cpu:0","/cpu:0"]):    
#    for idd ,dev in enumerate( ["/gpu:0","/gpu:0","/gpu:0","/gpu:0","/gpu:0","/gpu:0","/gpu:0","/gpu:0"]):        
    for idd ,dev in enumerate( ["/cpu:0","/gpu:0","/cpu:0","/gpu:0"]):    
    # for idd ,dev in enumerate( ["/gpu:0"]):    
        with tf.device(dev):
          
#            del inp1_tf
#            
#            inp1 = np.reshape(np.random.choice(1000000, wn1*wc1*wx1*wy1, replace=False)*10.0,(wn1,wy1,wx1,wc1))
#            inp1_tf = tf.constant(inp1)  

            result_tf = median_filter_4D(inp1_tf,filtersize=str(MEDSIZE),filtertype=filtertype)
#            resJ, = tf.test.compute_gradient([inp1_tf],
#                                     [inp1_tf.shape.as_list()],
#                                     result_tf,
#                                     result_tf.shape.as_list(),
#                                     )
##            print ("gradient",resJ)
#            res = tf.test.compute_gradient_error([inp1_tf],
#                                     [inp1_tf.shape.as_list()],
#                                     result_tf,
#                                     result_tf.shape.as_list(),
#                                     )
#            
            result_np, = sess.run([result_tf])
            xdata_lst = [inp1]
            x_lst = [inp1_tf]
            y_gradIn = np.ones_like (result_np)
            
            def calcGrad (x_lst, y_tf, xdata_lst,y_gradIn, eps=1e-3):
                grad_lst = []
                for xi,xidata in zip(x_lst,xdata_lst):
                 
                    xidata = xidata.copy().astype(np.float64)
                    xi_el = np.product(xidata.shape)
                    y_gradIn_el = np.product(y_gradIn.shape)
                    xi_Jacobi = np.empty((xi_el,y_gradIn_el)).astype(np.float64)
                    
                    for idx in range( xi_el ):
                        back = xidata.ravel()[idx]
                        xidata.ravel()[idx] += eps
                        
                        ypos, = sess.run([y_tf],feed_dict={xi:xidata})
                        ypos = ypos.astype(np.float64)
                        
                        xidata.ravel()[idx] -= 2*eps
                        yneg, = sess.run([y_tf],feed_dict={xi:xidata})
                        yneg = yneg.astype(np.float64)
                        
                        xidata.ravel()[idx] = back
                                   
                        xi_Jacobi[idx,:] =  ( (ypos-yneg)/(2*eps) ).ravel()
                    
                    xi_grad = np.reshape(xi_Jacobi @ np.reshape(y_gradIn,(y_gradIn.size,1)) ,xidata.shape)
                    grad_lst.append(xi_grad)
                return grad_lst,xi_Jacobi
            
            def calcJacobi_ana(x_lst, y_tf):
                outshape = y_tf.shape.as_list()
                inshape = x_lst[0].shape.as_list()
                xi_el = np.product(inshape)
                y_gradIn_el = np.product(outshape)
                Jacobi = np.empty((xi_el,y_gradIn_el)).astype(np.float64)
                #
                y_gradIn_feed = np.zeros(outshape)
                y_gradIn_tf = tf.constant(y_gradIn_feed)
                
                calcGradAna = tf.gradients(result_tf,inp1_tf,y_gradIn_tf)
                for idx in range( y_gradIn_el ):
                  y_gradIn_feed.ravel()[idx] = 1
                  currGrad, = sess.run(calcGradAna, {y_gradIn_tf:y_gradIn_feed})
                  y_gradIn_feed.ravel()[idx] = 0
                  
                  Jacobi[idx,:] = currGrad.ravel()
                return Jacobi
              
            jAna = calcJacobi_ana(x_lst,result_tf)
            gradNumeric,jNumeric = calcGrad(x_lst,result_tf,xdata_lst,y_gradIn)
            gradNumeric = gradNumeric[0]
            gradCalc, = sess.run(tf.gradients(result_tf,inp1_tf,y_gradIn))
            
            gradDelta = np.sum(np.abs(gradNumeric- gradCalc))
            print ("gradient delta (%s)"%dev, gradDelta)
#            print ("gradient",np.sum(res),res)
            
            def myplt(figid,analytic,num,title):
                if num.ndim == 2:
                    wdy,wdx= num.shape
                    wdc = 0
                    border = np.zeros((1,wdy)).T
                elif num.ndim == 3:
                    wdy,wdx,wdc= num.shape
                    border = np.zeros((wdc,1,wdy)).T
                elif num.ndim == 4:
                    wdn, wdy,wdx,wdc= num.shape
                    num = num[0,:,:,:]
                    analytic = analytic[0,:,:,:]
                    border = np.zeros((wdc,1,wdy)).T
                else:
                    assert False, "size not as expected"
                
                plt.figure(figid);
                plt.clf();
                if analytic is None:
                    analytic = np.zeros_like(num)
                pltvar = np.hstack((num,border,analytic,border,np.abs(num-analytic ) ))
                if wdc ==1:
                    pltvar = pltvar[:,:,0]
                plt.imshow(pltvar);
                plt.tight_layout()
                plt.title(title +"\n numeric, analytic, delta");
                plt.pause(1e-6);
                
#            resJ0 = np.reshape(resJ[0] @ np.reshape(y_gradIn,(y_gradIn.size,1)) ,inp1.shape)
#            resJ1 = np.reshape(resJ[1] @ np.reshape(y_gradIn,(y_gradIn.size,1)) ,inp1.shape)
            myplt(10,gradCalc,gradNumeric,"my Delta " +dev )
#            myplt(11,resJ0,resJ1,"delta (TF)" + dev)
#            myplt(12,resJ[0],resJ[1],"delta (TF)"+dev)
            myplt(13,jNumeric,jAna.T,"delta (jacobßi own)" + dev)
            
  print ("end of gradient test...")            

# %%             
      
#  assert False
#  expectedOut = inp +1
  
  
  import matplotlib.pyplot as plt
  import scipy.ndimage as nd
  
  if inp.ndim == 4:
    expectedOut = np.empty_like(inp)
    for idn in range(wn):
      expectedOut[idn,:,:,:] = nd.filters.median_filter(inp[idn,:,:,:], size = (MEDSIZE,MEDSIZE,1),mode="constant")
  else:
    expectedOut = nd.filters.median_filter(inp, size = (MEDSIZE,MEDSIZE),mode="constant")
  
  repeat = 100;
  print ("starting a functional test against scipy.ndimage.filters.median_filter for %d times..."%repeat)            
  for i in range(repeat):
    wx = int(np.random.uniform(2,1000))
    wy = int(np.random.uniform(2,1000))
    if wn == 0 and wc == 0:
      inp = (10+100*np.random.rand(wy,wx)).astype(np.int32)
      expectedOut = nd.filters.median_filter_4D(inp, size = (MEDSIZE,MEDSIZE),mode="constant")
    else:
      inp = (10+100*np.random.rand(wn,wy,wx,wc)).astype(np.int32)
      expectedOut = nd.filters.median_filter(inp, size = (1,MEDSIZE,MEDSIZE,1),mode="constant")
    
    result = median_filter_4D(inp,filtersize=str(MEDSIZE),filtertype=filtertype).eval() 
    
    delta = result-expectedOut
    dsum = np.sum(np.abs(delta))
    if dsum != 0:
        print (wx,wy,dsum)
        break
  else:
    print("no errors found so far...")
  
  
  if np.all(result == expectedOut):
    print("result shape",result.shape)
    print(" - SIMPLE TEST OK-")
  elif np.all(result[2:-2,2:-2] == expectedOut[2:-2,2:-2]):  
      print(" - SIMPLE PARTLY OK (Corner Cases still wrong)-")
  else:
      print("SIMPLE TEST FAILD")

  # Simple Timing test
  # Run once before to allow tensorflow to do its optimizing magic
  # put values on 
  var = tf.Variable(inp,caching_device="/gpu:0")
  init = tf.global_variables_initializer()
  sess.run(init)
  median_filter_4D(var,filtertype=filtertype).eval()
  import time
  t1 = time.time()
  cnt = 10
  for i in range(cnt):
    median_filter_4D(var,filtertype=filtertype).eval()
  t2 = time.time()
  print ("timing measurement from python (ms):", (t2-t1)*1000/cnt)

else:
    print("FAIL: Tensorflow with CUDA Needed")
#delta = np.abs(result-expectedOut)    
if wn>= 1:
  plt.figure(1);plt.clf();plt.imshow (idx[0,:,:,:].astype(np.float32),"gray");plt.colorbar();plt.title("idx");plt.pause(1e-3)
  plt.figure(2);plt.clf();plt.imshow (result[0,:,:,:].astype(np.float32),"gray");plt.colorbar();plt.title("result");plt.pause(1e-3)
  plt.figure(3);plt.clf();plt.imshow( np.clip(np.abs(result-expectedOut),0,100)[0,:,:,:].astype(np.float32),"gray");plt.colorbar();plt.title(" delta ");plt.pause(1e-3)
else:
  plt.figure(1);plt.clf();plt.imshow (idx,"gray");plt.colorbar();plt.title("idx");plt.pause(1e-3)
  plt.figure(2);plt.clf();plt.imshow (result,"gray");plt.colorbar();plt.title("result");plt.pause(1e-3)
  plt.figure(3);plt.clf();plt.imshow( np.clip(np.abs(result-expectedOut),0,100),"gray");plt.colorbar();plt.title(" delta ");plt.pause(1e-3)
  
