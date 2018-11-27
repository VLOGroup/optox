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


import warpimage
my_warpimage = warpimage.my_warpimage
mgrad = warpimage.mgrad

#%% Example using int16
tf.reset_default_graph() # Start with an empty graph
######
config = tf.ConfigProto()
### Allow memory growth => Use memory as needed, easy but not as fast
config.gpu_options.allow_growth=True
#sess = tf.InteractiveSession(config=config)    
sess = tf.Session(config=config)    


dtype = np.int16  #
       


I1 = sp.misc.face().astype(dtype)
if np.issubdtype(dtype,np.float):
  I1 /= 255.0
I1 = np.expand_dims(I1 @ [0.299, 0.587, 0.114],axis=2) # activate this line to use grayscale (y,x,0:3) -> (y,x,1)
I1 = np.expand_dims(I1,axis=0)
wdn,wdy,wdx,wdc = I1.shape

coordsX, coordsY = np.meshgrid(np.arange(wdx),np.arange(wdy))
coordsX = np.zeros_like (coordsX)
coordsY = coordsX.copy()

coordsY = coordsY +  wdy/2 - 2*np.arange(0,wdy)[:,np.newaxis] # flip image
coordsX = coordsX + 25.2

Coords4D = np.expand_dims(np.stack(( coordsX,coordsY)), axis=0).astype(dtype)

res = my_warpimage(I1,Coords4D).eval(session=sess)
#plt.imshow(np.squeeze(res))
plt.figure(1); plt.clf()
fig, axes = plt.subplots(1, 4,num=1)
axes[0].imshow(np.squeeze(I1))
axes[1].imshow(np.squeeze(Coords4D[:,0,:,:]))
axes[2].imshow(np.squeeze(Coords4D[:,1:,:]))
axes[3].imshow(np.squeeze(res))


#%% Example using float 32

tf.reset_default_graph() # Start with an empty graph
######
config = tf.ConfigProto()
### Allow memory growth => Use memory as needed, easy but not as fast
config.gpu_options.allow_growth=True
#sess = tf.InteractiveSession(config=config)    
sess = tf.Session(config=config)    


dtype = np.float32  #
       
I1 = (sp.misc.face()/ 255.0).astype(dtype)
I1 = np.expand_dims(I1,axis=0)
wdn,wdy,wdx,wdc = I1.shape

coordsX, coordsY = np.meshgrid(np.arange(wdx),np.arange(wdy))
coordsX = np.zeros_like (coordsX)
coordsY = coordsX.copy()

coordsY = coordsY +wdy - 2*np.arange(0,wdy)[:,np.newaxis] # flip image
coordsX = coordsX + 25.2

Coords4D = np.expand_dims(np.stack(( coordsX,coordsY)), axis=0).astype(dtype)

print ("I1 NHWC:",I1.shape, ",  Coords N2HWC:",Coords4D.shape)

Iwarped = my_warpimage(I1,Coords4D).eval(session=sess)

plt.figure(1); plt.clf()
fig, axes = plt.subplots(1, 4,num=1)
axes[0].imshow(np.squeeze(I1))
axes[1].imshow(np.squeeze(Coords4D[:,0,:,:]))
axes[2].imshow(np.squeeze(Coords4D[:,1:,:]))
axes[3].imshow(np.squeeze(Iwarped))


