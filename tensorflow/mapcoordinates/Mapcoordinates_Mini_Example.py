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

import mapcoordinates
my_mapcoordinates = mapcoordinates.my_mapcoordinates
mgrad = mapcoordinates.mgrad


#
tf.reset_default_graph() # Start with an empty graph
###### Create a new session that uses minimal RAM
config = tf.ConfigProto()
### Allow memory growth => Use memory as needed, easy but not as fast
config.gpu_options.allow_growth=True
#sess = tf.InteractiveSession(config=config)    
sess = tf.InteractiveSession(config=config)    


dtype = np.int16  #
       
I1 = sp.misc.face().astype(dtype)
if np.issubdtype(dtype,np.float):
  I1 /= 255.0
I1 = np.expand_dims(I1 @ [0.299, 0.587, 0.114],axis=2) # activate this line to use grayscale (y,x,0:3) -> (y,x,1)
I1 = np.expand_dims(I1,axis=0)
wdn,wdy,wdx,wdc = I1.shape

coordsX, coordsY = np.meshgrid(np.arange(wdx),np.arange(wdy))
coordsY = wdy/2  - coordsY/2 # scale by two and flip Y axis
coordsX = coordsX + 25.2
Coords4D = np.expand_dims(np.stack(( coordsY,coordsX)), axis=0).astype(dtype)

res = my_mapcoordinates(I1,Coords4D).eval(session=sess)
#plt.imshow(np.squeeze(res))
plt.figure(1); plt.clf()
fig, axes = plt.subplots(1, 4,num=1)
axes[0].imshow(np.squeeze(I1))
axes[1].imshow(np.squeeze(Coords4D[:,0,:,:]))
axes[2].imshow(np.squeeze(Coords4D[:,1:,:]))
axes[3].imshow(np.squeeze(res))



import tensorflow as tf 
#from tensorflow.contrib.icg import mapcoordinates
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt 

dtype = np.float32  #
       
I1 = (sp.misc.face()/ 255.0).astype(dtype)
I1 = np.expand_dims(I1,axis=0)
wdn,wdy,wdx,wdc = I1.shape

coordsX, coordsY = np.meshgrid(np.arange(wdx),np.arange(wdy))
coordsY = wdy/2  - coordsY/2 # scale by two and flip Y axis
coordsX = coordsX + 25.2
Coords4D = np.expand_dims(np.stack(( coordsY,coordsX)), axis=0).astype(dtype)

print ("I1 NHWC:",I1.shape, ",  Coords N2HWC:",Coords4D.shape)

Iwarped = my_mapcoordinates(I1,Coords4D).eval(session=sess)

plt.figure(1); plt.clf()
fig, axes = plt.subplots(1, 4,num=1)
axes[0].imshow(np.squeeze(I1))
axes[1].imshow(np.squeeze(Coords4D[:,0,:,:]))
axes[2].imshow(np.squeeze(Coords4D[:,1:,:]))
axes[3].imshow(np.squeeze(Iwarped))

