#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 17:16:16 2017

@author: max
"""

#if __name__ == "__main__":
import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" # only allow second GPU to be visible to tensorflow..
#os.environ["CUDA_VISIBLE_DEVICES"] = "1" # only allow second GPU to be visible to tensorflow..
#    os.environ["CUDA_VISIBLE_DEVICES"] = "" # only allow second GPU to be visible to tensorflow..

import tensorflow as tf
import matplotlib
if os.name == 'posix' and "DISPLAY" not in os.environ:
    print ("no display found using Agg backend")
    matplotlib.use('Agg')
import matplotlib.pyplot as plt


from tensorflow.contrib.util import loader
from tensorflow.python.platform import resource_loader
import scipy.ndimage as nd


from scipy import ndimage
import numpy as np

import time
plt.ion()


import mapcoordinates
my_mapcoordinates = mapcoordinates.my_mapcoordinates
mgrad = mapcoordinates.mgrad

try:       
    import scipy as sp       
    I1col = sp.misc.face().astype(dtype)
    if np.issubdtype(dtype,np.float):
        I1col /= 255.0    
except:
    # I1col = np.random.randn(100,100,3) + np.arange(100,100,3)
    I1col = np.random.randn(100,100,3)/100 + np.atleast_3d( np.sum(np.meshgrid(np.arange(0,100)+0.1,np.arange(0,100)-0.1),axis=0))
    
cnt = 15
I1col = I1col[15:15+cnt,15:15+cnt,:]
I1col = np.random.rand(*I1col.shape)*1
   
N= 1
I1_batch4D = np.stack((I1col,)*N,axis=0) #NHWC
wn,wy,wx,wc = I1_batch4D.shape

C_np = np.meshgrid(np.arange(0,wx)  + .1 , np.arange(0,wy)-0.1)
# Watch out if 
#    assert False

# avoid coordinate transformations at the boundary 
# => +/- eps of numerical gradient would jump to other neighbourhood => not valid
#    C_np = [ np.random.rand(wy,wx)*0.95 + 0.02,]*2

Coords3D = np.stack((C_np[1],C_np[0]),axis=0) # 2HW
Coords4D = np.stack((Coords3D,)*N,axis=0) # N2HW

#    plt.figure(1);plt.clf();plt.imshow(np.concatenate(I1_batch4D,axis=1),"gray");plt.title("raw");plt.pause(1e-6)


######
config = tf.ConfigProto()
### Allow memory growth => Use memory as needed, easy but not as fast
config.gpu_options.allow_growth=True
#sess = tf.InteractiveSession(config=config)    
sess = tf.Session(config=config)    

print("session started...")
          
TFDTYPE = tf.float32;      NPDTYPE = np.float32    ;MAXERROR = 1e-3
#    TFDTYPE = tf.float64;      NPDTYPE = np.float64    ;MAXERROR = 1e-7

eps = 1e-3
      



#    for idd ,dev in enumerate( ["/cpu:0"]):#,"/gpu:0"]):    
#    for idd ,dev in enumerate( ["/gpu:0"]):#,"/gpu:0"]):
for idd ,dev in enumerate( ["/cpu:0","/gpu:0"]):
#    for idd ,dev in enumerate( ["/cpu:0","/cpu:0","/cpu:0","/cpu:0"]):
#    for idd ,dev in enumerate( ["/gpu:0","/gpu:0","/gpu:0","/gpu:0"]):
#    for idd ,dev in enumerate( ["/cpu:0","/gpu:0","/cpu:0","/gpu:0","/cpu:0","/gpu:0","/cpu:0","/gpu:0"]):    
    with tf.device(dev):


        I_test = I1_batch4D.copy()
        C_test = Coords4D.copy()
        ImgTest_tf = tf.constant(I_test,dtype=TFDTYPE)
        CordsTest_tf = tf.constant(C_test,dtype=TFDTYPE)
        
        y_gradIn = np.ones(ImgTest_tf.shape.as_list())
    
            
        x_lst = [ImgTest_tf,CordsTest_tf]
        y_tf =  my_mapcoordinates(*x_lst,interp_type="BILINEAR") #"BILINEAR","BICUBIC_2POINTS","BICUBIC_4POINTS"
        y_res_np, = sess.run([y_tf])
        xdata_lst = [I_test,C_test]
        
        def calcGrad_lx  (x_lst, y_tf, xdata_lst,order=2):
            """
            Calculates an analytic gradient for the operator that maps between 
            x_tf and y_tf
            It does this by converting the output to a scalar using an l2 loss
             y_tf =   op(x_tf) 
             tf.reduce_sum( y_tf**2)
            and varyying every input by tiny amounts
            
            => for comparison, build the same setup l2(op(x_tf)) with the analytic gradient fucntion
            
            """
            grad_lst = []
            if order == 1:
              loss_tf = tf.reduce_sum( tf.cast( tf.abs(y_tf), tf.float64) )
            elif order == 2:
              loss_tf = tf.reduce_sum( tf.cast(y_tf**2, tf.float64) )
            else:
              assert  False, "order is either 1 (l1 norm) or 2 (l2 norm)"
            for xi,xidata in zip(x_lst,xdata_lst):
             
                xidata = xidata.copy()
                xi_grad = np.empty_like(xidata)
                xi_el = np.product(xidata.shape)
                for idx in range( xi_el ):
                    xidata.ravel()[idx] += eps
                    
                    loss_pos, = sess.run([loss_tf],feed_dict={xi:xidata})
                    
                    xidata.ravel()[idx] -= 2*eps
                    loss_neg, = sess.run([loss_tf],feed_dict={xi:xidata})
                    
                    xidata.ravel()[idx] += eps
                               
                    xi_grad.ravel()[idx] =   (loss_pos-loss_neg)/(2*eps)
                
                grad_lst.append(xi_grad)
            return grad_lst
            
            
        def calcGrad (x_lst, y_tf, xdata_lst):        
            """
            Calculates an analytic gradient for the operator that maps between 
            x_tf and y_tf
            It does this by calculating the jacobi matrix 
            (derivation of all outputs to all inputs)
            and varyying every input by tiny amounts
            
            The jacobi matrix can become huge, so this can take a while
            
            """
            grad_lst = []
            for xi,xidata in zip(x_lst,xdata_lst):
             
                xidata = xidata.copy()
                xi_el = np.product(xidata.shape)
                y_gradIn_el = np.product(y_gradIn.shape)
                xi_Jacobi = np.empty((xi_el,y_gradIn_el))
                
#                    y_tf =  my_mapcoordinates(*x_lst)
                
                for idx in range( xi_el ):
                    xidata.ravel()[idx] += eps
                    
                    ypos, = sess.run([y_tf],feed_dict={xi:xidata})
                    ypos = ypos
                    
                    xidata.ravel()[idx] -= 2*eps
                    yneg, = sess.run([y_tf],feed_dict={xi:xidata})
                    yneg = yneg
                    
                    xidata.ravel()[idx] += eps
                               
                    xi_Jacobi[idx,:] =  ( (ypos-yneg)/(2*eps) ).ravel()
                
                xi_grad = np.reshape(xi_Jacobi @ np.reshape(y_gradIn,(y_gradIn.size,1)) ,xidata.shape)
                grad_lst.append(xi_grad)
            return grad_lst
        


        
        reslstCalc = []
        reslstAna = []
            
        for i in range (1):                
            
            #analytic gradients with lx norm
            order = 1
            if order == 1:
              loss_tf = tf.reduce_sum( tf.cast( tf.abs(y_tf), tf.float64) )
            elif order == 2:
              loss_tf = tf.reduce_sum( tf.cast(y_tf**2, tf.float64) )

            calcGrads_lx = tf.gradients(ys=loss_tf,
                                 xs=[ImgTest_tf,CordsTest_tf],
                                 grad_ys=NPDTYPE(1) )
            anagrad_lx,loss_np = sess.run([calcGrads_lx,loss_tf])     
            
            #numeric gradients l2 loss based                
            grad_lst_lx = calcGrad_lx (x_lst=x_lst, y_tf=y_tf,xdata_lst = xdata_lst,order=order);

            for el1,el2 in zip (anagrad_lx,grad_lst_lx):
                print( np.allclose(el1,el2,atol = MAXERROR) )
#                    print( np.allclose(el1,el2,atol= 1e-3) )
            
            angrad_img,anagrad_coords= anagrad_lx
            reslstAna += [anagrad_lx]
            grad_lst = grad_lst_lx
            reslstCalc += [grad_lst_lx]

                
                
                
#                #analytic gradients input gradient
#                calcGrads = tf.gradients(ys=y_tf,
#                                     xs=[ImgTest_tf,CordsTest_tf],
#                                     grad_ys=y_gradIn.astype(NPDTYPE) )
#                angrad_img,anagrad_coords= sess.run(calcGrads)     
#                
#                
#                #numeric gradients with input gradient field
#                grad_lst = calcGrad (x_lst=x_lst, y_tf=y_tf,xdata_lst = xdata_lst);
#                reslstCalc += [grad_lst]

            
            
            
            
#                out = mgrad(ImgTest_tf,CordsTest_tf,y_gradIn.copy())
#                out.grad_coords,out.grad_img
#                out_res, = sess.run([out],{ImgTest_tf:I_test, CordsTest_tf: C_test })
#                out_res.grad_img,out_res.grad_coords
            
#                angrad_img,anagrad_coords =  out_res.grad_img,out_res.grad_coords
            reslstAna += [ (angrad_img,anagrad_coords) ]
        
 
            def myplt(figid,analytic,num,title):
                if num.ndim == 3:
                    wdy,wdx,wdc= num.shape
                elif num.ndim == 4:
                    wdn, wdy,wdx,wdc= num.shape
                    num = num[0,:,:,:]
                else:
                    assert False, "size not as expected"
                border = np.zeros((wdc,1,wdy)).T
#                    plt.figure(figid);
#                    plt.clf();284502945
                if analytic is None:
                    analytic = np.zeros_like(num)
#                    pltvar = np.hstack((num,border,analytic,border,np.abs(num-analytic ) ))
                if wdc ==1:
#                        pltvar = pltvar[:,:,0]
                    num = num[:,:,0].copy()
                    analytic = analytic[:,:,0].copy()
                  
                avg = np.average( [np.abs(analytic),np.abs(num)])
#                    plt.imshow(pltvar);
                
                fig,axes = plt.subplots(num=figid, ncols=3,clear=True)
#                    fig.subplots_adjust(wspace=0.01)               
                fig.suptitle(title)
                axes[0].imshow(num)
                axes[0].set_title("Numeric")
                
                axes[1].imshow(analytic)
                axes[1].set_title("Analytic")
                
                delta = np.abs(num-analytic)/avg
                
                
                axes[2].imshow(delta)
                axes[2].set_title("rel. Delta\nA:%.0e  M:%.0e" %(np.average(delta),np.max(delta)))

                
#                    plt.tight_layout()
#                    plt.title(title +"\n numeric, analytic, delta");
                plt.pause(1e-6);
                
            deltaMax = MAXERROR
            
            imgrad_avg =   (np.average(np.abs(angrad_img)))
            coordgrad_avg =   (np.average(np.abs(anagrad_coords)))
            imgdelta = np.max(np.abs(angrad_img-grad_lst[0])) / imgrad_avg
            xdelta = np.max(np.abs((anagrad_coords-grad_lst[1])[:,0,:,:])) /coordgrad_avg
            ydelta = np.max(np.abs((anagrad_coords-grad_lst[1])[:,1,:,:])) / coordgrad_avg
            print(dev+", im grad delta:",imgdelta, " <-OK" if imgdelta < deltaMax else " <- TOO BIG!" )
            print(dev+", y grad delta:",xdelta, " <-OK" if xdelta < deltaMax else " <- TOO BIG!" )
            print(dev+", x grad delta:",ydelta, " <-OK" if ydelta < deltaMax else " <- TOO BIG!" )
            print ("---")
  
            if imgdelta > deltaMax or xdelta > deltaMax or ydelta > deltaMax:
                print ("\n\n ERROR DIFFERENCE TOO BIG!!! \n\n")
            if True:
                if N<=1:
                    I1w = np.empty_like(I1col)
                    for idc in range(I1col.shape[2]):
                        I1 = I1col[:,:,idc]
                        Coords = np.array([C_test[0,0,:,:].flatten(),C_test[0,1,:,:].flatten()])            
                        Iwres_spNDI = (ndimage.map_coordinates(I1,Coords,mode="constant",order=1) )#.reshape(I2.shape)
                        Iw_spNDI = np.reshape(Iwres_spNDI,I1.shape) 
                        I1w[:,:,idc] = Iw_spNDI            
                    
                    if y_res_np.ndim == 4:
                        y_res_np = y_res_np[0,:,:,:]
                    
                    
                    myplt(2+idd , I1w, y_res_np,  "NP map coordinates vs TF "+ dev )
                    myplt(4+idd , np.squeeze(angrad_img), np.squeeze(grad_lst[0]) , "grad wrt. IMG) "+ dev )
                    myplt(6+idd , anagrad_coords[0,0,:,:,np.newaxis], grad_lst[1][0,0,:,:,np.newaxis] , "grad wrt coords y "+ dev )
                    myplt(8+idd , anagrad_coords[0,1,:,:,np.newaxis], grad_lst[1][0,1,:,:,np.newaxis] , "grad wrt coords x "+ dev )
                    
#                        plt.figure(2+idd);plt.clf();plt.imshow(y_res_np);plt.title("out "+ dev);plt.pause(1e-3);
#                        plt.figure(4+idd);plt.clf();plt.imshow(np.squeeze(np.abs(angrad_img-grad_lst[0])),"gray");plt.title("grad IMG"+ dev);plt.colorbar();plt.pause(1e-3)
#                        plt.figure(4+idd);plt.clf();plt.imshow(np.squeeze(np.abs(grad_lst[0])),"gray");plt.title("grad IMG"+ dev);plt.colorbar();plt.pause(1e-3)
#                        
#                        plt.figure(6+idd);plt.clf();plt.imshow(np.squeeze((np.abs(anagrad_coords-grad_lst[1])[0,0,:,:])),"gray");plt.title("grad coords y"+ dev);plt.colorbar();plt.pause(1e-3)
#                        plt.figure(8+idd);plt.clf();plt.imshow(np.squeeze((np.abs(anagrad_coords-grad_lst[1])[0,1,:,:])),"gray");plt.title("grad coords x"+ dev);plt.colorbar();plt.pause(1e-3)
            
    
#                        plt.figure(10+idd);plt.clf();plt.imshow(y_res_np-I1w);plt.title("out delta "+ dev);plt.pause(1e-3);
#                        print(dev+", out delta:",np.sum(np.abs((y_res_np-I1w))))
#            

                
for i in range (len(reslstAna)):
    print ("Analytic Gradient change Img idx0 -> %d :"%i ,np.sum(np.abs(reslstAna[0][0] - reslstAna[i][0])))
    print ("Analytic Gradient change  x  idx0 -> %d :"%i ,np.sum(np.abs(reslstAna[0][1] - reslstAna[i][1])))       
for i in range (len(reslstCalc)):
    print ("Calc Gradient change Img idx0 -> %d :"%i ,np.sum(np.abs(reslstCalc[0][0] - reslstCalc[i][0])))
    print ("Calc Gradient change  x  idx0 -> %d :"%i ,np.sum(np.abs(reslstCalc[0][1] - reslstCalc[i][1])))   
#exit()
#        
#        plt.pause(3);
        
    

#    
#def _compute_numeric_jacobian(x, x_shape, x_data, y, y_shape, delta,
#
#  jacobian = np.zeros((x_size, y_size), dtype=x_dtype)
#  # For each of the entry of x, we slightly perturbs this by adding and
#  # subtracting a delta and then compute difference between the outputs. This
#  # will give us one row of the Jacobian matrix.
#  for row in range(x_size):
#    x_pos = x_data.copy()
#    x_neg = x_data.copy()
#    
#    x_pos.ravel().view(x_dtype)[row] += delta
#    y_pos = y.eval(feed_dict=_extra_feeds(extra_feed_dict, {x: x_pos}))
#    
#    x_neg.ravel().view(x_dtype)[row] -= delta
#    y_neg = y.eval(feed_dict=_extra_feeds(extra_feed_dict, {x: x_neg}))
#    diff = (y_pos - y_neg) / (2*delta)
#    jacobian[row, :] = diff.ravel().view(y_dtype)
#
#  logging.vlog(1, "Numeric Jacobian =\n%s", jacobian)
#  return jacobian