# -*- coding: UTF-8 -*-    
#coding=utf-8
"""
Created on Sat Feb 12 11:15:02 2022

@author: leixi
"""

import os
from PIL import Image
import sys
import tensorflow as tf
from matplotlib import pyplot
from keras import datasets, layers, models
import matplotlib.pyplot as plt
from matplotlib import image
from matplotlib import pyplot
import scipy.io as sio
import random
from numpy.random import seed 
from keras.callbacks import EarlyStopping
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from numpy import expand_dims
from keras.regularizers import Regularizer
from keras import regularizers
import numpy as np
seed(1) 
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import atomai as aoi
import pandas as pd
import time
import cnntrainv2_funclib as flib
import matlab.engine
from scipy.interpolate import griddata
from skopt.callbacks import EarlyStopper
import cv2
from decimal import Decimal
eng = matlab.engine.start_matlab()
import matplotlib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors

def addmask(a,cmap,vminn,vmaxx):
    a = a.astype('float')
    ia=np.argwhere(a==0)
    for ii in range(0,ia.shape[0]):
        a[ia[ii][0],ia[ii][1]]=float('nan')
    ax.set_facecolor('w')
    # plt.colorbar()
    plt.show()
    masked_array = np.ma.array (a, mask=np.isnan(a))
    cmap.set_bad('white')
    ax.imshow(masked_array, interpolation='nearest', cmap=cmap, vmin=vminn, vmax=vmaxx)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
def p_root(value, root):
     
    root_value = 1 / float(root)
    return round (Decimal(value) **
             Decimal(root_value), 3)
 
def minkowski_distance(x, y, p_value):
    return (p_root(sum(pow(abs(a-b), p_value)
            for a, b in zip(x, y)), p_value))


class StoppingCriterion(EarlyStopper):
  def __init__(self, delta=0.05, n_best=10):
    super(EarlyStopper, self).__init__()
    self.delta = delta
    self.n_best = n_best
def _criterion(self, result):
  if len(result.func_vals) >= self.n_best:
    func_vals = np.sort(result.func_vals)
    worst = func_vals[self.n_best - 1]
    best = func_vals[0]
    return abs((best - worst)/worst) < self.delta
  else:
    return None

def normalize(figureout,maxx):
    index1=np.argwhere(figureout>maxx/2)
    index2=np.argwhere(figureout<=maxx/2)
    figureout[index1[:,0],index1[:,1]]=maxx
    figureout[index2[:,0],index2[:,1]]=0
    return figureout
def inputvae(inpp,inpang,latdim):
    zlats=np.zeros((latdim,2))
    zangs=np.zeros((1,2))
    
    for ii in range(1,latdim+1):
        zlats[ii-1,:]=inpp[ii-1]
    zangs[0,:]=inpang
    return zlats, zangs

def cropimg(imagedata):
    image_array = np.array(imagedata)
    print(image_array.shape)
    dxx=int(50*1.5)
    rr=1.5 ## rr=1  dingge

    image_arraynew0 = image_array[int((dxx+27)/rr):int((-dxx-37)/rr),int((dxx+40)/rr):int((-dxx+10)/rr)]
    print(str(np.min(image_arraynew0))+'_'+str(np.max(image_arraynew0)) )

    imagedata = Image.fromarray(image_arraynew0)
    #imagedata = imagedata.resize((dshape,dshape),Image.ANTIALIAS)    
    image_arraynew0 = np.array(imagedata)
    image_arraynew= image_arraynew0
    #image_arraynew=(image_arraynew0-np.min(image_arraynew0))/(255-np.min(image_arraynew0))*255
    image_arraynew=(255-image_arraynew)/255
    print(str(np.min(image_arraynew))+'_'+str(np.max(image_arraynew)) )

    imagedatanew = Image.fromarray(image_arraynew)
    return imagedatanew

def getytarget(ii,rot=120): 
    readstringd = ''
    if ii==2:
        filestring=readstringd+"saddle0.016_figout"
        screensize=1.0

    elif ii==43:
        filestring=readstringd+"peanutsad0.0241.20.8_figout"
        screensize=1.0    
    elif ii==431:
        filestring=readstringd+"peanutsad0.0241.20.8_figout"
        screensize=1.5
    elif ii==432:
        filestring=readstringd+"peanutsad0.02721.20.8_figout"
        screensize=1.5
    elif ii==923:
        filestring=readstringd+"flower0.0241.550.8_figout"#readstringd+"flower0.01610.8_figout"
        screensize=1.5   
    elif ii==1121:
        filestring=readstringd+"pyr0.0169711_figout"#"pyr0.0148491_figout"
        screensize=1.5       
    elif ii==1123:
        filestring=readstringd+"pyr0.02551.7_figout"#"pyr0.0148491_figout"
        screensize=1.5     
    elif ii==1124:
        filestring=readstringd+"pyr0.02251.5_figoutflat"#"pyr0.0148491_figout"
        screensize=1.5            
    elif ii==92:
        filestring=readstringd+"flower0.0161.20.8_figout"#readstringd+"flower0.01610.8_figout"
        screensize=1.0        
    elif ii==102:
        filestring=readstringd+"saddle0.0181.210_figout"
        screensize=1.0
    elif ii==21:
        filestring=readstringd+"saddle0.024155_figout"
        screensize=1.5
    elif ii==81:
        filestring=readstringd+"boat0.0241.2_figout"
        screensize=1.5
    elif ii ==82:
        filestring=readstringd+"boat0.0240.8_figout"
        screensize=1.5
    elif ii ==83:
        filestring=readstringd+"boat0.0241fat1.3_figout"
        screensize=1.5
    sio.savemat('screensize.mat',{'screensize':screensize})
    
    image_array= sio.loadmat(filestring+'.mat')
    image_array= image_array['z1mod']

    image_array = image_array/maxheight#*1
    if ii ==1123:
        image_array =image_array*1.2
    if ii ==921:
        image_array =image_array
    if ii ==43:
        image_array =image_array*.75
    if ii==2 or ii==21:
        image_array = image_array+0.1    
    image_array[np.isnan(image_array)] = 0

    imagedata= Image.fromarray(image_array)
    print(image_array.shape)
    print(str(image_array[0,0]),'_',str(np.max(image_array)),'_',str(np.min(image_array)))

    
    imagedatarot=imagedata.rotate(rot, fillcolor=(0)) 
    data= np.array(imagedatarot)           
    ytarget = data
    
    xsize=np.array(imagedata).shape[0]
    ysize=np.array(imagedata).shape[1]    


    return ytarget,xsize,ysize
smooth = 1. # 
def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten()# 
    y_pred_f = y_pred.flatten()
    intersection = sum(y_true_f * y_pred_f)
    return (2. * intersection ) / (sum(y_true_f * y_true_f) + sum(y_pred_f * y_pred_f) )

def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)
    
def getypredict(num):
    readstringd = ''
    filestring=readstringd+"mvertr-test"+str(6)+"num"+str(num)+"_figout"

    image_array= sio.loadmat(filestring+'.mat')
    image_array= image_array['z1mod']
    image_array = image_array/maxheight#*1
    if indexy==2 or indexy==21:
        image_array = image_array+0.1
    image_array[np.isnan(image_array)] = 0

    imagedata= Image.fromarray(image_array)

    print(image_array.shape)
    print(str(image_array[0,0]),'_',str(np.max(image_array)),'_',str(np.min(image_array)))
   
    data= np.array(imagedata)           
    ytarget =data
    xsize=np.array(imagedata).shape[0]
    ysize=np.array(imagedata).shape[1]
    print(str(np.min(ytarget))+ '_'+str(np.max(ytarget)))

    return ytarget,xsize,ysize

def SSIMLoss(y_true, y_pred):
 
  return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0, filter_size=20))

def SSIMLossnum(y_true, y_pred):
  eii  = sio.loadmat("eii.mat")
  eii = eii['eii']
  print(len(eii[0]))
  y_true0 = tf.constant(y_true/1.0,dtype=tf.float64)
  y_pred0 = tf.constant(y_pred/1.0,dtype=tf.float64)  
  if len(eii[0])==1:
    return 1 - tf.reduce_mean(tf.image.ssim(y_true0, y_pred0, 1.0, filter_size=20)).numpy(), 0 
  else:
    dtt = 10
    dtt2 = 2
    y_true_e =np.zeros((1,len(range(30-dtt,30+dtt)),len(range(34-dtt2,34+dtt2)),1))
    y_pred_e =np.zeros((1,len(range(30-dtt,30+dtt)),len(range(34-dtt2,34+dtt2)),1))
    
    y_true_e[0,:,:,0]= y_true[0,30-dtt:30+dtt,34-dtt2:34+dtt2,0]      
    y_pred_e[0,:,:,0]= y_pred[0,30-dtt:30+dtt,34-dtt2:34+dtt2,0]      

    y_true0_e = tf.constant(y_true_e/1.0,dtype=tf.float64)
    y_pred0_e = tf.constant(y_pred_e/1.0,dtype=tf.float64)  
    print('e1'+str( tf.reduce_mean(tf.image.ssim(y_true0, y_pred0, 1.0, filter_size=5)).numpy()))
    print('e2'+str( tf.reduce_mean(tf.image.ssim(y_true0_e, y_pred0_e, 1.0, filter_size=2)).numpy()))
    msec = - 10*tf.reduce_mean(tf.image.ssim(y_true0_e, y_pred0_e, 1.0, filter_size=2)).numpy()
    outt = 1+msec
    print('ees'+str(msec)+ str(outt))    
    return np.float32(outt), np.float32(msec)
def SSIMLoss_ms(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim_multiscale(
        y_true, y_pred,  filter_size=5,
        filter_sigma=1.5, k1=0.01, k2=0.03
    ))

def SSIMLoss_msnum(y_true, y_pred):
  y_true0 = tf.constant(y_true/1.0,dtype=tf.float64)
  y_pred0 = tf.constant(y_pred/1.0,dtype=tf.float64)    
  return 1 - tf.reduce_mean(tf.image.ssim_multiscale(
        y_true0, y_pred0,  filter_size=5,
        filter_sigma=1.5, k1=0.01, k2=0.03
    ))
def getfuncsvaepred():
    stringsavemod='modelkirgamivaepredictor'+str(nrot)+str(latdim)+str(embedd)+str(usemsssim)+str(niter)

    json_file = open(stringsavemod+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = tf.keras.models.model_from_json(loaded_model_json)
    loaded_model.load_weights(stringsavemod+'.h5')
        
    return loaded_model
def getfunc():
    
    stringsavemod='modelkirgamiforwardnrotvert'+str(nrot)+str(latdim)+str(embedd)+str(usemsssim)+str(niter)       
    # load json and create model
    json_file = open(stringsavemod+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = tf.keras.models.model_from_json(loaded_model_json)
    loaded_model.load_weights(stringsavemod+'.h5')
    print("Loaded model from disk")
        
    return loaded_model

def getfuncmyvae(x, rt=1, pretrained=0):
    
    
    [zlatsold, zangsold]=flib.inputvae(x,0,latdim) ## latent space, latent angle space
    figureoutold=vae.manifoldnd(d=1,ll=zlatsold,cmap='viridis')
    print(figureoutold.shape)
    
    scalee = 1.2
    figureoutoldn = normalize(figureoutold,scalee)

    if rt==20 or pretrained==1:
        figureoutoldn = figureoutoldn+basescale*figureoutoldbase
    
        index1=np.argwhere(figureoutoldn>1/2)
        index2=np.argwhere(figureoutoldn<=1/2)
        figureoutoldn[index1[:,0],index1[:,1]]=1
        figureoutoldn[index2[:,0],index2[:,1]]=0    
    print(rt)        
    plt.figure(figsize=(15,15)) 
    ax = plt.subplot(1,1,1)
    vminn=0
    vmaxx=1
    plt.imshow(1-figureoutoldbase,cmap=plt.cm.jet, interpolation='nearest', vmin=vminn, vmax=vmaxx)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    addmask(figureoutoldbase,plt.cm.jet,vminn,vmaxx)
    plt.savefig('samplefinal.png', dpi=600, transparent=True)

    plt.figure(figsize=(15,15))
    ax = plt.subplot(1,1,1)
    plt.imshow(figureoutoldn,label = 'Validation loss')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    print('figaa'+figureoutoldn.shape)
    plt.savefig('sample.png', dpi=600, transparent=True)
    ok =1
    return ok
def getfuncrt():
    
    stringsavemod='modelkirgamiforwardnrotvert'+str(nrot)+str(latdim)+str(embedd)+str(usemsssim)+str(niter)       
    # load json and create model
    json_file = open(stringsavemod+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = tf.keras.models.model_from_json(loaded_model_json)
    loaded_model.load_weights(stringsavemod+'.h5')
    print("Loaded model from disk")
        
    return loaded_model
def getfuncimg():
    
    stringsavemod='modelkirgamiforwardnrotverte1'+str(nrot)+str(latdim)     
    # load json and create model
    json_file = open(stringsavemod+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = tf.keras.models.model_from_json(loaded_model_json)
    loaded_model.load_weights(stringsavemod+'.h5')
    print("Loaded model from disk")
        
    return loaded_model

def getfuncvae():
    vaeinp=aoi.models.load_model("my2d_modelvaevert"+str(nrot)+str(latdim)+"vae.tar")## input the whole model
    vaeinp=aoi.models.load_model("my2d_modelvaelotuscombined"+str(nrot)+str(latdim)+"vae.tar")## input the whole model

    vae=vaeinp
    return vae

def getfuncvaeall(inputnum,nrot,latdim):
    if inputnum==0:
        nrot=541
        stringss="my2d_modelvaevertcombined"+str(nrot)+str(latdim)+"vae.tar"
    elif inputnum==3:
        nrot=3
        stringss="my2d_modelvaelotuscombined"+str(nrot)+str(latdim)+"vae.tar"
    
    elif inputnum==31:
        stringss="my2d_modelvaelotuscombinedwp"
    elif inputnum==21:
        nrot=2
        stringss="my2d_modelvaevertcombinedwp"+str(nrot)+str(latdim)+"vae.tar"
    elif inputnum==20:
        stringss="my2d_modelvaevertcombinednp"+str(nrot)+str(latdim)+"vae.tar"
    
    elif inputnum==50:
        stringss="my2d_modelvaequad3np"+str(nrot)+str(latdim)+"vae.tar"

    elif inputnum==51:
        stringss="my2d_modelvaequad5np"+str(nrot)+str(latdim)+"vae.tar"
    elif inputnum==52:
        stringss="my2d_modelvaecircquad5np"+str(nrot)+str(latdim)+"vae.tar"
    elif inputnum==101:
        stringss="my2d_modelvaeflcircquad11nrotnp"+str(nrot)+str(latdim)+"vae.tar"


    vaeinp=aoi.models.load_model(stringss)## input the whole model
    print(stringss)
    vae=vaeinp
    return vae
def black_box_functionrt(x, noise_level=0.01):
    """Function with unknown internals we wish to maximize.

    This is just serving as an example, for all intents and
    purposes think of the internals of this function, i.e.: the process
    which generates its output values, as unknown.
    """
    wsize=sio.loadmat("wsize.mat")
    if wsize ==0:
        xall= x[0:-1]
    else:
        xall= x[0:-2]
    ok = getfuncmyvae(xall,rt, pretrained)
    print(x)
    if solid ==1:
            ok=eng.Main_trycutsimplified_verticalcuts_arb_funcwsize3d(float(x[-2]), float(x[-1])/10)
       
    else:
        if wsize ==0:
            ok=eng.Main_trycutsimplified_verticalcuts_arb_func(float(x[-1])/10)   
        else:
            ok=eng.Main_trycutsimplified_verticalcuts_arb_funcwsize(float(x[-2]), float(x[-1])/10)
    prestretch=sio.loadmat("prestretch.mat")
    prestretch=prestretch['prestretch']
    print(str(prestretch)+str(x[-1]))
    ytarget,xsizetarget,ysizetarget=getytarget(indexy,grot) ## 3d
    print('next')
    
    ypred,xsizepred,ysizepred = getypredict(ncount) ## 3d
    print(str(xsizetarget)+str(ysizetarget)+str(xsizepred)+str(ysizepred))
    ypred.resize((1,dshape,dshape,1))
    ytarget.resize((1,dshape,dshape,1))
    print(str(np.min(ytarget))+ '_+'+str(np.max(ytarget))+ '_+'+str(np.min(ypred)) + '_'+str(np.max(ypred)))

    plt.figure(figsize=(15,10))
    ax=plt.subplot(1,2,1)     #
    plt.imshow(ypred[0,:,:,0],cmap=plt.cm.jet,vmin=vminn,vmax=vmaxx)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax=plt.subplot(1,2,2)     #
    plt.imshow(ytarget[0,:,:,0],cmap=plt.cm.jet,vmin=vminn,vmax=vmaxx)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.savefig('figurepred'+str(nrot)+str(latdim)+str(embedd)+str(usemsssim)+str(niter)+str(indexy)+str(rtall)+str(inputnum)+'.png', dpi=600, transparent=True)

    msee, msec= SSIMLossnum(ypred,ytarget)
    
    print('msee'+str(msee))
    filef = open('out.log', 'a')
    filef.write(str(msee))
    filef.write('\ n')
    filef.flush()
    filef.close()
    return msee #-x ** 2 - (y - 1) ** 2 + 1



def black_box_function(x, noise_level=0.01):
    """Function with unknown internals we wish to maximize.

    This is just serving as an example, for all intents and
    purposes think of the internals of this function, i.e.: the process
    which generates its output values, as unknown.
    """
    ytarget,xsizetarget,ysizetarget=getytarget(indexy,grot)
    fnn = getfunc()
    xall= x
    xall=[x]
    ypred = fnn.predict(xall)#.values())

#    msee=np.mean((ypred[0,:,:,0]-ytarget)**2)
    ytargeta= ypred*0.0
    ytargeta[0,:,:,0]=ytarget
    msee, msec= SSIMLossnum(ypred[0:1,:,:,0:1],ytargeta)
#    msee= tf_ms_ssim(tf.constant(ypred[0:1,:,:,0:1]/1.0,dtype=tf.float32),tf.constant(ytargeta/1.0,dtype=tf.float32)).numpy()
    return msee #-x ** 2 - (y - 1) ** 2 + 1

def _tf_fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g / tf.reduce_sum(g)


def tf_ssim(img1, img2, cs_map=False, mean_metric=True, size=11, sigma=1.5):
    window = _tf_fspecial_gauss(size, sigma) # window shape [size, size]
    K1 = 0.01
    K2 = 0.03
    L = 1  # depth of image (255 in case the image has a differnt scale)
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = tf.nn.conv2d(img1, window, strides=[1,1,1,1], padding='VALID')
    mu2 = tf.nn.conv2d(img2, window, strides=[1,1,1,1],padding='VALID')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = tf.nn.conv2d(img1*img1, window, strides=[1,1,1,1],padding='VALID') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2*img2, window, strides=[1,1,1,1],padding='VALID') - mu2_sq
    sigma12 = tf.nn.conv2d(img1*img2, window, strides=[1,1,1,1],padding='VALID') - mu1_mu2
    if cs_map:
        value = (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2)),
                (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
    else:
        value = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2))

    if mean_metric:
        value = tf.reduce_mean(value)
    return value

def tf_ms_ssim(img1, img2, mean_metric=True, level=5):
    weight = tf.constant([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=tf.float32)
    mssim = []
    mcs = []
    for l in range(level):
        ssim_map, cs_map = tf_ssim(img1, img2, cs_map=True, mean_metric=False)
        mssim.append(tf.reduce_mean(ssim_map))
        mcs.append(tf.reduce_mean(cs_map))
        a=[1,1,1,1]
        filtered_im1 = tf.nn.avg_pool(img1, a, a, padding='SAME')
        filtered_im2 = tf.nn.avg_pool(img2, a, a, padding='SAME')
        img1 = filtered_im1
        img2 = filtered_im2

    # list to tensor of dim D+1
    mssim = tf.stack(mssim, axis=0)
    mcs = tf.stack(mcs, axis=0)
    alpha= tf.constant(1)
    value = (tf.reduce_prod(mcs[0:level-1]**weight[0:level-1])*
                            (mssim[level-1]**weight[level-1]))#+ (tf.constant(1)-alpha)*tf.math.multiply(abs(img1-img2))

    if mean_metric:
        value = -tf.reduce_mean(value)
    return value

maxheight = .015#.0005
ncount=1
nrot = 541
latdim = 10
dshape=64*4
embedd=0
usemsssim = 0
embedd=0
niter=0
svae=0

## rotate target
indexy=21
grot=0

indexy=1123
grot=0

indexy=1124
grot=0

indexy=431
grot=0

rtall=1 ## only 0, 1, 2 (w init), 20 (w init, sum base)

## random initialization
ytarget,xsize,ysize=getytarget(indexy, grot)

scsize=sio.loadmat("screensize.mat")
scsize= scsize["screensize"]
maxheight = maxheight * scsize
if indexy==921:
    maxheight=maxheight/1.5
if indexy==43:
    maxheight=maxheight*.75
if indexy==11:
    maxheight=maxheight*.75
print('maxheight',str(maxheight))
## resort
ytarget,xsize,ysize=getytarget(indexy, grot)


numcalls=120 ## 60 
numcallsinit = 0 
totrain =1
pretrained=1 ## 1: run the bayesian optimization; 0: read the optimized data from excel sheet
basescale=0  ## scale factor applied to base figure, 0 or 1
numcalls=100 ## 60 
numcallsinit = 0 
totrain =1
inputnum=52# 101#52#102# 52#20#52#21 ## adjust shape generations
solid = 0

vminn=0
vmaxx=1
wsize=1
sio.savemat('wsize.mat',{'wsize':wsize})
if pretrained ==100:
    eii  = [0,1,2,3]
    print('yess')
else:
    eii = [0] 
sio.savemat('eii.mat',{'eii':eii})

if rtall==20:
    sumbase=1
else:
    sumbase=0

if inputnum==20:
    nrot = 5
    latdim = 6
    inpallorg=sio.loadmat("zmean"+str(nrot)+str(latdim)+"vae"+str(0)+".mat")
elif inputnum==52:
    nrot = 5
    latdim = 6
    inpallorg=sio.loadmat("zmean"+str(nrot)+str(latdim)+"vae"+str(inputnum)+".mat")
elif inputnum==101:
    nrot = 3
    latdim = 6
    inpallorg=sio.loadmat("zmean"+str(nrot)+str(latdim)+"vae"+str(inputnum)+".mat")
elif inputnum==102:
    nrot = 3
    latdim = 6
    inpallorg=sio.loadmat("zmean"+str(nrot)+str(latdim)+"vae"+str(inputnum)+".mat")

else:
    inpallorg=sio.loadmat("zmean"+str(nrot)+str(latdim)+"vae.mat")
inpallorg=inpallorg['zmean']
inpall = inpallorg+0
aall=[]
for ii in range(0,inpall.shape[0]):
    aall=aall+[inpall[ii,:]]
print(len(aall))
print(inpall.shape)

dataall2d=sio.loadmat('kirbishapevaecircquat5all'+str(nrot)+'.mat')    
dataall2d=dataall2d['dataall']
dataall=255-dataall2d[:,:,:,:]
dataall1=dataall/255+0
print(dataall.shape)
print(np.max(dataall[10,:,:,:]))
neigh = NearestNeighbors(n_neighbors=2,algorithm='kd_tree')
neigh.fit(aall)

if svae ==0:
    vae=getfuncvaeall(inputnum,nrot,latdim)
else:
    niter = 0 
    stringsavemod2decoder='modelkirgamivaedecoder'+str(nrot)+str(latdim)+str(embedd)+str(usemsssim)+str(niter)
    stringsavemod2encoder='modelkirgamivaeencoder'+str(nrot)+str(latdim)+str(embedd)+str(usemsssim)+str(niter)
    stringsavemod2predictor='modelkirgamivaepredictor'+str(nrot)+str(latdim)+str(embedd)+str(usemsssim)+str(niter)
    encoder = flib.getfuncsvae(stringsavemod2encoder)
    decoder = flib.getfuncsvae(stringsavemod2decoder)

ytarget,xsize,ysize=getytarget(indexy, grot)
ytarget2,xsize,ysize=getytarget(indexy, grot)

plt.figure(figsize=(15,15))

ax = plt.subplot(1,1,1)
cs=plt.imshow(ytarget, cmap=plt.cm.jet,vmin=0,vmax=1)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

plt.savefig('csample.png', dpi=600, transparent=True)
ytarget.resize((1,dshape,dshape,1))
ytarget2.resize((1,dshape,dshape,1))

msee, msec= SSIMLossnum(ytarget,ytarget2)
print(str(msee))
#sys.exit()
figureoutoldbase = np.zeros((64,64))
if pretrained>0:
    checkperturb=1
else:
    checkperturb = 0 

if pretrained==1 or checkperturb==1:
    ssxy='xy'+str(nrot)+str(latdim)+str(embedd)+str(usemsssim)+str(niter)+str(indexy)+str(rtall)+str(inputnum)#+'_init30'
    data=pd.read_csv(ssxy+'.csv')   
    data=data.values

    ss='err'+str(nrot)+str(latdim)+str(embedd)+str(usemsssim)+str(niter)+str(indexy)+str(rtall)+str(inputnum)#+'_init30'
    error=pd.read_csv(ss+'.csv') 
    errorall=error.values   
    
    x00 = data[:,0]
    y00 = data[:,1]
    print(str(len(x00)))
    for itotrain in range(totrain):
        numcallsinit = len(x00)
        x0=[]
        y0=[]
        innall= range(numcalls*itotrain,numcalls*(itotrain+1))
        for inn in innall:
            x0n=data[inn,0][1:-1]
            xkir=np.fromstring(x0n, dtype=float, sep=',')    
            y0n=data[inn,1]
            x0.append(xkir.tolist())
            y0=y0+[y0n]
        print(str(y0n))

        indexx=np.argsort(y0)

        iinn_inp=indexx[0]#77#62#indexx[0]#indexx[2]  ##-16
        x=data[iinn_inp,0][1:-1]
        xkir=np.fromstring(x, dtype=float, sep=',')    

        [zlatsold, zangsold]=inputvae(xkir,0,latdim) ## latent space, latent angle space
        print(xkir[0:latdim])
        print(neigh.kneighbors([xkir[0:latdim]]))
        '''
        find the nearst neighbor of the vae generated image
        '''
        neibout= neigh.kneighbors([xkir[0:latdim]])
        print(neibout[1][0][0])
        print(neibout[1][0][1])
        print(aall[9521])
        print(aall[9196])

        nsel1 = neibout[1][0][1]

        #xkir[0:latdim]=aall[nsel1]
        [zlatsold, zangsold]=inputvae(xkir,0,latdim) ## latent space, latent angle space

        plt.figure(figsize=(15,15))
        ax=plt.subplot(1,1,1)     #
        plt.imshow(dataall[nsel1,:,:,0],label = '2D')
        plt.rc('font',family='Times New Roman') 
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.savefig('test.png', dpi=600, transparent=True)

        figureoutoldbase0=vae.manifoldnd(d=1,ll=zlatsold,cmap='viridis')
        # print(figureoutold.shape)
        scalee = 1.2
        figureoutoldbase0 = normalize(figureoutoldbase0,scalee)
        figureoutoldbase = figureoutoldbase0 + figureoutoldbase
        figureoutoldbase = normalize(figureoutoldbase,scalee)

    plt.figure(figsize=(15,15))
    ax=plt.subplot(1,1,1)     #
    plt.imshow(figureoutoldbase,label = '2D')
    plt.rc('font',family='Times New Roman') 
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.savefig('figureoutoldbase'+str(nrot)+str(latdim)+str(embedd)+str(usemsssim)+str(niter)+str(indexy)+str(rtall)+str(inputnum)+'.png', dpi=600, transparent=True)

    #sys.exit()
    if checkperturb==1:
        for nplot in range(1):
            if pretrained ==100:
                if nplot ==0:
                    dtt = 10
                    dtt2 = 2        
                else:
                    dtt =20
                    dtt2= 12
            rt=rtall+0

            plt.figure(figsize=(15,10))
            ax=plt.subplot(1,1,1)     #
            if pretrained ==100:
                plt.imshow(figureoutoldbase,cmap=plt.cm.get_cmap("Reds"), interpolation='nearest', vmin=vminn, vmax=vmaxx)
            else:
                plt.imshow(1-figureoutoldbase,cmap=plt.cm.jet, interpolation='nearest', vmin=vminn, vmax=vmaxx)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            addmask(figureoutoldbase,plt.cm.jet,vminn,vmaxx)

            if solid ==1:
                plt.savefig('3dshape'+str(nrot)+str(latdim)+str(embedd)+str(usemsssim)+str(niter)+str(indexy)+str(rtall)+str(inputnum)+'_'+str(iinn_inp+1)+'.png', dpi=600, transparent=True)
            else:
                plt.savefig('shape'+str(nrot)+str(latdim)+str(embedd)+str(usemsssim)+str(niter)+str(indexy)+str(rtall)+str(inputnum)+'_'+str(iinn_inp+1)+'.png', dpi=600, transparent=True)

            #sys.exit()
            msee =black_box_functionrt(xkir, noise_level=0.01)  ## only works for the no image operation
            if pretrained ==100:
                ydes = [msee]
                xdes =[]
                xdes.append(xkir.tolist())
                print(str(xdes))
            vminn=0
            vmaxx=1           


            plt.figure(figsize=(15,10))
            ax=plt.subplot(1,3,1)     #
            if pretrained ==100:
                plt.imshow(figureoutoldbase,cmap=plt.cm.get_cmap("Reds"), interpolation='nearest', vmin=vminn, vmax=vmaxx)
            else:
                plt.imshow(1-figureoutoldbase,cmap=plt.cm.jet, interpolation='nearest', vmin=vminn, vmax=vmaxx)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            addmask(figureoutoldbase,plt.cm.jet,vminn,vmaxx)

            ax=plt.subplot(1,3,2)     #
            ytarget,xsize,ysize=getytarget(indexy, grot)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            print(np.max(ytarget))
            vminn=0
            vmaxx=1

            if pretrained ==100:
                plt.imshow(ytarget[30-dtt:30+dtt,34-dtt2:34+dtt2], cmap=plt.cm.jet,vmin=vminn,vmax=vmaxx)
            else:        
                plt.imshow(ytarget,cmap=plt.cm.jet,vmin=vminn,vmax=vmaxx)
            addmask(ytarget,plt.cm.jet,vminn,vmaxx)
        
            ypred,xsizepred,ysizepred = getypredict(1) ## 3d
        
            ax=plt.subplot(1,3,3)     #
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            print(np.max(ypred))
            if pretrained ==100:
                #plt.contourf(ypred[30-dtt:30+dtt,34-dtt2:34+dtt2],label = '2D')
                plt.imshow(ypred[30-dtt:30+dtt,34-dtt2:34+dtt2], cmap=plt.cm.jet,vmin=vminn,vmax=vmaxx)
            else:        
                #plt.imshow(ypred,label = 'Predicteds')
                plt.imshow(ypred, cmap=plt.cm.jet,vmin=vminn,vmax=vmaxx)
            addmask(ypred,plt.cm.jet,vminn,vmaxx)
            if solid ==1:
                plt.savefig('3dbayesoutputbase'+str(nrot)+str(latdim)+str(embedd)+str(usemsssim)+str(niter)+str(indexy)+str(rtall)+str(inputnum)+str(iinn_inp)+'.png', dpi=600, transparent=True)
            else:
                plt.savefig('bayesoutputbase'+str(nrot)+str(latdim)+str(embedd)+str(usemsssim)+str(niter)+str(indexy)+str(rtall)+str(inputnum)+str(iinn_inp)+'.png', dpi=600, transparent=True)
        err3d = np.abs(ytarget-ypred)
        err3dabs = err3d*0
        eii0=np.argwhere(err3d>np.percentile(err3d,95))
        for nii in range(0,len(eii0)):
            err3dabs[eii0[nii][0],eii0[nii][1]] = 1
        plt.figure(figsize=(15,10))
        ax=plt.subplot(1,3,1)     #
        #plt.imshow(err3d,label = 'Predicted ks')
        plt.imshow(err3d,cmap=plt.cm.jet,vmin=vminn,vmax=vmaxx)
        
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax=plt.subplot(1,3,2)     #
        #plt.imshow(err3dabs,label = 'Predicted ks')
        plt.imshow(err3dabs,cmap=plt.cm.jet,vmin=vminn,vmax=vmaxx)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.savefig('boundaryoutputbase'+str(nrot)+str(latdim)+str(embedd)+str(usemsssim)+str(niter)+str(indexy)+str(rtall)+str(inputnum)+'.png', dpi=600, transparent=True)
        #print('msec'+str(msec),'msee'+str(msee))
        print('mh'+str(maxheight))
        md = minkowski_distance(ytarget.flatten(),ypred.flatten(),3)
        print('mdd',str(md))

'''
Set the boundary/constraints on the variables for bayesian optimization
'''
pbounds = []
pboundsdes = []
dtt =[]
deldtt = 4
for ii in range(0,latdim):
    a=(np.min(inpall[:,ii]),np.max(inpall[:,ii]))
    pbounds= pbounds +[a]
    dtt = dtt+[(np.max(inpall[:,ii])-np.min(inpall[:,ii]))/deldtt]

if wsize==0:
    pbounds= pbounds+ [(0.5,2.5)]
    dtt = dtt+[(2.5-.5)/deldtt]

else:
    pbounds= pbounds+ [(.8,1.3)]+ [(0.5,3.0)]
    dtt = dtt+[(1.3-.8)/deldtt]+[(3.0-.5)/deldtt]

print(str(pbounds))
if pretrained ==100:
    for ii in range(0,latdim+wsize+1):
        pboundsdes = pboundsdes +[(xkir[ii]-dtt[ii],xkir[ii]+dtt[ii])]
    print(xkir)
    print(pboundsdes)


'''
Start the bayesian optimization
'''
from skopt import gp_minimize

## initialization step for res
if rtall==2 or rtall==20:
    rt= 2
    readstringd = 'C:\\Users\\leixi\\Downloads\\Codes\\Codes\\'
    if inputnum==3:
        inpall=sio.loadmat(readstringd+'kirigamiparam.mat')
        inpall=inpall['inpall']          
        for ii in range(0,len(inpall)):
            # if inpall[ii,0]==9 and inpall[ii,1]==40 and inpall[ii,2]==20:
            if inpall[ii,0]==4 and inpall[ii,1]==60 and inpall[ii,2]==80:
                iicor=ii
        
        
    elif inputnum==21:
        rows, cols = (5,latdim+1)
        x0 = [[0 for i in range(cols)] for j in range(rows)]
        y0=[]
        #yc0 = []
        print(x0)

        for ncc in range(0,rows):
            
            x0[ncc]=list(inpall[-1,:])+[.2/5*(ncc+1)]
            msee=black_box_functionrt(x0[ncc], noise_level=0.001)
            y0=y0+[msee]
            #yc0 = yc0 + [msec]
    df=pd.concat([pd.DataFrame({'xx':x0}),pd.DataFrame({'yy':y0})], axis=1)
    df.to_csv(ssxy+'.csv',index=False,sep=',')   
    print(y0)
    
    [zlatsold, zangsold]=inputvae(x0[0][0:-1],0,latdim) ## latent space, latent angle space
    figureoutoldbase=vae.manifoldnd(d=1,ll=zlatsold,cmap='viridis')
    scalee = 1.2
    figureoutoldbase = normalize(figureoutoldbase,scalee)
    
    plt.figure(figsize=(15,10))
    ax=plt.subplot(1,3,1)     #
    plt.imshow(figureoutoldbase,cmap=plt.cm.jet,vmin=vminn,vmax=vmaxx)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax=plt.subplot(1,3,2)     #
    ytarget,xsize,ysize=getytarget(indexy, grot)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    print(np.max(ytarget))
    plt.imshow(ytarget,cmap=plt.cm.jet,vmin=vminn,vmax=vmaxx)

    ypred,xsizepred,ysizepred = getypredict(1) ## 3d
    
    ax=plt.subplot(1,3,3)     
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    print(np.max(ypred))
    plt.imshow(ypred,cmap=plt.cm.jet,vmin=vminn,vmax=vmaxx)
    plt.savefig('bayesoutput0'+str(nrot)+str(latdim)+str(embedd)+str(usemsssim)+str(niter)+str(indexy)+str(rt)+str(inputnum)+'.png', dpi=600, transparent=True)



rt= rtall+0
if rtall==0:
    print('rtall'+str(0))
    res = gp_minimize(black_box_function,                  # the function to minimize
                      pbounds,      # the bounds on each dimension of x
                      x0=x0,
                      y0=y0,
                      acq_func="EI",      # the acquisition function
                      n_calls=numcalls,         # the number of evaluations of f
                      n_random_starts=10,  # the number of random initialization points#                      callback=[checkpoint_saver],
                      noise=0.1**2,       # the noise level (optional)
                      random_state=1234)   # the random seed
elif pretrained==100:
    print('rtall'+str(200))
    print ('ydes' + str(ydes))
    res = gp_minimize(black_box_functionrt,                  # the function to minimize
                      pboundsdes,      # the bounds on each dimension of x
                      x0=xdes,
                      y0=ydes,
                      acq_func="EI",      # the acquisition function
                      n_calls=numcalls,         # the number of evaluations of f
                      n_random_starts=0,  # the number of random initialization points
                      noise=0.01**2,       # the noise level (optional)
                      random_state=0)   # the random seed
elif rtall==1 and pretrained==0:
    print('rtall'+str(1))

    x0 = []
    y0 = []
    if len(x0)>0:
        res = gp_minimize(black_box_functionrt,                  # the function to minimize
                        pbounds,      # the bounds on each dimension of x
                        x0=x0,
                        y0=y0,
                        acq_func="EI",      # the acquisition function
                        n_calls=numcalls,         # the number of evaluations of f
                        n_random_starts=0,  # the number of random initialization points
                        noise=0.01**2,       # the noise level (optional)
                        random_state=0)   # the random seed       
    else:

        res = gp_minimize(black_box_functionrt,                  # the function to minimize
                        pbounds,      # the bounds on each dimension of x
                        acq_func="EI",      # the acquisition function
                        n_calls=numcalls,         # the number of evaluations of f
                        n_random_starts=10,  # the number of random initialization points
                        noise=0.01**2,       # the noise level (optional)
                        random_state=60)   # the random seed50 40  30 20  1234  

numcallsall = numcalls+numcallsinit
minerr=np.zeros((numcallsall,1))
for ii in range (0,numcallsall):
    minerr[ii]= np.min(res.func_vals[0:ii+1])
    
## save and plot error
plt.figure(figsize=(15,10))
plt.plot(minerr,'o-')

if pretrained <100:
    ss='err'+str(nrot)+str(latdim)+str(embedd)+str(usemsssim)+str(niter)+str(indexy)+str(rt)+str(inputnum)
    if solid==1:
        ss = '3d'+ss
    df=pd.concat([pd.DataFrame({'Measured': 1-minerr.flatten()})], axis=1)
    df.to_csv(ss+'.csv',index=False,sep=',')   

    ssxy='xy'+str(nrot)+str(latdim)+str(embedd)+str(usemsssim)+str(niter)+str(indexy)+str(rt)+str(inputnum)
    if solid==1:
        ssxy = '3d'+ssxy
    df=pd.concat([pd.DataFrame({'xx':res.x_iters}),pd.DataFrame({'yy':res.func_vals})], axis=1)
    df.to_csv(ssxy+'.csv',index=False,sep=',')   
else:
    ss='errloc'+str(nrot)+str(latdim)+str(embedd)+str(usemsssim)+str(niter)+str(indexy)+str(rt)+str(inputnum)
    df=pd.concat([pd.DataFrame({'Measured': 1-minerr.flatten()})], axis=1)
    df.to_csv(ss+'.csv',index=False,sep=',')   

    ssxy='xyloc'+str(nrot)+str(latdim)+str(embedd)+str(usemsssim)+str(niter)+str(indexy)+str(rt)+str(inputnum)
    df=pd.concat([pd.DataFrame({'xx':res.x_iters}),pd.DataFrame({'yy':res.func_vals})], axis=1)
    df.to_csv(ssxy+'.csv',index=False,sep=',')   

##
zopt = res.x

if wsize==0:    
    [zlatsold, zangsold]=inputvae(res.x[0:-1],0,latdim) ## latent space, latent angle space
else:
    [zlatsold, zangsold]=inputvae(res.x[0:-2],0,latdim) ## latent space, latent angle space


msee=black_box_functionrt(res.x, noise_level=0.01)
figureoutold=vae.manifoldnd(d=1,ll=zlatsold,cmap='viridis')
print(figureoutold.shape)
scalee = 1.2
figureoutoldn = normalize(figureoutold,scalee)
figureoutoldn = figureoutoldn + figureoutoldbase
figureoutoldn = normalize(figureoutoldn,scalee)


plt.figure(figsize=(15,10))
ax=plt.subplot(1,3,1)     #
if pretrained ==100:
    plt.imshow(figureoutoldn,cmap=plt.cm.get_cmap("Reds"), interpolation='nearest', vmin=vminn, vmax=vmaxx)
else:
    plt.imshow(1-figureoutoldn,cmap=plt.cm.jet, interpolation='nearest', vmin=vminn, vmax=vmaxx)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
addmask(figureoutoldn,plt.cm.jet,vminn,vmaxx)

ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax=plt.subplot(1,3,2)     #
ytarget,xsize,ysize=getytarget(indexy, grot)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
print(np.max(ytarget))
plt.imshow(ytarget,cmap=plt.cm.jet,vmin=vminn,vmax=vmaxx)

ypred,xsizepred,ysizepred = getypredict(1) ## 3d

ax=plt.subplot(1,3,3)     #
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
print(np.max(ypred))
plt.imshow(ypred,cmap=plt.cm.jet,vmin=vminn,vmax=vmaxx)
plt.savefig('bayesoutput'+str(nrot)+str(latdim)+str(embedd)+str(usemsssim)+str(niter)+str(indexy)+str(rt)+str(inputnum)+'.png', dpi=600, transparent=True)

