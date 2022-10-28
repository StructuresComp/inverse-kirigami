#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in 2022

@author: leixinma
"""
import numpy as np
import os

from keras.layers import Input, Dense
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D,Dropout,Flatten,BatchNormalization
from keras.models import Model

from PIL import Image
import sys
import tensorflow as tf
from matplotlib import pyplot
from tensorflow.keras import datasets, layers, models
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
import atomai as aoi
from skimage.metrics import structural_similarity as ssim
from numpy import linalg as LA
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torchvision.utils import save_image
import pandas as pd
import cv2
def addmask(a,cmap,vminn,vmaxx):
    a = a.astype('float')
    ia=np.argwhere(a==0)
    for ii in range(0,ia.shape[0]):
        a[ia[ii][0],ia[ii][1]]=float('nan')
    # ax.set_facecolor('w')
    # # plt.colorbar()
    # plt.show()
    masked_array = np.ma.array (a, mask=np.isnan(a))
    # cmap.set_bad('white')
    # ax.imshow(masked_array, interpolation='nearest', cmap=cmap, vmin=vminn, vmax=vmaxx)
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)
    return masked_array

def procrustes(X, Y, scaling=True, reflection='best'):

    n,m = X.shape
    ny,my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = np.linalg.norm(X0, 'fro')**2 #(X0**2.).sum()
    ssY = np.linalg.norm(Y0, 'fro')**2 #(Y0**2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U,s,Vt = np.linalg.svd(A,full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection is not 'best':

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:,-1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA**2

        # transformed coords
        Z = normX*traceTA*np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my,:]
    c = muX - b*np.dot(muY, T)

    tform = {'rotation':T, 'scale':b, 'translation':c}

    return d, Z, T

def rotaxis_general(imagedata,image_array, nrot ):
    for i in range(0,image_array.shape[0]):
        for j in range(0,image_array.shape[0]):
            if np.sqrt((i-31)**2+(j-31)**2)>=image_array.shape[0]/2-3:
                image_array[i,j]=255
    imagedata = Image.fromarray(image_array)
    datamod=255-image_array+0
            
        
    for nii in range(0,nrot-1):
        imagedatarot=imagedata.rotate(int(360/nrot*(nii+1)), fillcolor=(255))
        imagedatarot_array=np.array(imagedatarot)
        datamod=datamod+ 255-imagedatarot_array
    datamod = normalize(datamod,100)
    datamod = 255-datamod


    figure_size=4
    datamod3 = cv2.blur(datamod,(figure_size, figure_size))
    # plt.imshow(datamod3)
    datamod = datamod3
    return datamod

def mse(imageA, imageB):
    mse_error = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    mse_error /= float(imageA.shape[0] * imageA.shape[1])
    return mse_error

seed(1) 
def plot_history(trainloss,testloss):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(range(0,len(trainloss)),trainloss,
             label='Train Loss')
    plt.plot(range(0,len(testloss)),testloss,
             label = 'Test loss')
    plt.legend()
    plt.savefig('historyplot.png', dpi=600, transparent=True)    

def SSIMLoss(y_true, y_pred):
  return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0, filter_size=5))
def SSIMLoss_ms(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim_multiscale(
        y_true, y_pred,  filter_size=5,
        filter_sigma=1.5, k1=0.01, k2=0.03
    ))
def SSIMLossnum(y_true, y_pred):
  y_true0 = tf.constant(y_true/1.0,dtype=tf.float64)
  y_pred0 = tf.constant(y_pred/1.0,dtype=tf.float64)  
  return 1 - tf.reduce_mean(tf.image.ssim(y_true0, y_pred0, 1.0, filter_size=5)).numpy()

def SSIMLoss_msnum(y_true, y_pred):
  y_true0 = tf.constant(y_true/1.0,dtype=tf.float64)
  y_pred0 = tf.constant(y_pred/1.0,dtype=tf.float64)    
  return 1 - tf.reduce_mean(tf.image.ssim_multiscale(
        y_true0, y_pred0,  filter_size=5,
        filter_sigma=1.5, k1=0.01, k2=0.03
    ))
'''
2 dim vectors
'''
def inputvae(inpp,inpang,latdim):
    zlats=np.zeros((latdim,2))
    zangs=np.zeros((1,2))
    
    for ii in range(1,latdim+1):
        zlats[ii-1,:]=inpp[ii-1]
    zangs[0,:]=inpang
    return zlats, zangs

def normalize(figureout,maxx):
    index1=np.argwhere(figureout>maxx/2)
    index2=np.argwhere(figureout<=maxx/2)
    figureout[index1[:,0],index1[:,1]]=maxx
    figureout[index2[:,0],index2[:,1]]=0
    return figureout
class Dataset(torch.utils.data.Dataset):

    def __init__(self, X):
        
        self.X = X

        
    def __len__(self):
        
        return len(self.X)

    def __getitem__(self, index):
        
        x = self.X[index]

        return x

"""## Create a sampling layer"""

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


'''

data = np.zeros((h, w, 3), dtype=np.uint8)
data[0:256, 0:256] = [255, 0, 0] # red patch in upper left
img = Image.fromarray(data, 'RGB')
'''
readstring = 'C:\\Users\\leixi\\OneDrive\\Desktop\\sci03\\'
readstring0 = 'C:\\Users\\leixi\\Box\\allfigs_vertical\\'
readstringtr ='C:\\Users\\leixi\\Box\\codes\\'
checksample=0


dtt=0.0001
totrain=0## THIS NEED TO CHANGE TRAIN OR IMPORT
inp2d=101#101#102#52 #52# 0 for 2d, 1 for 3d, 2 for combined strip and circles, 3 for combined lotus

if inp2d==20:
    nrot=5#2#5
    latdim=6#10
    dataall2d=sio.loadmat('kirbishapevaevertall'+str(nrot)+'.mat')    
    dataall2d=dataall2d['dataall']
    dataall=255-dataall2d[:,:,:,:]
    
    print(dataall.shape)
    stringsinp="my2d_modelvaevertcombinednp"
    
elif inp2d==52:                
    nrot=5
    latdim= 6
    dataall2d=sio.loadmat('kirbishapevaecircquat5all'+str(nrot)+'.mat')    
    dataall2d=dataall2d['dataall']
    dataall=255-dataall2d[:,:,:,:]
    dataall1=dataall+0
    print(dataall.shape)

    stringsinp="my2d_modelvaecircquad5np"   

elif inp2d==101:                
    # nrot=17
    # latdim= 2
    nrot=3
    latdim= 6
    dataall2d=sio.loadmat('kirbishapevaeflcircquat11rotall'+str(nrot)+'.mat')    
    dataall2d=dataall2d['dataall']
    dataall=255-dataall2d[:,:,:,:]
    dataall1=dataall+0

    dataall=dataall1
    print(dataall.shape)

    stringsinp="my2d_modelvaeflcircquad11nrotnp" 
    plt.figure(figsize=(15,15))
    plt.imshow(dataall1[500,:,:,0])    

dataall_1=dataall
if checksample==1:
    sys.exit()

#dataall=np.concatenate((dataall_1,dataall_2), axis=0)
dataall=dataall_1
numall=dataall.shape[0]

ax=plt.subplot(1,1,1)     #将窗口分为两行两列四个子图，则可显示四幅图片
plt.figure(figsize=(15,15))
plt.imshow(dataall[-13,:,:,0])
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
dshape=dataall.shape[1]

## split test train for image data
if inp2d==60:
    trainrow=np.random.choice(range(0,numall),round(numall*.95),replace=False)
else:
    trainrow=np.random.choice(range(0,numall),round(numall*.7),replace=False)

##input
train_imagesinput=dataall[trainrow]
test_imagesinput= np.delete(dataall,trainrow, axis = 0)

##output
train_imagesoutput=dataall[trainrow]
test_imagesouput= np.delete(dataall,trainrow, axis = 0)


train_imagesinput, test_imagesinput = train_imagesinput / 255.0, test_imagesinput / 255.0
dataall_scaled=dataall/255.0


input_img = Input(shape=(dshape,dshape, 1))  
# sys.exit()
############
# Encoding #
############
strr='same'
regurat=0.0001
# latdim= 10#10 ## please input a value that is even number
if inp2d==2:
    nrot=541
input_dim = (dshape,dshape)

if totrain==1:    
    if inp2d ==60:
        vae = aoi.models.VAE(input_dim, latent_dim=latdim,
                                numlayers_encoder=2, numhidden_encoder=40,
                                numlayers_decoder=2, numhidden_decoder=40)
    else:
        vae = aoi.models.VAE(input_dim, latent_dim=latdim,
                                numlayers_encoder=2, numhidden_encoder=128,
                                numlayers_decoder=2, numhidden_decoder=128)
    # vae.fit(train_imagesinput[:,:,:,0],None,test_imagesinput[:,:,:,0],None,training_cycles=800,batchsize=64*10)
    optimizer = lambda p: torch.optim.Adam(p, lr=5e-4)  # specify your learning rate
    if  inp2d==60:
        optimizer = lambda p: torch.optim.Adam(p, lr=1e-3)  # specify your learning rate
        vae.fit(train_imagesinput[:,:,:,0],None,test_imagesinput[:,:,:,0],None, optimizer=optimizer,training_cycles= 2000,batchsize=round(dataall.shape[0]/10))

    else:
        
        if latdim ==2:
            vae.fit(train_imagesinput[:,:,:,0],None,test_imagesinput[:,:,:,0],None, optimizer=optimizer,training_cycles= 350,batchsize=round(dataall.shape[0]/10))
    
        else:
            vae.fit(train_imagesinput[:,:,:,0],None,test_imagesinput[:,:,:,0],None, optimizer=optimizer,training_cycles=300,batchsize=round(dataall.shape[0]/10))

else:
    vaeinp=aoi.models.load_model(stringsinp+str(nrot)+str(latdim)+"vae.tar")## input the whole model
    vae=vaeinp

trainloss_vaeneg=vae.loss_history.get('train_loss')
testloss_vaeneg=vae.loss_history.get('test_loss')
trainloss_vae=trainloss_vaeneg+[]
testloss_vae=testloss_vaeneg+[]
for num in range(len(testloss_vaeneg)):
    testloss_vae[num]=-testloss_vaeneg[num]
for num in range(len(trainloss_vaeneg)):
    trainloss_vae[num]=-trainloss_vaeneg[num]
#sys.exit()
plot_history(trainloss_vae,testloss_vae)

'''
check the quality of reconstruction --3 random samples
'''
#inpput=train_imagesinput[:,:,:,0]
inpput=dataall_scaled ## input is scaled now  0 to 1

'''
1d vector
'''


## Check the quality of reconstruction
if totrain==0 and inp2d==20:
    dataallax=sio.loadmat('kirbioutaxis1.mat')    
    dataallax=dataallax['dataall']
    dataallax=(255-dataallax[:,:,:,:])/255
    z_mean_vae, z_sd_vae= vae.encode(dataallax)
    [zlatsold, zangsold]=inputvae(z_mean_vae[0,0:],0,latdim) ## latent space, latent angle space
    figureoutold=vae.manifoldnd(d=1,ll=zlatsold,cmap='viridis')
    scalee = 1.2
    figureoutold = normalize(figureoutold,scalee)    
    plt.figure(figsize=(15,10))
    ax=plt.subplot(1,2,1)     #将窗口分为两行两列四个子图，则可显示四幅图片
    plt.imshow(dataallax[0,:,:,0],label = 'Validation loss')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
#    plt.imshow(figureoutrot)
    ax=plt.subplot(1,2,2)     #将窗口分为两行两列四个子图，则可显示四幅图片
    plt.imshow(figureoutold,label = 'Validation loss')
    plt.rc('font',family='Times New Roman') 
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.savefig('axisout.png', dpi=600, transparent=True)
z_mean_vae, z_sd_vae= vae.encode(inpput)


# sio.savemat("zmean"+str(nrot)+str(latdim)+"vae.mat",{'zmean':z_mean_vae})

sio.savemat("zmean"+str(nrot)+str(latdim)+"vae"+str(inp2d)+".mat",{'zmean':z_mean_vae})

# mmean = []
# mmeanold = []
# for ii in range(z_mean_vae.shape[0]):
#     zz=z_mean_vae[ii,:]
#     [zlatsold, zangsold]=flib.inputvae(zz,0,latdim) ## latent space, latent angle space
    
#     figureoutold=vae.manifoldnd(d=1,ll=zlatsold,cmap='viridis')
#     mmean=mmean+[np.mean(figureoutold[55:,55:])]
    
#     mmeanold = mmeanold + [np.mean(dataall[ii,55:,55:,0])]
# print('check if the image is flipped'+ str(mmean))
# from scipy.spatial.distance import cdist

### these lines of codes are for comparisons
if inp2d==52:
    ztarget=[2.2979518594281103, -2.2784827143791695, 3.4123991116433956, -3.027647987802186, 2.0995787238604517, 4.884079933166504]
elif inp2d==101:
    ztarget=[-1.5853050267449338, 3.220978556715221, 0.33974305625891876, 1.5983715566472423, -3.2919416791223393, -3.5500636100769043]
distt = np.zeros((z_mean_vae.shape[0],1))
for ii in range(0,z_mean_vae.shape[0]):
    temp =np.asarray(ztarget)-z_mean_vae[ii,0:]
    distt[ii,0] = np.linalg.norm(temp)
locc=np.argwhere(distt==np.min(distt))

comparevae=1
if inp2d==52 or inp2d==101:
    iiall = [locc[0][0]]
else:
    iiall=[180,412]


if inp2d==3:

    iiall=[0,620,iicor*nrot]
for ii in iiall:
    [zlatsold, zangsold]=inputvae(z_mean_vae[ii,0:],0,latdim) ## latent space, latent angle space

    if latdim==2: 
        figureoutold=vae.manifold2d(d=1,l1=[z_mean_vae[ii,0],z_mean_vae[ii,0]],l2=[z_mean_vae[ii,1],z_mean_vae[ii,1]],cmap='viridis')
    else:
        figureoutold=vae.manifoldnd(d=1,ll=zlatsold,cmap='viridis')
    if inp2d==0 or 2:
        scalee = 1.0
        figureoutold = normalize(figureoutold,scalee)
            
    if ii==0:
        figureoutold0=figureoutold
    else:
        figureoutold1=figureoutold
    quickcheck=0
    if quickcheck==1:
        filestring=readstring0+"vert2shape-vaetry_figout.png"
        filestring="vert2shape-vae_3_231_figout.png"
        filename = tf.constant(filestring)
        imagedata = Image.open(filestring).convert('L')
        
        imagedata = imagedata.resize((dshape,dshape),Image.ANTIALIAS) 
        imagedatarot=imagedata.rotate(90, fillcolor=(255)) 
        imagerot_array = np.array(imagedatarot)
        image_array = np.array(imagedata)
        image_array=(255-image_array)/255+(255-imagerot_array)/255
        index=np.argwhere(image_array>1)
        for ii in range(0,len(index)):
            image_array[index[ii][0],index[ii][1]]=1
        z_mean_vae, z_sd_vae= vae.encode(image_array)
        
        ii=0        
        [zlatsold, zangsold]=inputvae(z_mean_vae[ii,0:],0,latdim) ## latent space, latent angle space
        figureoutold=vae.manifoldnd(d=1,ll=zlatsold,cmap='viridis')
        inpput[ii,:,:,0]=image_array
        
        
    plt.figure(figsize=(15,10))
    ax=plt.subplot(1,2,1)     #将窗口分为两行两列四个子图，则可显示四幅图片
    plt.imshow(inpput[ii,:,:,0],label = 'Validation loss')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
#    plt.imshow(figureoutrot)
    ax=plt.subplot(1,2,2)     #将窗口分为两行两列四个子图，则可显示四幅图片
    plt.imshow(figureoutold,label = 'Validation loss')
    plt.rc('font',family='Times New Roman') 
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.savefig('sample.png', dpi=600, transparent=True)
    ax=plt.subplot(1,1,1)     #将窗口分为两行两列四个子图，则可显示四幅图片
    plt.imshow(figureoutold,label = 'Validation loss')
    plt.rc('font',family='Times New Roman') 
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.savefig('sample.png', dpi=600, transparent=True)
## save image sample

    # for ii in range(500,510):
    #     plt.figure(figsize=(15,10))
    #     ax=plt.subplot(1,2,1)     #将窗口分为两行两列四个子图，则可显示四幅图片
    #     plt.imshow(inpput[ii,:,:,0],label = 'Validation loss')
    #     ax.get_xaxis().set_visible(False)
    #     ax.get_yaxis().set_visible(False)
# plt.figure(figsize=(15,15))
# ax=plt.subplot(1,1,1)     #将窗口分为两行两列四个子图，则可显示四幅图片
# plt.imshow(figureoutold,label = 'Validation loss')
# plt.rc('font',family='Times New Roman') 
# ax.get_xaxis().set_visible(False)
# ax.get_yaxis().set_visible(False)
# plt.savefig('sample.png', dpi=600, transparent=True)

#if inout_inp>0:
#    dataall=normalize(dataall,255)
'''
check the variation of z1 vs z2
'''

fig,ax2 = plt.subplots(1, 1, figsize =(6, 6))
im2 = ax2.scatter(z_mean_vae[:,1], z_mean_vae[:,0], s=1, cmap='jet')
ax2.set_xlabel("$z_1$", fontsize=14)
ax2.set_ylabel("$z_2$", fontsize=14)
cbar2 = fig.colorbar(im2, ax=ax2, shrink=.8)
cbar2.set_label("Labels", fontsize=14)
cbar2.ax.tick_params(labelsize=10)

'''
SAVE TRAINED MODELS
'''

vae.save_model(stringsinp+str(nrot)+str(latdim)+'vae')
vaeinp=aoi.models.load_model(stringsinp+str(nrot)+str(latdim)+"vae.tar")## input the whole model

mserot=[]
mseold=[]
ssimrot=[]
ssimold=[]
    
#for ii in range(0,len(z_mean_rvae)):
for ii in range(1,len(dataall_scaled)):
    [zlatsold, zangsold]=inputvae(z_mean_vae[ii,0:],0,latdim) ## latent space, latent angle space
    figureoutold=vae.manifoldnd(d=1,ll=zlatsold,cmap='viridis')
    if inp2d==0 or 2:
        scalee = 1.0
        figureoutold = normalize(figureoutold,scalee)
        inpputsca = normalize(inpput[ii,:,:,0],scalee)
    else:
        inpputsca=inpput[ii,:,:,0]
    mseold=mseold+[mse(inpputsca,figureoutold)]
    ssimold=ssimold+[ssim(inpputsca,figureoutold)]
ssimold= np.mean(np.array(ssimold))

print([np.mean(mserot),np.mean(mseold),np.mean(ssimrot),np.mean(ssimold)])
sys.exit()

df=pd.concat([pd.DataFrame({'trloss':trainloss_vae}),  pd.DataFrame({'ttloss':testloss_vae}),pd.DataFrame({np.mean(ssimold)}),pd.DataFrame({np.mean(mseold)})], axis=1)
if inp2d==0:
    df.to_csv('error'+str(nrot)+str(latdim)+str(inp2d)+'.csv',index=False,sep=',')        
elif inp2d==1:
    df.to_csv('error3d'+str(nrot)+str(latdim)+str(inp2d)+'.csv',index=False,sep=',')        
elif inp2d==2:
    df.to_csv('errorall'+str(nrot)+str(latdim)+str(inp2d)+'.csv',index=False,sep=',')        
else:
    df.to_csv('errorall'+str(nrot)+str(latdim)+str(inp2d)+'.csv',index=False,sep=',')        

# z_mean_vae, z_sd_vae= vae.encode(test_imagesinput[:,:,:,0])
# ssi=[]
# for nii in range(0,test_imagesinput[:,:,:,0].shape[0]):
#     [zlatsold, zangsold]=inputvae(z_mean_vae[nii,0:],0,latdim) ## latent space, latent angle space
#     figureoutold=vae.manifoldnd(d=1,ll=zlatsold,cmap='viridis')
#     if inp2d==0 or 2:
#         scalee = 1.2
#         figureoutold = normalize(figureoutold,scalee)
#         figureoutold.resize(1,64,64,1)
#     ssi=ssi+[SSIMLossnum(figureoutold,test_imagesinput[:,:,:,:])]
'''
Perturb to generate new designs. This step is independent of vae / rvae
'''

'''
Interpoalte to generate new designs. This step is independent of vae / rvae
'''
import random
perturb=0
# sel=[8000,86200*1]
sel=[12000,12001]
sss= 5
sel= [12000+sss,12000+4+sss]
if inp2d==50:
    sel=[120,801]

rotatee=0
if rotatee==1:
    z_samp=z_mean_rvae[sel,1:]
else:
    z_samp=z_mean_vae[sel,0:]
scalee =1
y_dim=0
latent_dim=latdim
npairs=2
Ninterp=10  ## No of figures include the boundaries
znew =np.zeros((Ninterp*npairs, latent_dim+y_dim))
j=0
for npair in range (0, len(z_samp), 2):
   # print (npair, npair+1)
    z1= z_samp[npair]
    if perturb==0:
        z2= z_samp[npair+1]
    else:
        z2= z_samp[npair]+random.random()*.7+.1
        
    z1n = LA.norm(z1)
    z2n = LA.norm(z2)
    dot = np.sum(z1*z2)/z1n/z2n
    omega=np.arccos(dot)
    #print(omega/2/np.pi*360)
    ## BOUNDARY FIGURES
    zslerp=z1
    [zlats, zangs]=inputvae(zslerp,0,latdim) ## latent space, latent angle space
    figureoutold0=vae.manifoldnd(d=1,ll=zlats,cmap='viridis')
    figureoutold0=normalize(figureoutold0,scalee)
    
    zslerp=z2
    [zlats, zangs]=inputvae(zslerp,0,latdim) ## latent space, latent angle space
    figureoutold1=vae.manifoldnd(d=1,ll=zlats,cmap='viridis')

    figureoutold1=normalize(figureoutold1,scalee)
    plt.figure(figsize=(15,10))

    for i in range(0, Ninterp): 
        f = (i)/(Ninterp-1)

        zslerp=(np.sin((1-f)*omega)/np.sin(omega)*z1+np.sin(f*omega)/np.sin(omega)*z2) ## zslerp has the same dimension as z1, z2
        znew[j, :] = zslerp
        
        
        [zlats, zangs]=inputvae(zslerp,0,latdim) ## latent space, latent angle space
        
        if latdim==2: 
            figureoutold=vae.manifold2d(d=1,l1=[z_mean_vae[ii,0],z_mean_vae[ii,0]],l2=[z_mean_vae[ii,1],z_mean_vae[ii,1]],cmap='viridis')
        else:
            figureoutold=vae.manifoldnd(d=1,ll=zlats,cmap='viridis')
#        figureout=normalize(figureout ,scalee)
        figureoutold=normalize(figureoutold,scalee)
        if i==0:
            ax=plt.subplot(1,Ninterp,1)     #将窗口分为两行两列四个子图，则可显示四幅图片
            plt.imshow(figureoutold,label = 'Validation loss')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        #    plt.imshow(figureoutrot)
        elif i== Ninterp-1:
            ax=plt.subplot(1,Ninterp,Ninterp)     #将窗口分为两行两列四个子图，则可显示四幅图片
            plt.imshow(figureoutold,label = 'Validation loss')
            plt.rc('font',family='Times New Roman') 
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        else:
            ax=plt.subplot(1,Ninterp,i+1)     #将窗口分为两行两列四个子图，则可显示四幅图片
            plt.imshow(figureoutold,label = 'Validation loss')
            plt.rc('font',family='Times New Roman') 
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        #    plt.legend('Validation loss')
        j= j+1
    if inp2d==0:
        plt.savefig('ri2drecon'+str(nrot)+str(latdim)+'.png', dpi=600, transparent=True)
    elif inp2d==1:
        plt.savefig('ri3drecon'+str(nrot)+str(latdim)+'.png', dpi=600, transparent=True)
    elif inp2d==2:
        plt.savefig('riallrecon'+str(nrot)+str(latdim)+'.png', dpi=600, transparent=True)
    elif inp2d==3:
        plt.savefig('rilotusrecon'+str(nrot)+str(latdim)+'.png', dpi=600, transparent=True)
    plt.show() 



sys.exit()
from sklearn.cluster import KMeans

Sum_of_squared_distances = []
K = range(1,50)
for k in K:
    km = KMeans(n_clusters=k, random_state=0).fit(z_mean_vae)
    Sum_of_squared_distances.append(km.inertia_)
fig = plt.figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
plt.plot(K, Sum_of_squared_distances, 'bx-')

plt.ylabel('Sum of squared distances', fontsize = 20)
plt.xlabel('k', fontsize = 20)
# plt.xlim([0, 300])
# plt.ylim([0, 2.5])
plt.xticks(fontsize=20)  
plt.yticks(fontsize=20)
plt.grid()
plt.savefig('cluster'+str(nrot)+str(latdim)+str(inp2d)+'.png', dpi=600, transparent=True)

#if inout_inp>0:
if latdim>=2:
    nc=5
    kmeans = KMeans(n_clusters=nc, random_state=0).fit(z_mean_vae)
    kmeans.labels_
    kc=kmeans.cluster_centers_
    y_kmeans=kmeans.predict(z_mean_vae)
    fig = plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(111)

    plt.scatter(z_mean_vae[:,1], z_mean_vae[:,2], c=y_kmeans, s=50, cmap='viridis')

#    plt.scatter(kc[:, 0],kc[:, 1], c='black', s=200, alpha=0.5);
    vminn=0
    vmaxx=1
    figure_size=1
    for jj in range(0,1):
        iiall=range(5*jj,5*(jj+1))
        plt.figure(figsize=(15,10))

        for ii in iiall:
    #        figureout=rvae.manifold2d(d=1,l1=[kc[ii,0],kc[ii,0]],l2=[kc[ii,1],kc[ii,1]],cmap='viridis')

            [zlats, zangs]=inputvae(kc[ii],0,latdim) ## latent space, latent angle space
            figureoutold0=vae.manifoldnd(d=1,ll=zlats,cmap='viridis')
            datamod3 = cv2.blur(figureoutold0,(figure_size, figure_size))
            # plt.imshow(datamod3)
            figureoutold0 = datamod3
            figureoutold0=normalize(figureoutold0,1)

            nrow=1
            ax=plt.subplot(nrow,int(len(iiall)/nrow),ii+1-iiall[0])     #将窗口分为两行两列四个子图，则可显示四幅图片
            # plt.imshow(figureoutold0)
            maskk=addmask(figureoutold0,plt.cm.jet,0,1)
            plt.imshow(maskk,cmap=plt.cm.jet, interpolation='nearest', vmin=vminn, vmax=vmaxx)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.rc('font',family='Times New Roman') 
            plt.savefig('clustercenter'+str(nrot)+str(latdim)+str(inp2d)+'.png', dpi=600, transparent=True)
    '''
    check the quality of ssim in detail
    '''


# if inout_inp>0:©
#     dataall=normalize(dataall,255)
if latdim==2:

    mserot=[]
    mseold=[]
    ssimrot=[]
    ssimold=[]
        
    #for ii in range(0,len(z_mean_rvae)):
    for ii in range(44,45):
        figureout=rvae.manifold2d(d=1,l1=[z_mean_rvae[ii,1],z_mean_rvae[ii,1]],l2=[z_mean_rvae[ii,2],z_mean_rvae[ii,2]],cmap='viridis')
        figureoutrot=rvae.manifold2d(d=1,l1=[z_mean_rvae[ii,1],z_mean_rvae[ii,1]],l2=[z_mean_rvae[ii,2],z_mean_rvae[ii,2]],theta=[z_mean_rvae[ii,0],z_mean_rvae[ii,0]],cmap='viridis')
        figureoutold=vae.manifold2d(d=1,l1=[z_mean_vae[ii,0],z_mean_vae[ii,0]],l2=[z_mean_vae[ii,1],z_mean_vae[ii,1]],cmap='viridis')
        mserot=mserot+[mse(inpput[ii,:,:,0],figureoutrot)]
        mseold=mseold+[mse(inpput[ii,:,:,0],figureoutold)]
        ssimrot=ssimrot+[ssim(inpput[ii,:,:,0],figureoutrot)]
        ssimold=ssimold+[ssim(inpput[ii,:,:,0],figureoutold)]
    
    print([np.mean(mserot),np.mean(mseold),np.mean(ssimrot),np.mean(ssimold)])
    plt.figure(figsize=(15,10))
    ax=plt.subplot(1,3,1)     #将窗口分为两行两列四个子图，则可显示四幅图片
    plt.imshow(255*inpput[ii,:,:,0],label = 'Validation loss')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
#plt.imshow(255*(figureoutrot-inpput[ii,:,:,0]))

mmm=[]
for i in range(0, inpput.shape[0]):
    
    mmm= mmm+[np.mean(inpput[i,:,:,:])]


mmm = np.asarray(mmm)
indexx=np.argwhere(mmm==np.min(mmm))

plt.figure(figsize=(15,10))
for ii in range(1,4):
    ax=plt.subplot(1,3,ii)     #将窗口分为两行两列四个子图，则可显示四幅图片
    plt.imshow(255*inpput[indexx[ii-1][0],:,:,0],label = 'Validation loss')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

filestring="peanutsad0.0241.2_figout"
image_array1= sio.loadmat(filestring+'.mat')
image_array1= image_array1['z1mod']
image_array1[np.isnan(image_array1)] = 0
filestring="dome0.0241.2_figout"
image_array2= sio.loadmat(filestring+'.mat')
image_array2= image_array2['z1mod']
image_array2[np.isnan(image_array2)] = 0


ypredict= image_array1.flatten()
ytarget= image_array2.flatten()
intersection = sum(ypredict * ytarget)
dcoeff=(2. * intersection ) / (sum(ypredict *ypredict) + sum(ytarget * ytarget))
print(str(dcoeff))


filestring="peanut"
image_array1= sio.loadmat(filestring+'.mat')
image_array1= image_array1['z1mod']
image_array1[np.isnan(image_array1)] = 0

filestring="dome"
image_array2= sio.loadmat(filestring+'.mat')
image_array2= image_array2['z1mod']
image_array2[np.isnan(image_array2)] = 0

filestring="notoptimal"
image_array3= sio.loadmat(filestring+'.mat')
image_array3= image_array3['z1mod']
image_array3[np.isnan(image_array3)] = 0

blur = image_array1
org = image_array1
print("MSE: ", mse(blur,org))
# print("RMSE: ", rmse(blur, org))
# print("PSNR: ", psnr(blur, org))
# print("SSIM: ", ssim(blur, org))
print("UQI: ", uqi(blur, org))
# print("MSSSIM: ", msssim(blur, org))
print("ERGAS: ", ergas(blur, org))
print("SCC: ", scc(blur, org))
print("RASE: ", rase(blur, org))
print("SAM: ", sam(blur, org))
print("VIF: ", vifp(blur, org))