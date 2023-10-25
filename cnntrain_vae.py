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

ax=plt.subplot(1,1,1)    
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
    ax=plt.subplot(1,2,1)    
    plt.imshow(dataallax[0,:,:,0],label = 'Validation loss')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
#    plt.imshow(figureoutrot)
    ax=plt.subplot(1,2,2)   
    plt.imshow(figureoutold,label = 'Validation loss')
    plt.rc('font',family='Times New Roman') 
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.savefig('axisout.png', dpi=600, transparent=True)
z_mean_vae, z_sd_vae= vae.encode(inpput)


# sio.savemat("zmean"+str(nrot)+str(latdim)+"vae.mat",{'zmean':z_mean_vae})

sio.savemat("zmean"+str(nrot)+str(latdim)+"vae"+str(inp2d)+".mat",{'zmean':z_mean_vae})
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
    ax=plt.subplot(1,2,1)  
    plt.imshow(inpput[ii,:,:,0],label = 'Validation loss')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
#    plt.imshow(figureoutrot)
    ax=plt.subplot(1,2,2)     
    plt.imshow(figureoutold,label = 'Validation loss')
    plt.rc('font',family='Times New Roman') 
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.savefig('sample.png', dpi=600, transparent=True)
    ax=plt.subplot(1,1,1)     
    plt.imshow(figureoutold,label = 'Validation loss')
    plt.rc('font',family='Times New Roman') 
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.savefig('sample.png', dpi=600, transparent=True)
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
            ax=plt.subplot(1,Ninterp,1)    
            plt.imshow(figureoutold,label = 'Validation loss')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        #    plt.imshow(figureoutrot)
        elif i== Ninterp-1:
            ax=plt.subplot(1,Ninterp,Ninterp)     
            plt.imshow(figureoutold,label = 'Validation loss')
            plt.rc('font',family='Times New Roman') 
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        else:
            ax=plt.subplot(1,Ninterp,i+1)    
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
