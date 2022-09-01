# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 08:32:32 2020

@author: hatam
"""

import cv2 
import math
import numpy as np 
from skimage import color
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
import argparse

def image_gradient(mat):
    img_gradient=((np.roll(mat,1,axis=0)-np.roll(mat,-1,axis=0))*(np.roll(mat,1,axis=0)-np.roll(mat,-1,axis=0)))
    +((np.roll(mat,1,axis=1)-np.roll(mat,-1,axis=1))*(np.roll(mat,1,axis=1)-np.roll(mat,-1,axis=1)))
    img_final=np.zeros((len(mat),len(mat[0])))
    img_final=np.sum(img_gradient[:,:,0:2], axis=2)
    return img_final


# Read the main image 
img= cv2.imread('slic.jpg')
img1=img.copy()

img=np.delete(img, np.s_[::2], 1)
img=np.delete(img, np.s_[::2], 0)
img=np.delete(img, np.s_[::2], 1)
img=np.delete(img, np.s_[::2], 0)

#define k
k =64
print("k=",k) 
lab_img = color.rgb2lab(img)
img_gra_lab= image_gradient(lab_img)
#find center
center=np.zeros([int(math.sqrt(k)),int(math.sqrt(k)),2])
for i in range(int(math.sqrt(k))):
    for j in range(int(math.sqrt(k))):
        center[i,j,:]=[int(i*len(img)/math.sqrt(k)+len(img)/(2*math.sqrt(k))),int(j*len(img[0])/math.sqrt(k)+len(img[0])/(2*math.sqrt(k)))]
        x=int(center[i,j,0])
        y=int(center[i,j,1])
        q,w=np.where(img_gra_lab[x-6:x+6,y-6:y+6]==(img_gra_lab[x-6:x+6,y-6:y+6].min()))
        center[i,j,:]=x+ q[0] ,y+ w[0]

df_cen=np.zeros((2,k))
df_cen[0,:]=center[:,:,0].reshape(1,-1)
df_cen[1,:]=center[:,:,1].reshape(1,-1)
df_cen = df_cen.astype(int)
'''
img[df_cen[0,:],df_cen[1,:],:]=(0,0,255)
cv2.imwrite('center.jpg',img)
'''
#make Lab from image
print("make property vector of image ...")

b=np.indices((len(img),len(img[0])))

property_img=np.zeros((len(img), len(img[0]), 5))
property_img[:,:,0:3]=lab_img
property_img[:, :, 3]=b[0,:,:]
property_img[:, :, 4]=b[1,:,:]
classi=np.zeros_like(img[:,:,0])
img_slic=np.zeros_like(img)
print("segmention ...\n \t calculate dlab")
lab_img = color.rgb2lab(img)
s=math.sqrt(img.size/(3*k))
n=0
iterion=10
m=10
alpha=m/s
while iterion>n:
    for i in range(0,k,1):
        d=(property_img-property_img[df_cen[0,i],df_cen[1,i],:])*(property_img-property_img[df_cen[0,i],df_cen[1,i],:])
        D=np.sqrt(np.sum(d[:,:,0:3], axis=2))+alpha*np.sqrt(np.sum(d[:,:,3:5], axis=2))
        x , y = np.where(np.logical_and(np.sqrt(np.sum(d[:,:,3:5], axis=2)) <2*s,D<10*(n+2)/2))
        classi[x,y]=i
                        
    for i in range(0,k,1):      
        x , y = np.where(classi==i)
        img[x,y,0]=np.mean(img[x,y,0])
        img[x,y,1]=np.mean(img[x,y,1])
        img[x,y,2]=np.mean(img[x,y,2])
        property_img[df_cen[0,i],df_cen[1,i],0]=np.mean(img[x,y,0])
        property_img[df_cen[0,i],df_cen[1,i],1]=np.mean(img[x,y,1])
        property_img[df_cen[0,i],df_cen[1,i],2]=np.mean(img[x,y,2])
        property_img[df_cen[0,i],df_cen[1,i],3]=np.mean(x)
        property_img[df_cen[0,i],df_cen[1,i],4]=np.mean(y)
        
    n=n+1
    print("ite:" ,n,"\n")
    filename = "images%d.jpg"%n
    cv2.imwrite(filename,img)   
    #filename = "images_%d.jpg"%n     
    
newSize=( len(img[0]),len(img))

img_slic[:,:,0]=cv2.resize(img[:,:,0], newSize,  interpolation = cv2.INTER_CUBIC)
img_slic[:,:,1]=cv2.resize(img[:,:,1], newSize,  interpolation = cv2.INTER_CUBIC)
img_slic[:,:,2]=cv2.resize(img[:,:,2], newSize,  interpolation = cv2.INTER_CUBIC)

cv2.imwrite("res07.jpg",img_slic)


