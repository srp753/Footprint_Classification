#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 15:09:31 2017

@author: snigdha
"""

# THIS IS A PROGRAM IMPLEMENTING DCT + PCA + SVM with rbf kernel
#in order to classify left and right footprints
    
import numpy as np   
from numpy import array   
from sklearn import svm                                                         
import cv2                                         
import glob
from natsort import natsorted
from sklearn.decomposition import PCA

#Obtaining train data for left foot
   
left_data = list()

files = glob.glob("/users/snigdha/Desktop/Desktop/10701/data/1/train/left/*.png")
#Opening the files in the numerical order 
sorted_files=natsorted(files)
for myFile in sorted_files:
    
    from PIL import Image
    #Converting an RGB image into a grey-scale image
    img = Image.open(myFile).convert('L')
    img = img.resize((100,300))
    image_arr = array(img)
    #Applying DCT to the image array
    image_arr= np.float32(image_arr)/255.0  # float conversion/scale
    dst = cv2.dct(image_arr)  #dct
    #Taking the chunk of 50 rows and 50 columns to obtain non-zero components
    new_arr=image_arr[0:50,0:50]
    image_arr1=image_arr.flatten()
    left_data.append(image_arr1)
        
lfoot_train=np.array(left_data)
print ('lfoot_data shape:',lfoot_train.shape)

#Obtaining train data for right foot and performing similar operations
#such as DCT as above

right_data = list()

files1 = glob.glob ("/users/snigdha/Desktop/Desktop/10701/data/1/train/right/*.png")
sorted_files1=natsorted(files1)
for myFile in sorted_files1:
    
    img1 = Image.open(myFile).convert('L')
    img1 = img1.resize((100,300))
    image_arr1 = array(img1)
    image_arr1= np.float32(image_arr1)/255.0  # float conversion/scale
    dst = cv2.dct(image_arr1)  #dct
    new_arr=image_arr1[0:50,0:50]
    image_arr1=image_arr1.flatten()
    right_data.append(image_arr1)
    
rfoot_train=np.array(right_data)
print ('rfoot_data shape:',rfoot_train.shape)
    
#Obtaining the test data and applying DCT to it
test_data = list()

files1 = glob.glob("/users/snigdha/Desktop/Desktop/10701/data/1/test/*.png")
sorted_files1=natsorted(files1)
for myFile in sorted_files1:
    
    img1 = Image.open(myFile).convert('L')
    img1 = img1.resize((100,300))
    image_arr1 = array(img1)
    image_arr1= np.float32(image_arr1)/255.0  # float conversion/scale
    dst = cv2.dct(image_arr1)  #dct
    #image_arr1 = np.uint8(dst)*255.0    # convert back
    new_arr=image_arr1[0:50,0:50]
    image_arr1=image_arr1.flatten()
    test_data.append(image_arr1)
  
X_test=np.array(test_data)
print ('test data shape:',X_test.shape)


#Concatenating the left and right foot train data
X_train=np.concatenate((lfoot_train, rfoot_train), axis=0)
y_train = np.array(list(np.ones(5000)*2) + list(np.ones(5000)))

#Applying PCA to get the most important 30 properties having
#high variance

pca=PCA(n_components=30)
X_train=pca.fit_transform(X_train)
X_test=pca.transform(X_test)

#Tuning the paramters of SVM with rbf kernel with GridSearchCV

from sklearn.model_selection import GridSearchCV 

Cs = [0.0001,0.001,0.01,0.1,1,10,100,1000]
gammas = [0.0001,0.001,0.01,1,10,100,1000]
param_grid = {'C': Cs, 'gamma' : gammas}
grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=3)
grid_search.fit(X_train, y_train)
print grid_search.best_params_

#Predicting left and right footprint using the best C and gamma value
predicted = grid_search.predict(X_test)
print predicted

np.savetxt("1.csv", predicted, delimiter=",")


