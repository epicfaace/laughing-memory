# coding: utf-8

# ## U-net First Implementation

# The architecture used is the so-called U-Net, which is very common for image segmentation problems such as this. It works well even with small datasets (which is weird for a NN!).

# ### I. Initial Setup 
# 
# #### a) Libraries

# In[1]:

import os
import sys
import random
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm
import itertools #chain function 

import skimage.io
import skimage.transform
import skimage.morphology
#.io : imread, imshow, imread_collection, concatenate_images
#.transform: resize
#.morphology: label

import keras.models
import keras.layers
import keras.layers.core
import keras.layers.convolutional
import keras.layers.pooling
import keras.layers.merge
import keras.callbacks
#.models: Model, load_model
#.layers: Input
#   .core: Dropout, Lambda 
#   .convolutional: Conv2D, Conv2DTranspose
#   .pooling: MaxPooling2D
#   .merge: concatenate
#.callbacks: EarlyStopping, ModelCheckpoint

from keras import backend as K
import tensorflow as tf


# In[2]:

# Set some parameters
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
TRAIN_PATH = './Data/stage1_train/'
TEST_PATH = './Data/stage1_test/'


# #### b) Warnings filter
# 
# The warnings filter controls whether warnings are ignored, displayed, or turned into errors (raising an exception).
# The warnings filter maintains an ordered list of filter specifications. Each entry is a tuple of the form ```(action, message, category, module, lineno).```
# 
# For more documentation: [Warning Control](https://docs.python.org/2/library/warnings.html#warning-filter)
# 

# In[3]:

#Parameters: Action, 
warnings.filterwarnings('ignore', category=UserWarning, module='skimage')


# In[4]:

seed = 42
random.seed = seed
np.random.seed = seed


# #### c) Id Retrieval from file explorer
# 
# ```os.walk()``` walks your directory tree and returns the path, a list of directories, and a list of file (see [doc](https://www.saltycrane.com/blog/2007/03/python-oswalk-example/)). It is an iterator.

# In[5]:

#os.walk example
i = 0
for (path, dirs, files) in os.walk(TRAIN_PATH):
    print(dirs,files)
    i+=1
    if i == 4:
        break


# In[6]:

# Get train and test IDs (Next() returns the next item from the iterator)
train_ids = next(os.walk(TRAIN_PATH))[1] #We are interested in list at index 1: list with directories at path
test_ids = next(os.walk(TEST_PATH))[1]


# In[7]:

train_ids


# ### II. Data Loading

# #### a) Image and Mask storage/visualization
# 
# In this subsection, we will import all the images and associated masks. This will allow us to understand our data in terms of dimensionality and appearance.

# In[15]:

print("Size of the training set: " + str(len(train_ids)))


# In[16]:

print("Size of the test set: " + str(len(test_ids)))


# In[9]:

imgs_original = [] #List with all np.arrays that store pixel information
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = TRAIN_PATH + id_
    img = skimage.io.imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS] #Gets a np.array from an image
    imgs_original.append(img)
    skimage.io.imshow(img)
    plt.show()
    print(img.shape)


# In[19]:

#Sanity Check
print(len(imgs_original))
#print(imgs_original[0])


# In[ ]:




# #### b) Downsampling of images
# 
# We will downsample both the training and test images to keep things light and manageable, but we need to keep a record of the original sizes of the test images to upsample our predicted masks and create correct run-length encodings later on.

# In[ ]:

# Get and resize train images and masks(Initialize containers for storing downsampled images)

X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8) #dtype recommended for pixels
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool) #dtype recommended for boolean matric

#Note that the output of the NN in image segmentation (two classes) is a boolean matrix 
#with the dimensions of the image

print('Getting and resizing train images and masks ... ')

sys.stdout.flush() #forces it to "flush" the buffer

#Tip: Use tqdm() on top of the iterator to have a progress bar
#enumerate() returns a tuple (index, item) of the iterator. Define 2 looping variables if you use enumerate() 

for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = TRAIN_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_train[n] = img
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    for mask_file in next(os.walk(path + '/masks/'))[2]:
        mask_ = imread(path + '/masks/' + mask_file)
        mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', 
                                      preserve_range=True), axis=-1)
        mask = np.maximum(mask, mask_)
    Y_train[n] = mask

# Get and resize test images
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
sizes_test = []
print('Getting and resizing test images ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = TEST_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_test[n] = img

print('Done!')


# In[ ]:

X_train.shape


# In[ ]:

train_ids[0]


# In[ ]:

path = TRAIN_PATH + train_ids[0]
img = imread(path + '/images/' + train_ids[0] + '.png')[:,:,:IMG_CHANNELS]


# In[ ]:

img


# In[ ]: