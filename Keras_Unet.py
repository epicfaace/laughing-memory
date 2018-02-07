
# coding: utf-8

# ## U-net First Implementation

# The architecture used is the so-called U-Net, which is very common for image segmentation problems such as this. It works well even with small datasets (which is weird for a NN!).

# ### I. Initial Setup 
# 
# #### a) Libraries

# In[41]:

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


# Keras has three backend implementations available: the TensorFlow backend, the Theano backend, and the CNTK backend.

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

train_ids, test_ids


# ### II. Data Loading

# #### a) Image and Mask storage/visualization
# 
# In this subsection, we will import all the images and associated masks. This will allow us to understand our data in terms of dimensionality and appearance.

# In[8]:

print("Size of the training set: " + str(len(train_ids)))


# In[9]:

print("Size of the test set: " + str(len(test_ids)))


# Visualization of the training dataset + Dimensionality of the images:

# In[10]:

imgs_train_original = [] #List with all np.arrays that store pixel information
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = TRAIN_PATH + id_
    img = skimage.io.imread(path + '/images/' + id_ + '.png')[:,:,:] #Gets a np.array from an image
    imgs_train_original.append(img)
    skimage.io.imshow(img[:,:,:IMG_CHANNELS])
    plt.show()
    print(img[:,:,:IMG_CHANNELS].shape)


# **Observation**: Training images have different dimensions (#pixels!)

# In[11]:

#Sanity Check
print(len(imgs_train_original))
#print(imgs_original[0])


# In[12]:

imgs_train_original[0].shape


# What is the fourth dimension? Let's dive deeper into the data:

# In[13]:

imgs_train_original[1][2]


# **Observation**: The fourth data number associated to each pixel is 255, always! -->No worries

# Visualization of the test dataset + Dimensionality of the images:

# In[14]:

imgs_test_original = [] #List with all np.arrays that store pixel information
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = TEST_PATH + id_
    img = skimage.io.imread(path + '/images/' + id_ + '.png')[:,:,:] #Gets a np.array from an image
    imgs_test_original.append(img)
    skimage.io.imshow(img[:,:,:IMG_CHANNELS])
    plt.show()
    print(img[:,:,:IMG_CHANNELS].shape)


# In[15]:

#Sanity Check
print(len(imgs_test_original))


# #### b) Downsampling of images
# 
# We will downsample both the training and test images to keep things light and manageable, but we need to keep a record of the original sizes of the test images to upsample our predicted masks and create correct run-length encodings later on. We will use the function resize from skimage (see [doc](http://scikit-image.org/docs/dev/auto_examples/transform/plot_rescale.html)).

# In[16]:

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
    img = skimage.io.imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    #Image resizing to lower resolution
    img = skimage.transform.resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    #X_train is a tensor of order 3: A "cube" of data <-> n matrices stacked together
    X_train[n] = img
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool) #Initialize container of the mask
    for mask_file in next(os.walk(path + '/masks/'))[2]:           #Use index 2 for getting name of files (.png)
        mask_ = skimage.io.imread(path + '/masks/' + mask_file)
        #print(mask_.shape) Remove the comment to see how that mask_ is an array of two dimensions! Not three!
        #Insert a new axis
        mask_ = np.expand_dims(skimage.transform.resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', 
                                      preserve_range=True), axis=-1) #preserve_range=True is important!
        #axis = -1 adds an axis at the end of the tuple of dimensions of a np.array
        mask = np.maximum(mask, mask_)
        #mask gets updated at each loop and includes all the masks!
    Y_train[n] = mask #Stores all the masks (true labels in a tensor of order 4: 1 tensor of order 3 per mask)

print('Training images succesfully downsampled!')


# Useful documentation: [np.expand_dims()](https://www.tutorialspoint.com/numpy/numpy_expand_dims.htm), [skimage.transform.resize()](http://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.resize)

# In[17]:

# Get and resize test images
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
sizes_test = []
print('Getting and resizing test images ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = TEST_PATH + id_
    img = skimage.io.imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    sizes_test.append([img.shape[0], img.shape[1]])
    img = skimage.transform.resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_test[n] = img
    
print('Test images successfully downsampled!')


# Let's see if things look all right by drawing some random images and their associated masks.

# In[20]:

# Check if training data looks all right (downsampled data)
ix = random.randint(0, len(train_ids)) #Generate an indez randomly
skimage.io.imshow(X_train[ix])
plt.show() #Remember: need imshow in conjunction with plt.show() to see images!
skimage.io.imshow(np.squeeze(Y_train[ix]))
plt.show()


# ### III. Definition of the evaluation metric for the competition

# In this section, we try to define the mean average precision at different intersection over union (IoU) thresholds metric in Keras. TensorFlow has a mean IoU metric, but it doesn't have any native support for the mean over multiple thresholds, so we need to implement this by ourselves. However, in a first stage, we will define the [Dice Coefficient](https://stats.stackexchange.com/questions/195006/is-the-dice-coefficient-the-same-as-accuracy/253992). **Needs review**
# 
# The Dice score is not only a measure of how many positives you find, but it also penalizes for the false positives that the method finds, similar to precision. So it is more similar to precision than accuracy. The only difference is the denominator, where you have the total number of positives instead of only the positives that the method finds. So the Dice score is also penalizing for the positives that your algorithm/method could not find.

# Documentation for [Keras](https://keras.io/backend/).

# In[22]:

# Define IoU metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


# ### IV. Definition of and training of the Neural Network (U-net)

# Some observations on the functions used:
# 
# -->```keras.layers.Input()```:
# 
# A Keras tensor is a tensor object from the underlying backend (Theano, TensorFlow or CNTK), which we augment with certain attributes that allow us to build a Keras model just by knowing the inputs and outputs of the model. (see [doc](https://keras.io/layers/core/))
# 
# -->```keras.layers.core.Lambda()```:
# 
# Wraps arbitrary expression as a Layer object. (see [doc](https://keras.io/layers/core/))
# 
# 
# 
# 
# 
# 

# In[33]:

from keras.layers.merge import concatenate
# Build U-Net model
inputs = keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)) 
s = keras.layers.core.Lambda(lambda x: x / 255) (inputs)

###########################################First Layer

c1 = keras.layers.convolutional.Conv2D(16, (3, 3), 
                                       activation='elu', kernel_initializer='he_normal', padding='same')(s)

c1 = keras.layers.core.Dropout(0.1) (c1)

c1 = keras.layers.convolutional.Conv2D(16, (3, 3), 
                                       activation='elu', kernel_initializer='he_normal', padding='same') (c1)

p1 = keras.layers.pooling.MaxPooling2D((2, 2)) (c1)

c2 = keras.layers.convolutional.Conv2D(32, (3, 3), 
                                       activation='elu', kernel_initializer='he_normal', padding='same') (p1)
c2 = keras.layers.core.Dropout(0.1) (c2)
c2 = keras.layers.convolutional.Conv2D(32, (3, 3),  
                                       activation='elu', kernel_initializer='he_normal', padding='same') (c2)
p2 = keras.layers.pooling.MaxPooling2D((2, 2)) (c2)

c3 = keras.layers.convolutional.Conv2D(64, (3, 3),   
                                       activation='elu', kernel_initializer='he_normal', padding='same') (p2)
c3 = keras.layers.core.Dropout(0.2) (c3)
c3 = keras.layers.convolutional.Conv2D(64, (3, 3),   
                                       activation='elu', kernel_initializer='he_normal', padding='same') (c3)
p3 = keras.layers.pooling.MaxPooling2D((2, 2)) (c3)

c4 = keras.layers.convolutional.Conv2D(128, (3, 3),  
                                       activation='elu', kernel_initializer='he_normal', padding='same') (p3)
c4 = keras.layers.core.Dropout(0.2) (c4)
c4 = keras.layers.convolutional.Conv2D(128, (3, 3),  
                                       activation='elu', kernel_initializer='he_normal', padding='same') (c4)
p4 = keras.layers.pooling.MaxPooling2D(pool_size=(2, 2)) (c4)

c5 = keras.layers.convolutional.Conv2D(256, (3, 3),  
                                       activation='elu', kernel_initializer='he_normal', padding='same') (p4)
c5 = keras.layers.core.Dropout(0.3) (c5)
c5 = keras.layers.convolutional.Conv2D(256, (3, 3),  
                                       activation='elu', kernel_initializer='he_normal', padding='same') (c5)

u6 = keras.layers.convolutional.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
u6 = concatenate([u6, c4])
c6 = keras.layers.convolutional.Conv2D(128, (3, 3),  
                                       activation='elu', kernel_initializer='he_normal', padding='same') (u6)
c6 = keras.layers.core.Dropout(0.2) (c6)
c6 = keras.layers.convolutional.Conv2D(128, (3, 3),  
                                       activation='elu', kernel_initializer='he_normal', padding='same') (c6)

u7 = keras.layers.convolutional.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = concatenate([u7, c3])
c7 = keras.layers.convolutional.Conv2D(64, (3, 3),   
                                       activation='elu', kernel_initializer='he_normal', padding='same') (u7)
c7 = keras.layers.core.Dropout(0.2) (c7)
c7 = keras.layers.convolutional.Conv2D(64, (3, 3),   
                                       activation='elu', kernel_initializer='he_normal', padding='same') (c7)

u8 = keras.layers.convolutional.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = concatenate([u8, c2])
c8 = keras.layers.convolutional.Conv2D(32, (3, 3),   
                                       activation='elu', kernel_initializer='he_normal', padding='same') (u8)
c8 = keras.layers.core.Dropout(0.1) (c8)
c8 = keras.layers.convolutional.Conv2D(32, (3, 3),   
                                       activation='elu', kernel_initializer='he_normal', padding='same') (c8)

u9 = keras.layers.convolutional.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
u9 = concatenate([u9, c1], axis=3)
c9 = keras.layers.convolutional.Conv2D(16, (3, 3),   
                                       activation='elu', kernel_initializer='he_normal', padding='same') (u9)
c9 = keras.layers.core.Dropout(0.1) (c9)
c9 = keras.layers.convolutional.Conv2D(16, (3, 3),   
                                       activation='elu', kernel_initializer='he_normal', padding='same') (c9)

outputs = keras.layers.convolutional.Conv2D(1, (1, 1), activation='sigmoid') (c9)

model = keras.models.Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])
model.summary()


# In[35]:

# Fit model
earlystopper = keras.callbacks.EarlyStopping(patience=5, verbose=1)
checkpointer = keras.callbacks.ModelCheckpoint('model-dsbowl2018-1.h5', verbose=1, save_best_only=True)
results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=2, 
                    callbacks=[earlystopper, checkpointer])


# #### V. Make predictions 

# In[38]:

# Predict on train, val and test
model = keras.models.load_model('model-dsbowl2018-1.h5', custom_objects={'mean_iou': mean_iou})
preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
preds_test = model.predict(X_test, verbose=1)

# Threshold predictions
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)

# Create list of upsampled test masks
preds_test_upsampled = []
for i in range(len(preds_test)):
    preds_test_upsampled.append(skimage.transform.resize(np.squeeze(preds_test[i]), 
                                       (sizes_test[i][0], sizes_test[i][1]), 
                                       mode='constant', preserve_range=True))


# In[51]:

# Perform a sanity check on some random training samples
ix = random.randint(0, len(preds_train_t))
skimage.io.imshow(X_train[ix])
plt.show()
skimage.io.imshow(np.squeeze(Y_train[ix]))
plt.show()
skimage.io.imshow(np.squeeze(preds_train_t[ix]))
plt.show()

