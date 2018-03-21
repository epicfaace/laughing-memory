# -*- coding: utf-8 -*-
"""keras unet data aug.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/13yU10fdtkwfkDcpOt8X8BlXvAf5ZP2_w

## U-net First Implementation

The architecture used is the so-called U-Net, which is very common for image segmentation problems such as this. It works well even with small datasets (which is weird for a NN).

### I. Initial Setup 


#### a) Libraries
"""

# wget https://raw.githubusercontent.com/epicfaace/laughing-memory/master/Data/stage1_test.zip -cq
# wget https://raw.githubusercontent.com/epicfaace/laughing-memory/master/Data/stage1_train.zip -cq

# mkdir stage1_train stage1_test

# unzip -q -o stage1_train.zip -d stage1_train/
# unzip -q -o stage1_test.zip -d stage1_test/ 
# pip install -q tqdm keras
# apt-get -qq install -y libsm6 libxext6 && pip install -q -U opencv-python
# pip install -e git+https://github.com/epicfaace/laughing-memory-util.git#egg=laughing_memory_util

#@title Save augmented data

import datetime
# 
MODEL_NAME = 'model-dsbowl2018-Data-Aug-2-lr-ud-BN-bestOnlyTrue-dropA0.1_311111123-MeanIoU-200e-Res256'
d = datetime.date.today()
DIR_NAME = 'Submission Results/{:02d}{:02d}/{}'.format(d.month, d.day, MODEL_NAME)
MODEL_NAME = DIR_NAME + "/" + MODEL_NAME
from shutil import copy2
import pathlib
import os
pathlib.Path(MODEL_NAME).mkdir(parents=True, exist_ok=True) 
copy2(os.path.basename(__file__), DIR_NAME)
import numpy as np
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
# https://gist.github.com/oeway/2e3b989e0343f0884388ed7ed82eb3b0
# Function to distort image
def elastic_transform(image, alpha=1000, sigma=30, spline_order=1, mode='nearest', random_state=np.random):
    """Elastic deformation of image as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    assert image.ndim == 3
    shape = image.shape[:2]

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1),
                         sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1),
                         sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = [np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))]
    result = np.empty_like(image)
    for i in range(image.shape[2]):
        result[:, :, i] = map_coordinates(
            image[:, :, i], indices, order=spline_order, mode=mode).reshape(shape)
    return result


import skimage.io, skimage.transform, skimage.morphology
import os
import sys
import random
from tqdm import tqdm
import numpy as np

def data_aug(TRAIN_PATH, TEST_PATH, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS):
    #os.walk example
    i = 0
    for (path, dirs, files) in os.walk(TRAIN_PATH):
        print(path,dirs,files)
        i+=1
        if i == 4:
            break
    # Get train and test IDs (Next() returns the next item from the iterator)
    train_ids = next(os.walk(TRAIN_PATH))[1] #We are interested in list at index 1: List with directories ('str' format) at path 
    test_ids = next(os.walk(TEST_PATH))[1]
    print("Size of the training set: " + str(len(train_ids)))
    print("Size of the test set: " + str(len(test_ids)))
    # Get and resize train images and masks(Initialize containers for storing downsampled images)

    num_aug = 2
    X_train_aug = np.zeros((len(train_ids) * num_aug, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    Y_train_aug = np.zeros((len(train_ids) * num_aug, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

    X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8) #dtype recommended for pixels
    Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool) #dtype recommended for boolean matric
    #X_train, Y_train = [],[]

    #Note that the output of the NN in image segmentation (two classes) is a boolean matrix 
    #with the dimensions of the image

    print('Getting and resizing train images and masks ... ')

    sys.stdout.flush() #forces it to "flush" the buffer

    #Tip: Use tqdm() on top of the iterator to have a progress bar
    #enumerate() returns a tuple (index, item) of the iterator. Define 2 looping variables if you use enumerate() 

    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
        path = TRAIN_PATH + id_
        h, w = IMG_HEIGHT, IMG_WIDTH
        img = skimage.io.imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
        #Image resizing to lower resolution
        img = skimage.transform.resize(img, (h, w), mode='constant', preserve_range=True)
        #h, w = img.shape[0], img.shape[1]
        #X_train is a tensor of order 4: A "cube" of data <-> n matrices stacked together
        X_train[n] = img
        #X_train.append(img)
        
        mask = np.zeros((h, w, 1), dtype=np.bool) #Initialize container of the mask
        for mask_file in next(os.walk(path + '/masks/'))[2]:           #Use index 2 for getting name of files (.png)
            mask_ = skimage.io.imread(path + '/masks/' + mask_file)
            #print(mask_.shape) Remove the comment to see how that mask_ is an array of two dimensions Not three
            #Insert a new axis
            mask_ = np.expand_dims(skimage.transform.resize(mask_, (h, w), mode='constant', 
                                        preserve_range=True), axis=-1) #preserve_range=True is important
            #axis = -1 adds an axis at the end of the tuple of dimensions of a np.array
            mask = np.maximum(mask, mask_)
            #mask gets updated at each loop and includes all the masks
        
        Y_train[n] = mask
        #Y_train.append(mask) #Stores all the masks (true labels in a tensor of order 4: 1 tensor of order 3 per mask)
        
        #v_min, v_max = np.percentile(img, (0.2, 99.8))
        #better_img = exposure.rescale_intensity(img, in_range=(v_min, v_max))
        X_train_aug[num_aug*n + 0] = img[:, ::-1] # horizontal flip
        Y_train_aug[num_aug*n + 0] = np.fliplr(mask)
        X_train_aug[num_aug*n + 1] = img[::-1, :] # vertical flip
        Y_train_aug[num_aug*n + 1] = np.flipud(mask)
        """X_train_aug[num_aug*n + 2] = elastic_transform(img)
        Y_train_aug[num_aug*n + 2] = elastic_transform(mask)
        
        for index in range(3, 10):
            randH = random.randint(20,h - 20)
            randW = random.randint(20,w - 20)
            X_train_aug[num_aug*n + index] = skimage.transform.resize(img[:randH, :randW], (h, w), mode='constant', preserve_range=True)
            Y_train_aug[num_aug*n + index] = skimage.transform.resize(mask[:randH, :randW], (h, w), mode='constant', preserve_range=True)

        """ 
        """skimage.io.imshow(X_train[n])
        plt.show()
        skimage.io.imshow(np.squeeze(Y_train[n]))
        plt.show()
        indexToShow = num_aug*n + 3
        skimage.io.imshow(X_train_aug[indexToShow])
        plt.show()
        skimage.io.imshow(np.squeeze(Y_train_aug[indexToShow]))
        plt.show()
        break"""
        
        """skimage.io.imshow(X_train_aug[3*n+3])
        plt.show()
        skimage.io.imshow(np.squeeze(Y_train[3*n]))
        plt.show()
        skimage.io.imshow(np.squeeze(Y_train_aug[3*n+3]))
        plt.show()
        break"""

    print('\n Training images succesfully downsampled')
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
        
    print('\n Test images successfully downsampled!')
    return np.concatenate((X_train_aug, X_train), axis=0), np.concatenate((Y_train_aug, Y_train), axis=0), train_ids, test_ids, X_test, sizes_test

# ls -l

# pip install -q tqdm keras
# apt-get -qq install -y libsm6 libxext6 && pip install -q -U opencv-python
# pip install --upgrade git+https://github.com/epicfaace/laughing-memory-util.git#egg=laughing_memory_util
# import laughing_memory_util
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3
TRAIN_PATH = 'Data/stage1_train/'
TEST_PATH = 'Data/stage1_test/'
X_train, Y_train, train_ids,  test_ids, X_test, sizes_test = data_aug(TRAIN_PATH, TEST_PATH, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)



#@title Load augmented data

print("HI")

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

"""Keras has three backend implementations available: the TensorFlow backend, the Theano backend, and the CNTK backend. It relies on a specialized, well-optimized tensor manipulation library, serving as its "backend engine".

#### b) Warnings filter

The warnings filter controls whether warnings are ignored, displayed, or turned into errors (raising an exception).
The warnings filter maintains an ordered list of filter specifications. Each entry is a tuple of the form ```(action, message, category, module, lineno).```

For more documentation: [Warning Control](https://docs.python.org/2/library/warnings.html#warning-filter)
"""

#Parameters: Action, 
warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

"""### II. Data Loading

### III. Definition of the evaluation metric for the competition

In this section, we try to define the mean average precision at different intersection over union (IoU) thresholds metric in Keras. TensorFlow has a mean IoU metric, but it doesn't have any native support for the mean over multiple thresholds, so we need to implement this by ourselves. However, in a first stage, we used the [Dice Coefficient](https://stats.stackexchange.com/questions/195006/is-the-dice-coefficient-the-same-as-accuracy/253992). **Needs review**

The Dice score is not only a measure of how many positives you find, but it also penalizes for the false positives that the method finds, similar to precision. So it is more similar to precision than accuracy. The only difference is the denominator, where you have the total number of positives instead of only the positives that the method finds. So the Dice score is also penalizing for the positives that your algorithm/method could not find.

Now, we are going to try to define a IoU metric
"""

#Define IoU metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2, y_true)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

"""### IV. Definition and training of the Neural Network (U-net)

Some observations on the functions used:

-->```keras.layers.Input()```:

A Keras tensor is a tensor object from the underlying backend (Theano, TensorFlow or CNTK), which we augment with certain attributes that allow us to build a Keras model just by knowing the inputs and outputs of the model. (see [doc](https://keras.io/layers/core/))

-->```keras.layers.core.Lambda()```:

Wraps arbitrary expression as a Layer object. (see [doc](https://keras.io/layers/core/))
"""

# Batch Norm!
# Build U-Net model

from keras.layers.merge import concatenate

#### Input Layer

inputs = keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)) # Input() is used to instantiate a Keras tensor

s = keras.layers.core.Lambda(lambda x: x / 255) (inputs) #Normalize all pixels

##### Block 1: 2 Conv + Dropout + Maxpool

c1 = keras.layers.convolutional.Conv2D(16, (3, 3), 
                                       activation='elu', kernel_initializer='he_normal', padding='same')(s)  #params: num filters, kernel_size

c1 = keras.layers.BatchNormalization()(c1)

c1 = keras.layers.core.Dropout(0.3) (c1)

c1 = keras.layers.convolutional.Conv2D(16, (3, 3), 
                                       activation='elu', kernel_initializer='he_normal', padding='same', 
                                       kernel_regularizer=keras.regularizers.l2(0.)) (c1)

c1 = keras.layers.BatchNormalization()(c1)

p1 = keras.layers.pooling.MaxPooling2D((2, 2)) (c1)

##### Block 2: 2 Conv + Dropout + Maxpool

c2 = keras.layers.convolutional.Conv2D(32, (3, 3), 
                                       activation='elu', kernel_initializer='he_normal', padding='same',
                                       kernel_regularizer=keras.regularizers.l2(0.)) (p1)

c2 = keras.layers.BatchNormalization()(c2)

c2 = keras.layers.core.Dropout(0.1) (c2)

c2 = keras.layers.convolutional.Conv2D(32, (3, 3),  
                                       activation='elu', kernel_initializer='he_normal', padding='same',
                                       kernel_regularizer=keras.regularizers.l2(0.)) (c2)

c2 = keras.layers.BatchNormalization()(c2)

p2 = keras.layers.pooling.MaxPooling2D((2, 2)) (c2)

##### Block 3: 2 Conv + Dropout + Maxpool

c3 = keras.layers.convolutional.Conv2D(64, (3, 3),   
                                       activation='elu', kernel_initializer='he_normal', padding='same',
                                       kernel_regularizer=keras.regularizers.l2(0.)) (p2)

c3 = keras.layers.BatchNormalization()(c3)

c3 = keras.layers.core.Dropout(0.1) (c3)

c3 = keras.layers.convolutional.Conv2D(64, (3, 3),   
                                       activation='elu', kernel_initializer='he_normal', padding='same',
                                       kernel_regularizer=keras.regularizers.l2(0.)) (c3)

c3 = keras.layers.BatchNormalization()(c3)

p3 = keras.layers.pooling.MaxPooling2D((2, 2)) (c3)

##### Block 4: 2 Conv + Dropout + Maxpool

c4 = keras.layers.convolutional.Conv2D(128, (3, 3),  
                                       activation='elu', kernel_initializer='he_normal', padding='same',
                                       kernel_regularizer=keras.regularizers.l2(0.)) (p3)

c4 = keras.layers.BatchNormalization()(c4)

c4 = keras.layers.core.Dropout(0.1) (c4)

c4 = keras.layers.convolutional.Conv2D(128, (3, 3),  
                                       activation='elu', kernel_initializer='he_normal', padding='same',
                                       kernel_regularizer=keras.regularizers.l2(0.)) (c4)

c4 = keras.layers.BatchNormalization()(c4)

p4 = keras.layers.pooling.MaxPooling2D(pool_size=(2, 2)) (c4)

##### Block 5: 2 Conv + Dropout + Maxpool

c5 = keras.layers.convolutional.Conv2D(256, (3, 3),  
                                       activation='elu', kernel_initializer='he_normal', padding='same',
                                       kernel_regularizer=keras.regularizers.l2(0.)) (p4)

c5 = keras.layers.BatchNormalization()(c5)

c5 = keras.layers.core.Dropout(0.1) (c5)

c5 = keras.layers.convolutional.Conv2D(256, (3, 3),  
                                       activation='elu', kernel_initializer='he_normal', padding='same',
                                       kernel_regularizer=keras.regularizers.l2(0.)) (c5)

c5 = keras.layers.BatchNormalization()(c5)

#### Block 6: Deconvolution + Concatenate + Convolution + Dropout + Convolution

u6 = keras.layers.convolutional.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', 
                                                kernel_regularizer=keras.regularizers.l2(0.)) (c5)

u6 = concatenate([u6, c4])

c6 = keras.layers.convolutional.Conv2D(128, (3, 3),  
                                       activation='elu', kernel_initializer='he_normal', padding='same',
                                       kernel_regularizer=keras.regularizers.l2(0.)) (u6)

c6 = keras.layers.core.Dropout(0.1) (c6)

c6 = keras.layers.convolutional.Conv2D(128, (3, 3),  
                                       activation='elu', kernel_initializer='he_normal', padding='same',
                                       kernel_regularizer=keras.regularizers.l2(0.)) (c6)

#### Block 7: Deconvolution + Concatenate + Convolution + Dropout + Convolution

u7 = keras.layers.convolutional.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', 
                                               kernel_regularizer=keras.regularizers.l2(0.)) (c6)  
u7 = concatenate([u7, c3])

c7 = keras.layers.convolutional.Conv2D(64, (3, 3),   
                                       activation='elu', kernel_initializer='he_normal', padding='same', 
                                       kernel_regularizer=keras.regularizers.l2(0.)) (u7)
c7 = keras.layers.core.Dropout(0.1) (c7)
c7 = keras.layers.convolutional.Conv2D(64, (3, 3),   
                                       activation='elu', kernel_initializer='he_normal', padding='same', 
                                       kernel_regularizer=keras.regularizers.l2(0.)) (c7)

#### Block 8: Deconvolution + Concatenate + Convolution + Dropout + Convolution

u8 = keras.layers.convolutional.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same', 
                                                kernel_regularizer=keras.regularizers.l2(0.)) (c7)

u8 = concatenate([u8, c2])

c8 = keras.layers.convolutional.Conv2D(32, (3, 3),   
                                       activation='elu', kernel_initializer='he_normal', padding='same',
                                       kernel_regularizer=keras.regularizers.l2(0.)) (u8)

c8 = keras.layers.core.Dropout(0.2) (c8)

c8 = keras.layers.convolutional.Conv2D(32, (3, 3),   
                                       activation='elu', kernel_initializer='he_normal', padding='same',
                                       kernel_regularizer=keras.regularizers.l2(0.)) (c8)

#### Block 9: Deconvolution + Concatenate + Convolution + Dropout + Convolution

u9 = keras.layers.convolutional.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same', 
                                               kernel_regularizer=keras.regularizers.l2(0.)) (c8)

u9 = concatenate([u9, c1], axis=3)

c9 = keras.layers.convolutional.Conv2D(16, (3, 3),   
                                       activation='elu', kernel_initializer='he_normal', padding='same', 
                                       kernel_regularizer=keras.regularizers.l2(0.)) (u9)

c9 = keras.layers.core.Dropout(0.3) (c9)

c9 = keras.layers.convolutional.Conv2D(16, (3, 3),   
                                       activation='elu', kernel_initializer='he_normal', padding='same',
                                       kernel_regularizer=keras.regularizers.l2(0.)) (c9)

#### Output Layer: Logistic Unit

outputs = keras.layers.convolutional.Conv2D(1, (1, 1), activation='sigmoid') (c9)

model = keras.models.Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])
model.summary()
"""#### a) Training set up

A callback is a set of functions to be applied at given stages of the training procedure. You can use callbacks to get a view on internal states and statistics of the model during training.

Some observations on the functions used:

-->```keras.callbacks.EarlyStopping()```: Stop training when a monitored quantity has stopped improving. Parameters:

- patience: number of epochs with no improvement after which training will be stopped

- verbose: verbosity mode

--> ```keras.callbacks.ModelCheckpoint()```: Save the model after every epoch. Parameters:

- filepath: string, path to save the model file
- save_best_only: if save_best_only = True, the latest best model according to the quantity monitored will not be overwritten.

--> ```model.fit()```: Parameters:

- validation_split: Float between 0 and 1. Fraction of the training data to be used as validation data. The model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch. The validation data is selected from the last samples in the x and y data provided, before shuffling.

- batch_size: Integer or None. Number of samples per gradient update. If unspecified, it will default to 32.
"""

MODEL_CHECKPOINT_FILE_NAME = MODEL_NAME + ".h5"
MODEL_WEIGHTS_FILE_NAME = MODEL_NAME + ".h5"

print(len(X_train) / 670 * 0.5)

# ls -l

### Fit model



# Keras models are trained on Numpy arrays of input data and labels. For training a model, you will typically use the  fit function.
earlystopper = keras.callbacks.EarlyStopping(patience=100, verbose=1) 
checkpointer = keras.callbacks.ModelCheckpoint(MODEL_CHECKPOINT_FILE_NAME, verbose=1, save_best_only=True)
results = model.fit((X_train), (Y_train), validation_split=len(train_ids) / len(X_train) * 0.1, batch_size=16, epochs=200, 
                    callbacks=[earlystopper, checkpointer])

# from keras.models import load_model
# model = load_model("model-dsbowl2018-Data-Aug-10-MeanIoU-50e-Res256.h5")

import matplotlib.pyplot as plt

history = results #Output model.fit()

# list all data in history
print(history.history.keys())

# summarize history for accuracy
fig_accuracy = plt.figure(1)
plt.plot(history.history['mean_iou'])
plt.plot(history.history['val_mean_iou'])
plt.title('Mean IoU evolution')
plt.ylabel('Mean IoU')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
fig_accuracy.savefig( MODEL_NAME + '-Accuracy_model' + '.png')

# summarize history for loss
fig_loss = plt.figure(2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss Evolution')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
fig_loss.savefig(MODEL_NAME+'-Loss_model' + '.png')

# ls -l


"""#### V. Make predictions"""

# Load trained U-net with best weights
model = keras.models.load_model(MODEL_CHECKPOINT_FILE_NAME, custom_objects={'mean_iou': mean_iou}) 
#custom_objects: Optional dictionary mapping names (strings) to custom classes or functions to be considered during deserialization.

# Predict
preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1) #Train set: You know, Keras uses the last 0.1 as dev set
preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1) #Dev set
preds_test = model.predict(X_test, verbose=1) #Test set

# Threshold predictions
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)

print(preds_test.shape)

# Create list of upsampled test masks
preds_test_upsampled = []
for i in range(len(preds_test)):
    preds_test_upsampled.append(skimage.transform.resize(np.squeeze(preds_test[i]), 
                                       (sizes_test[i][0], sizes_test[i][1]), 
                                       mode='constant', preserve_range=True))

# Perform a sanity check on some random training samples
"""ix = random.randint(0, len(preds_train_t))

skimage.io.imshow(X_train[ix])
plt.show()
skimage.io.imshow(np.squeeze(Y_train[ix]))
plt.show()
skimage.io.imshow(np.squeeze(preds_train_t[ix]))
plt.show()

# Perform a sanity check on some random validation samples
ix = random.randint(0, len(preds_val_t))

skimage.io.imshow(X_train[int(X_train.shape[0]*0.9):][ix])
plt.show()
skimage.io.imshow(np.squeeze(Y_train[int(Y_train.shape[0]*0.9):][ix]))
plt.show()
skimage.io.imshow(np.squeeze(preds_val_t[ix]))
plt.show()
"""
"""### V. Encode and submit results"""

# Run-length encoding from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python

def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def prob_to_rles(x, cutoff=0.5):
    lab_img = skimage.morphology.label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)

new_test_ids = []
rles = []
for n, id_ in enumerate(test_ids):
    rle = list(prob_to_rles(preds_test_upsampled[n]))
    rles.extend(rle)
    new_test_ids.extend([id_] * len(rle))

# Create submission DataFrame
sub = pd.DataFrame()
sub['ImageId'] = new_test_ids
sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
sub.to_csv(MODEL_NAME + '.csv', index=False)
