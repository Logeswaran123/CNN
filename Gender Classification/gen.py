# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 15:27:23 2020

@author: admin
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import cv2
import matplotlib.pyplot as plt
import os
import seaborn as sns

from PIL import Image
import imageio
from scipy import misc
from os import listdir
from os.path import isfile, join
import numpy as np
from scipy import misc
from random import shuffle
from collections import Counter
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils.np_utils import to_categorical
from smallervggnet import SmallerVGGNet



# initial parameters
epochs = 100
lr = 1e-3
batch_size = 64




path = 'utkface_aligned_cropped/UTKFace/UTKFace/'
files = os.listdir(path)
print(len(files))

shuffle(files)
gender = [i.split('_')[1] for i in files]
print(len(gender))

classes = []
for i in gender:
    i = int(i)
    classes.append(i)
print(len(classes))

X_data =[]
for file in files:
    print(file)
    face = cv2.imread('utkface_aligned_cropped/UTKFace/UTKFace/' +file)
    face = cv2.resize(face, (96, 96))
    X_data.append(face)

X = np.squeeze(X_data)
print(X.shape)

# normalize data
X = X.astype('float32')
X /= 255

print(classes[:10])
classes = np.array(classes)

# split dataset for training and validation
(trainX, testX, trainY, testY) = train_test_split(X, classes, test_size=0.2,
                                                  random_state=42)
trainY = to_categorical(trainY, num_classes=2)
testY = to_categorical(testY, num_classes=2)

# augmenting datset 
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")

# build model
model = SmallerVGGNet.build(width = 96, height = 96, depth = 3,
                            classes = 2)

# compile the model
opt = Adam(lr=lr, decay=lr/epochs)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the model
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=batch_size),
                        validation_data=(testX,testY),
                        steps_per_epoch=len(trainX) // batch_size,
                        epochs=epochs, verbose=1)

# save the model to disk
model.save('gender_detection.model')

# plot training/validation loss/accuracy
plt.style.use("ggplot")
plt.figure()
N = epochs
plt.plot(np.arange(0,N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0,N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0,N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0,N), H.history["val_acc"], label="val_acc")

plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper right")

# save plot to disk
plt.savefig('plot.jpg')



