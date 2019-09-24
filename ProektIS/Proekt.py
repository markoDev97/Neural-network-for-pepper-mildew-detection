import numpy as np
import numpy.random as rd 
import pickle
import cv2 
from os import listdir
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras.layers import AveragePooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as pyplot
import glob
#somehow enter data in program and make appropriate adjustments
class_healthy=1
class_mildew=2
all_images, all_classes=[], []
max_height, max_width=-1, -1
for filename in glob.glob('./healthy_images/*.jpg'):
    img=cv2.imread(filename)
    img_grayscaled=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    all_images.append(img)
    all_classes.append(class_healthy)
for filename in glob.glob('./mildew_images/*.jpg'):
    img=cv2.imread(filename)
    img_grayscaled=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    all_images.append(img)
    all_classes.append(class_mildew)
#create appropriate model/cnn
model=Sequential()
"""
model.add(Conv2D(4, kernel_size=(5, 5)))
model.add(Activation('relu'))
model.add(Conv2D(4, kernel_size=(5, 5)))
model.add(AveragePooling2D(pool_size=(3, 3)))
model.add(Dense(2))"""
model.add(Dense(32, activation='relu', use_bias=True))
model.add(Dense(10, activation='sigmoid'))
model.add(Dense(2))
#congfigure and compile model
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
#train model appropriately
model.fit([np.array(None), np.array(None)], all_classes, epochs=5, batch_size=13)
