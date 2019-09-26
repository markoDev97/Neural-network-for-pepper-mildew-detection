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
from keras.optimizers import *
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import glob
import math

def get_appropriate_place_for_unhealthy(id):
    return (id-13)*2+1 

def shuffle_for_uniformity(image_list, classes_list):
    threshold=math.ceil(len(image_list)/2)
    for id in range(threshold, len(image_list)):
        new_id=get_appropriate_place_for_unhealthy(id)
        image_list[new_id], image_list[id]=image_list[id], image_list[new_id]
        classes_list[new_id], classes_list[id]=classes_list[id], classes_list[new_id]


#somehow enter data in program and make appropriate adjustments
class_healthy=0
class_mildew=1
all_images, all_classes=[], []
default_image_size=(320, 320)
for filename in glob.glob('./healthy_images/*.jpg'):
    img=cv2.imread(filename)
    img=cv2.resize(img, default_image_size)
    all_images.append(img_to_array(img))
    all_classes.append(class_healthy)
for filename in glob.glob('./mildew_images/*.jpg'):
    img=cv2.imread(filename)
    img=cv2.resize(img, default_image_size)
    all_images.append(img_to_array(img))
    all_classes.append(class_mildew)
all_images_scaled=np.array(all_images, dtype=np.float16)/255.0

shuffle_for_uniformity(all_images_scaled, all_classes)#1, 2, 1, 2, ...
x_train, x_test, y_train, y_test=train_test_split(all_images_scaled, all_classes, test_size=0.2, random_state=42)
aug = ImageDataGenerator(
    rotation_range=25, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, 
    zoom_range=0.2,horizontal_flip=True, 
    fill_mode="nearest")
#create appropriate model/cnn

model=Sequential()
input_shape_=320, 320, 3
model.add(Conv2D(32, (5, 5), padding="same", input_shape=input_shape_))
model.add(Activation('elu'))
model.add(AveragePooling2D(2, 2))
model.add(Dropout(0.15))
model.add(Conv2D(32, (5, 5), padding="same"))
model.add(Activation('elu'))
model.add(Conv2D(32, (2, 2), padding="same"))
model.add(Dense(20, activation="tanh"))
model.add(Flatten("channels_last"))
model.add(Dense(1))
opt=SGD(0.2, 0.1)
model.compile(optimizer=opt,
               loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(x=x_train, y=y_train, batch_size=4, epochs=1, validation_data=(x_test, y_test))
# model = Sequential()
# chanDim = -1
# if K.image_data_format() == "channels_first":
#     chanDim = 1
# height, width, depth=320, 320, 3
# input_shape_=(height, width, depth)
# EPOCHS = 4
# INIT_LR = 0.2
# BS = 2
# model.add(Conv2D(32, (3, 3), padding="same", input_shape=input_shape_))
# model.add(Activation("relu"))
# model.add(BatchNormalization(axis=chanDim))
# model.add(Dropout(0.25))
# model.add(Conv2D(64, (3, 3), padding="same"))
# model.add(Activation("relu"))
# model.add(BatchNormalization(axis=chanDim))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(64, (3, 3), padding="same"))
# model.add(Activation("relu"))
# model.add(BatchNormalization(axis=chanDim))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Conv2D(128, (3, 3), padding="same"))
# model.add(Activation("relu"))
# model.add(BatchNormalization(axis=chanDim))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(128, (3, 3), padding="same"))
# model.add(Activation("relu"))
# model.add(BatchNormalization(axis=chanDim))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(1024))
# model.add(Activation("relu"))
# model.add(BatchNormalization())
# model.add(Dropout(0.125))
# model.add(Dense(1))
# model.summary()
# #configure and compile model
# opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
# model.compile(optimizer=opt,
#               loss='binary_crossentropy',
#               metrics=['accuracy'])
# #train model appropriately
# history = model.fit_generator(
#     aug.flow(x_train, y_train, batch_size=BS),
#     validation_data=(x_test, y_test),
#     steps_per_epoch=len(x_train) // BS,
#     epochs=EPOCHS, verbose=1
#     )
# model.fit(all_images_scaled, all_classes, epochs=2, batch_size=4)
