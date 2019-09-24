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
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as pyplot
#create model/cnn
model = Sequential([
    Dense(10),
    Activation('relu'),
    Dense(3),
    Activation('relu'),
    Dense(10),
    Activation('relu'),
    Dense(2),
])
#congifure and compile model
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
#somehow enter data in program
data=rd.random((1000, 10))
labels = rd.randint(2, size=(1000, 2))
#for i in data:
    #print(i)
#train neural network
model.fit(data, labels, epochs=10, batch_size=250)
