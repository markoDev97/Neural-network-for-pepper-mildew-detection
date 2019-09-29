import cv2
import numpy as np

import matplotlib.pyplot as plt

import os
import random
import gc

from sklearn.model_selection import train_test_split

from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator


def read_images_and_labels(list_of_images):
    """
    Returns list of labels and list of images as arrays from list of images. Label name is within image name.

    :param list_of_images: list of images
    :return: list of labels and list of images as arrays
    """
    X = []
    y = []
    for image in list_of_images:
        X.append(cv2.imread(image, cv2.IMREAD_COLOR))
        if 'B.Spot' in image:
            y.append(1)
        else:
            y.append(0)

    return X, y


# load data
infected_dir = 'Pepper__bell___Bacterial_spot'
healty_dir = 'Pepper__bell___healthy'
test_dir = 'test'

imgs = [infected_dir + "/" + i for i in os.listdir(infected_dir)] + \
       [healty_dir + "/" + i for i in os.listdir(healty_dir)]
# shufffle images
random.shuffle(imgs)

#
images, labels = read_images_and_labels(imgs)

test_images, test_labels = read_images_and_labels(os.listdir(test_dir))
del imgs
gc.collect()

X = np.array(images)
y = np.array(labels)

# split the data into training and test set
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=2)

del X
del y
gc.collect()

ntrain = len(X_train)
nval = len(X_val)
batch_size = 32

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)))
model.add(layers.MaxPool2D(2, 2))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPool2D(2, 2))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPool2D(2, 2))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# shows our model
model.summary()

model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])

# Lets create the augmentation configuration
# This helps prevent overfitting, since we are using a small dataset
train_datagen = ImageDataGenerator(rescale=1. / 255,  # Scale the image between 0 and 1
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True, )

val_datagen = ImageDataGenerator(rescale=1. / 255)  # We do not augment validation data. we only perform rescale

# Create the image generators
train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)

# The training part
# We train for 64 epochs with about 100 steps per epoch
history = model.fit_generator(train_generator,
                              steps_per_epoch=ntrain // batch_size,
                              epochs=64,
                              validation_data=val_generator,
                              validation_steps=nval // batch_size)

# Save the model
model.save_weights('model_wieghts.h5')
model.save('model_keras.h5')

# lets plot the train and val curve
# get the details form the history object
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# Train and validation accuracy
plt.plot(epochs, acc, 'b', label='Training accurarcy')
plt.plot(epochs, val_acc, 'r', label='Validation accurarcy')
plt.title('Training and Validation accurarcy')
plt.legend()

plt.figure()
# Train and validation loss
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()

plt.show()