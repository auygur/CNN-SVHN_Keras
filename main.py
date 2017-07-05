
from __future__ import absolute_import
from __future__ import print_function

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils import np_utils

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from keras.layers import BatchNormalization

from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD, Adadelta, Adagrad
import scipy.io

import os
import itertools
import numpy as np



batch_size = 128
nb_classes = 10
nb_epoch = 20
Train = scipy.io.loadmat('/rigel/edu/coms4995/train_32x32.mat')
Test = scipy.io.loadmat('/rigel/edu/coms4995/test_32x32.mat')

X_train = Train['X']
y_train = Train['y']

X_test = Test['X']
y_test = Test['y']

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

X_train = X_train[np.newaxis,...]
X_train = np.swapaxes(X_train,0,4).squeeze()



X_test = X_test[np.newaxis,...]
X_test = np.swapaxes(X_test,0,4).squeeze()




np.place(y_train,y_train == 10,0)
np.place(y_test,y_test == 10,0)

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)


model = Sequential()


#WITHOUT batch normalization
model.add(Convolution2D(32, 3, 3, border_mode='same',
                        input_shape=(32, 32, 3)))


model.add(Activation('relu'))



model.add(Convolution2D(32, 3, 3))


model.add(Activation('relu'))


model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))



model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))



model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())


model.add(Dense(512))
model.add(Activation('relu'))



model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))


sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, verbose=0)
print('loss:', score[0])
print('Test accuracy:', score[1])



#Test score: 0.332611555633
#Test accuracy: 0.904732636755



# WITH Batch Normalization

model = Sequential()

model.add(Convolution2D(32, 3, 3, border_mode='same',
                        input_shape=(32, 32, 3)))


model.add(Activation('relu'))
model.add(BatchNormalization())


model.add(Convolution2D(32, 3, 3))


model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())


model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(BatchNormalization())


model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(BatchNormalization())

model.add(Dense(512))
model.add(Activation('relu'))
model.add(BatchNormalization())


model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))


sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=20, verbose=1, validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, verbose=0)
print('loss:', score[0])
print('Test accuracy:', score[1])


#loss: 0.2233744846
#Test accuracy: 0.939420712969



