# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 12:50:08 2017

@author: JunTaniguchi
"""

import os, glob
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop

path = "/Users/JunTaniguchi/study_tensorflow/keras_project/read_place"
os.chdir(path)
from SpatialPyramidPooling import SpatialPyramidPooling

def use_spp_net_model(input_shape, NUM_CLASSES, optim):
    # モデルを構築(SPPNet)
    model = Sequential()
    model.add(Convolution2D(64, 3, 3, border_mode='same', input_shape=(None, None, 3)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.50))
    
    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.50))
    
    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.50))
    
    #model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(SpatialPyramidPooling([1, 2, 4]))
    model.add(Dense(NUM_CLASSES))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=optim,
                  metrics=['accuracy'])
    return model