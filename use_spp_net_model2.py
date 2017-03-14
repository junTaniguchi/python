# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 12:50:08 2017
@author: JunTaniguchi
"""

import os, glob
from keras.layers import Activation
from keras.layers import AtrousConvolution2D
from keras.layers import Convolution2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import GlobalAveragePooling2D
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import merge
from keras.layers import Reshape
from keras.layers import ZeroPadding2D
from keras.layers import BatchNormalization
from keras.models import Model


path = "/Users/j13-taniguchi/study_tensorflow/keras_project/read_place"
os.chdir(path)
from SpatialPyramidPooling import SpatialPyramidPooling
from keras.utils.visualize_util import plot



def use_spp_net_model2(input_shape, NUM_CLASSES):
    # モデルを構築(SPPNet)
    net = {}
    # Block 1
    input_tensor = input_tensor = Input(shape=(None, None, 3))
    #img_size = (input_shape[1], input_shape[0])
    net['input'] = input_tensor
    net['conv1_1'] = Convolution2D(64, 3, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv1_1')(net['input'])
    net['conv1_2'] = Convolution2D(64, 3, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv1_2')(net['conv1_1'])
    net['pool1'] = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same',
                                name='pool1')(net['conv1_2'])
    # Block 2
    net['conv2_1'] = Convolution2D(128, 3, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv2_1')(net['pool1'])
    net['conv2_2'] = Convolution2D(128, 3, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv2_2')(net['conv2_1'])
    net['pool2'] = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same',
                                name='pool2')(net['conv2_2'])
    # Block 3
    net['conv3_1'] = Convolution2D(256, 3, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv3_1')(net['pool2'])
    net['conv3_2'] = Convolution2D(256, 3, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv3_2')(net['conv3_1'])
    net['conv3_3'] = Convolution2D(256, 3, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv3_3')(net['conv3_2'])
    net['pool3'] = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same',
                                name='pool3')(net['conv3_3'])
    # Block 4
    net['conv4_1'] = Convolution2D(512, 3, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv4_1')(net['pool3'])
    net['conv4_2'] = Convolution2D(512, 3, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv4_2')(net['conv4_1'])
    net['conv4_3'] = Convolution2D(512, 3, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv4_3')(net['conv4_2'])
    net['pool4'] = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same',
                                name='pool4')(net['conv4_3'])
    # Block 5
    net['conv5_1'] = Convolution2D(512, 3, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv5_1')(net['pool4'])
    net['conv5_2'] = Convolution2D(512, 3, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv5_2')(net['conv5_1'])
    net['conv5_3'] = Convolution2D(512, 3, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv5_3')(net['conv5_2'])
    net['pool5'] = MaxPooling2D((3, 3), strides=(1, 1), border_mode='same',
                                name='pool5')(net['conv5_3'])
    # FC6
    net['fc6'] = AtrousConvolution2D(1024, 3, 3, atrous_rate=(6, 6),
                                     activation='relu', border_mode='same',
                                     name='fc6')(net['pool5'])                                
    # FC7 モデルを分離    
    net['sp1'] = Convolution2D(1024, 1, 1, activation='relu',
                               border_mode='same', name='sp1')(net['fc6'])
    net['sp2'] = Convolution2D(1024, 1, 1, activation='relu',
                               border_mode='same', name='sp2')(net['fc6'])
    # FC8-1 category
    net['conv8_1_1'] = Convolution2D(1024, 1, 1, activation='relu',
                                     border_mode='same', name='conv8_1_1')(net['sp1'])
    net['conv8_1_2'] = Convolution2D(1024, 1, 1, activation='relu',
                                     border_mode='same', name='conv8_1_2')(net['conv8_1_1'])
    net['spp1'] = SpatialPyramidPooling([1, 2, 4], #activation='relu',
                                        name='spp1')(net['conv8_1_2'])
    net['dense8_1'] = Dense(NUM_CLASSES, activation='softmax',
                            name='dense8_1')(net['spp1'])
    # FC8-2 xmin, xmax, ymin, ymax
    net['conv8_2_1'] = Convolution2D(1024, 1, 1, activation='relu',
                                     border_mode='same', name='conv8_2_1')(net['sp1'])
    net['conv8_2_2'] = Convolution2D(1024, 1, 1, activation='relu',
                                     border_mode='same', name='conv8_2_2')(net['conv8_2_1'])
    net['spp2'] = SpatialPyramidPooling([1, 2, 4], #activation='relu',
                                        name='spp2')(net['conv8_2_2'])
    net['dense8_2'] = Dense(4, activation='relu',
                            name='dense8_2')(net['spp2'])
    net['Normalize8_2'] = BatchNormalization(mode=2)(net['dense8_2'])                                      
    net['predictions'] = merge([net['dense8_1'],
                                net['Normalize8_2']],
                                mode='concat',
                                concat_axis=1,
                                name='predictions')
    model = Model(net['input'], net['predictions'])

    return model