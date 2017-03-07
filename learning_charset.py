# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 13:05:42 2017

@author: j13-taniguchi
"""

from keras.models import Sequential
import tensorflow
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from sklearn import cross_validation
import numpy as np
import matplotlib.pyplot as plt
import json

image_w = 28
image_h = 28


def main():
    # フォント画像のデータを読む
    xy = np.load("/Users/j13-taniguchi/study_tensorflow/keras_project/image/japanese_dataset.npz")
    X = xy["x"]
    Y = xy["y"]
    # データを正規化
    #X = X.reshape(X.shape[0], image_w * image_h).astype('float32')
    X /= 255
    # ラベル作成
    Y = np_utils.to_categorical(Y, 2195)
    with open("/Users/j13-taniguchi/study_tensorflow/keras_project/japanese_lang.txt") as japanese_file:
        japanese_lang = japanese_file.read()
        japanese_cat = len(japanese_lang)
    # 訓練データとテストデータに分割
    X_train, X_test, y_train, y_test = \
        cross_validation.train_test_split(X, Y)
    X_train = np.reshape(X_train, (len(X_train),  28, 28, 1))
    X_test  = np.reshape(X_test, (len(X_test), 28, 28, 1))
    print('X_train shape:', X_train.shape)
    # モデルを構築
    model = build_model(japanese_cat)
    model.summary()
    model.fit(X_train, y_train,
        batch_size=128, nb_epoch=10, verbose=1,
        validation_data=(X_test, y_test))
    # モデルを保存
    model.save_weights('/Users/j13-taniguchi/study_tensorflow/keras_project/japanese_dataset.hdf5')
    model_json = model.to_json()
    with open('/Users/j13-taniguchi/study_tensorflow/keras_project/japanese_dataset.json', 'w') as json_file:
        json.dump(model_json, json_file, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))
    
    # モデルを評価
    score = model.evaluate(X_test, y_test, verbose=0)
    print('score=', score)

def build_model(japanese_cat):
    # MLPのモデルを構築
    '''単純全結合モデル
    model = Sequential()
    model = Sequential()
    model.add(Dense(512, input_shape=(image_w * image_h,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(japanese_cat))
    model.add(Activation('softmax'))
    '''
    '''CNN
    model = Sequential()
    model.add(Convolution2D(32, 1, 3, border_mode='valid', input_shape=(0, 1, image_w * image_h)))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 1, 3, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(1, 2)))
    model.add(Dropout(0.25))
    
    model.add(Convolution2D(32, 1, 3, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 1, 3, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(1, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2195))
    model.add(Activation('softmax'))
    '''

    #VGG
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(image_w, image_h, 1)))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Convolution2D(32, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2195))
    model.add(Activation('softmax'))


    model.compile(loss='categorical_crossentropy',
        optimizer=RMSprop(),
        metrics=['accuracy'])
    return model

if __name__ == '__main__':
    main()
