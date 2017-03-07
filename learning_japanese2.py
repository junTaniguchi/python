# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 16:45:08 2017
@author: j13-taniguchi
"""
import os
from keras.models import Sequential
import tensorflow as tf
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from sklearn import cross_validation
import numpy as np
import keras.callbacks
import keras.backend.tensorflow_backend as KTF
import json

image_w = 28
image_h = 28
log_filepath = './log'
path = "/Users/Juntaniguchi/study_tensorflow/keras_project"

os.chdir(path)

# フォント画像のデータを読む
xy = np.load("japanese_lang.npz")
X = xy["x"]
Y = xy["y"]
# データを正規化
#X = X.reshape(X.shape[0], image_w * image_h).astype('float32')
X /= 255
# ラベル作成
Y = np_utils.to_categorical(Y, 2195)
with open("japanese_lang.txt") as japanese_file:
    japanese_lang = japanese_file.read()
    japanese_cat = len(japanese_lang)
# 訓練データとテストデータに分割
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y)
X_train = np.reshape(X_train, (len(X_train),  28, 28, 1))
X_test  = np.reshape(X_test, (len(X_test), 28, 28, 1))
print('X_train shape:', X_train.shape)

#VGG
old_session = KTF.get_session()

with tf.Graph().as_default():
    session = tf.Session('')
    KTF.set_session(session)
    KTF.set_learning_phase(1)
    # モデルを構築
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
    model.summary()
    
    # callback関数にて収束判定を追加    
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')
    # callback関数にてTensorboardを可視化
    tb_cb = keras.callbacks.TensorBoard(log_dir=log_filepath,
                                        histogram_freq=1,
                                        write_graph=True)
    cbks = [tb_cb]
    # 学習開始
    history = model.fit(X_train, y_train,
                        batch_size=128,
                        nb_epoch=1,
                        verbose=1,
                        validation_data=(X_test, y_test))
    # モデルを保存
    model.save_weights('japanese_dataset.hdf5')
    model_json = model.to_json()
    with open('./japanese_dataset.json', 'w') as json_file:
        json.dump(model_json, json_file, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))
    
    # モデルを評価
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy;', score[1])

KTF.set_session(old_session)
