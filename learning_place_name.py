# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 16:45:08 2017
@author: j13-taniguchi
"""
import os
import tensorflow as tf
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from sklearn import cross_validation
import numpy as np
import keras.callbacks
import keras.backend.tensorflow_backend as KTF
from keras.utils.visualize_util import plot
import json

image_w = 300
image_h = 300
log_filepath = './log'

path = "/Users/j13-taniguchi/study_tensorflow/keras_project/read_place"
os.chdir(path)

from ssd import SSD300

# フォント画像のデータを読む
xy = np.load("./param/place_name.npz")
X = xy["x"]
Y = xy["y"]
# データを正規化
X = X.reshape(X.shape[0], X.shape[1], X.shape[2], X.shape[3]).astype('float32')
X /= 255

#地名のリストを作成
with open("./param/place_tokyo.txt", "r") as place_file:
    place_list = place_file.readlines()
    place_list = [place_str.strip() for place_str in place_list]
    NUM_CLASSES = len(place_list)

Y = np_utils.to_categorical(Y, NUM_CLASSES)

# 訓練データとテストデータに分割
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y)
X_train = np.reshape(X_train, (len(X_train),  300, 300, 3))
X_test  = np.reshape(X_test, (len(X_test), 300, 300, 3))
print('X_train shape:', X_train.shape)

#VGG
old_session = KTF.get_session()

with tf.Graph().as_default():
    session = tf.Session('')
    KTF.set_session(session)
    KTF.set_learning_phase(1)
    # モデルを構築
    model = SSD300(input_shape=(300, 300, 3), num_classes=NUM_CLASSES)
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
    plot(model, to_file='learning_japanese2.png')
    # モデルを保存
    model.save_weights('japanese_dataset.hdf5')
    model_json = model.to_json()
    with open('japanese_dataset.json', 'w') as json_file:
        json.dump(model_json, json_file, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))
    
    # モデルを評価
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy;', score[1])

KTF.set_session(old_session)
