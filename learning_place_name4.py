# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 16:45:08 2017
@author: j13-taniguchi
"""
import os, glob
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
from keras.utils.visualize_util import plot
from PIL import Image, ImageDraw, ImageFont
import shutil
import cv2

path = "/Users/JunTaniguchi/study_tensorflow/keras_project/read_place"
os.chdir(path)

log_filepath = './log'

from ssd import SSD300
from use_spp_net_model import use_spp_net_model
from plot_history import plot_history

#地名のリストを作成
with open("./param/place_tokyo.txt", "r") as place_file:
    place_list = place_file.readlines()
    place_list = [place_str.strip() for place_str in place_list]
NUM_CLASSES = len(place_list)

# 存在するnpzファイル全てを取り込む
npz_list = glob.glob("./param/npz/*.npz") # Mac
xy = []
X = []
Y = []
cat_list = []
# フォント画像のデータを読む
for no, npz in enumerate(npz_list):
    xy.append(np.load(npz))
    X.append(xy[no]["x"])
    Y.append(xy[no]["y"])
    cat_list.append(xy[no]["y"][:,0])
    Y_rect = cat_list
    X[no] /= 255
    cat_list[no] = np_utils.to_categorical(cat_list[no], NUM_CLASSES)

X_train = []
y_train = []
X_test = []
y_test = []
for i in range(len(xy)):
    # 訓練データとテストデータに分割
    X_train_i, X_test_i, y_train_i, y_test_i = cross_validation.train_test_split(X[i], Y[i])
    X_train.append(X_train_i)
    y_train.append(y_train_i)
    X_test.append(X_test_i)
    y_test.append(y_test_i)

#VGG
#old_session = KTF.get_session()


with tf.Graph().as_default():
    '''
    session = tf.Session('')
    KTF.set_session(session)
    KTF.set_learning_phase(1)

    def schedule(epoch, decay=0.9):
        return base_lr * decay**(epoch)
    base_lr = 3e-4
    optim = keras.optimizers.Adam(lr=base_lr)

    # モデルを構築
    model = SSD300(input_shape=(300, 300, 3), num_classes=NUM_CLASSES + 1)
    model.compile(loss='categorical_crossentropy',
        optimizer=optim,
        metrics=['accuracy'])
    model.summary()
    
    # callback関数にて下記機能を追加
    #    重みパラメータの中間セーブ
    #    学習率のスケジューラ
    #    改善率が低い場合にトレーニングを終了する
    #    TensorBoardの使用
    callbacks = [
                 keras.callbacks.ModelCheckpoint('./param/checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5', verbose=1, save_weights_only=True),
                 keras.callbacks.LearningRateScheduler(schedule),
                 keras.callbacks.EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto'),
                 #keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True)
                ]
    # 学習開始
    history = model.fit(X_train, y_train,
                        batch_size=128,
                        nb_epoch=1,
                        verbose=1,
                        validation_data=(X_test, y_test))
    '''
    session = tf.Session('')
    KTF.set_session(session)
    KTF.set_learning_phase(1)

    def schedule(epoch, decay=0.9):
        return base_lr * decay**(epoch)
    
    base_lr = 3e-4
    optim = keras.optimizers.Adam(lr=3e-4)

    # モデルを構築(SPPNet)
    model = use_spp_net_model(input_shape=(None, None, 3),
                              NUM_CLASSES=NUM_CLASSES,
                              optim=optim)
    model.summary()
    # 中間チェックポイントのデータを一時保存するためのディレクトリを作成
    if not os.path.exists('./param/checkpoints'):
        os.mkdir('./param/checkpoints')
    # callback関数にて下記機能を追加
    #    重みパラメータの中間セーブ
    #    学習率のスケジューラ
    #    改善率が低い場合にトレーニングを終了する
    #    TensorBoardの使用 $tensorboard --logdir=/full_path_to_your_logs

    callbacks = [
                 keras.callbacks.ModelCheckpoint('./param/checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5', verbose=1, save_weights_only=True),
                 keras.callbacks.LearningRateScheduler(schedule),
                 keras.callbacks.EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto'),
                 #keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True)
                ]
        
    if os.path.exists('learning_place_name.hdf5'):
        model.load_weights('learning_place_name.hdf5', by_name=True)
    # 学習開始
    result_X_test_list = []
    history_list = []
    for i in range(len(xy)):
        history = model.fit(X_train[i], y_train[i],
                            batch_size=128,
                            nb_epoch=1,
                            verbose=1,
                            callbacks=callbacks,
                            validation_data=(X_test[i], y_test[i]))
        result_X_test_list.append(model.predict(X_test[i].astype(np.float32)))
        history_list.append(history)
        print("Learning No.%s is ended" % str(i))
    # 学習履歴をプロット        
    plot_history(history_list)
    plot(model, to_file='./param/learning_place_name.png')
    # モデルを保存
    model.save_weights('./param/learning_place_name.hdf5')
    # チェックポイントとなっていたファイルを削除
    shutil.rmtree('./param/checkpoints')
    
    # 重みパラメータをJSONフォーマットで出力
    model_json = model.to_json()
    with open('./param/learning_place_name.json', 'w') as json_file:
        #json.dump(model_json, json_file)
        json_file.write(model_json)
    
    # モデルを評価
    score = []
    for i in range(len(xy)):
        score = model.evaluate(X_test[i], y_test[i], verbose=0)
        print('Learning No.%s' %str(i))
        print('  Test score:', score[0])
        print('  Test accuracy;', score[1])

#KTF.set_session(old_session)
print("finish!!")
correct_count = 0
incorrect_count = 0
for idx1, result_X_test in enumerate(result_X_test_list):
    # 予測結果と正解のラベルを照合する
    for idx2, idx_result_X in enumerate(result_X_test):
        # 予測結果のargmaxを抽出    
        result_idx = idx_result_X.argmax()
        result_label = place_list[result_idx]
        # 正解の番号を抽出
        answer_idx = y_test[idx1][idx2].argmax()
        answer_label = place_list[answer_idx]
        # 予測結果と正解の値を比較
        if result_idx == answer_idx:
            correct_message = "correct Awesome: answer: %s result: %s" %(answer_label, result_label)    
            print(correct_message)
            correct_count+=1
            correct_flag = True
            
        # 不正解をコンソールへ表示   
        error_message = "incorrect: answer: %s result: %s" %(answer_label, result_label)    
        print(error_message)
        incorrect_count+=1
        
        # 不正解の画像をディレクトリへ格納する準備
        if correct_flag:
            output_dir = "./correct/%s/" %(answer_label)
        else:
            output_dir = "./incorrect/%s/" %(answer_label)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        incorrect_file_name = output_dir + answer_label + str(idx2) + ".png"
        # 不正解だったデータを画像化
        # idx_result_Xを非正規化
        X_test[idx1][idx2] *= 256
        X_img_array = X_test[idx1][idx2]
        x, y, w, h = xy[1][result_idx][1:5]
        img = cv2.rectangle(X_img_array, (x, y), (w, h), (0, 0, 255), 10)
        img = Image.fromarray(np.uint8(img))
        img.save(incorrect_file_name) 
print("correct_count   :%s" % str(correct_count))
print("incorrect_count :%s" % str(incorrect_count))