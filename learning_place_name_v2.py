# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 12:44:16 2017
@author: j13-taniguchi
"""

import os, glob
import cv2
import keras
from keras.applications.imagenet_utils import preprocess_input
from keras.backend.tensorflow_backend import set_session
from keras.models import Model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import pickle
from random import shuffle
from scipy.misc import imread
from scipy.misc import imresize
import tensorflow as tf
from keras.utils.visualize_util import plot
import shutil


path = "/Users/JunTaniguchi/study_tensorflow/keras_project/read_place"
os.chdir(path)

from Generator import Generator
from ssd import SSD300
from ssd_training import MultiboxLoss
from ssd_utils import BBoxUtility

plt.rcParams['figure.figsize'] = (8, 8)
plt.rcParams['image.interpolation'] = 'nearest'

np.set_printoptions(suppress=True)

# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.9
# set_session(tf.Session(config=config))

#地名のリストを作成
with open("./param/place_tokyo.txt", "r") as place_file:
    place_list = place_file.readlines()
    place_list = [place_str.strip() for place_str in place_list]
NUM_CLASSES = len(place_list) + 1
input_shape = (300, 300, 3)

priors = pickle.load(open('./param/prior_boxes_ssd300.pkl', 'rb'))
bbox_util = BBoxUtility(NUM_CLASSES, priors)

# 存在するpickleファイル全てを取り込む
X = pickle.load(open('./param/place_name_X.pkl', 'rb'))
X /= 255
num_train = int(round(0.8 * len(X)))
X_train = X[:num_train]
X_test = X[num_train:]
num_test = len(X_test)


gt = pickle.load(open('./param/place_name_Y.pkl', 'rb'))
keys = sorted(gt.keys())
num_train = int(round(0.8 * len(keys)))
y_train = keys[:num_train]
y_test = keys[num_train:]

path_prefix = './image/'
gen = Generator(gt, bbox_util, 16, path_prefix,
                y_train, y_test,
                (input_shape[0], input_shape[1]), do_crop=False)

model = SSD300(input_shape, num_classes=NUM_CLASSES)
model.summary()
# モデルをpngでプロット
plot(model,
     to_file='./param/learning_place_name_v2.png', 
     show_shapes=True,
     show_layer_names=True)

#model.load_weights('weights_SSD300.hdf5', by_name=True)

freeze = ['input_1', 'conv1_1', 'conv1_2', 'pool1',
          'conv2_1', 'conv2_2', 'pool2',
          'conv3_1', 'conv3_2', 'conv3_3', 'pool3']#,
#           'conv4_1', 'conv4_2', 'conv4_3', 'pool4']

for L in model.layers:
    if L.name in freeze:
        L.trainable = False

def schedule(epoch, decay=0.9):
    return base_lr * decay**(epoch)

callbacks = [keras.callbacks.ModelCheckpoint('./param/checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                                             verbose=1,
                                             save_weights_only=True),
             keras.callbacks.LearningRateScheduler(schedule),
             keras.callbacks.EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto'),
             #keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True)
             ]

base_lr = 3e-4
optim = keras.optimizers.Adam(lr=base_lr)
# optim = keras.optimizers.RMSprop(lr=base_lr)
# optim = keras.optimizers.SGD(lr=base_lr, momentum=0.9, decay=decay, nesterov=True)
model.compile(optimizer=optim,
              loss=MultiboxLoss(NUM_CLASSES, neg_pos_ratio=2.0).compute_loss)

nb_epoch = 30
history = model.fit_generator(gen.generate(True),
                              gen.train_batches,
                              nb_epoch,
                              verbose=1,
                              callbacks=callbacks,
                              validation_data=gen.generate(False),
                              nb_val_samples=gen.val_batches,
                              nb_worker=1)

inputs = []
images = []
img_path = path_prefix + sorted(y_test)[0]
img = image.load_img(img_path, target_size=(300, 300))
img = image.img_to_array(img)
images.append(imread(img_path))
inputs.append(img.copy())
inputs = preprocess_input(np.array(inputs))

preds = model.predict(inputs, batch_size=1, verbose=1)
results = bbox_util.detection_out(preds)

# 学習履歴をプロット        
plot(model, to_file='./param/learning_place_name_v2.png')
# モデルを保存
model.save_weights('./param/learning_place_name_v2.hdf5')
# チェックポイントとなっていたファイルを削除
shutil.rmtree('./param/checkpoints')

# 重みパラメータをJSONフォーマットで出力
model_json = model.to_json()
with open('./param/learning_place_name_v2.json', 'w') as json_file:
    json_file.write(model_json)

print("finish!!")

for i, img in enumerate(images):
    # Parse the outputs.
    det_label = results[i][:, 0]
    det_conf = results[i][:, 1]
    det_xmin = results[i][:, 2]
    det_ymin = results[i][:, 3]
    det_xmax = results[i][:, 4]
    det_ymax = results[i][:, 5]

    # Get detections with confidence higher than 0.6.
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.6]

    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].tolist()
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]

    colors = plt.cm.hsv(np.linspace(0, 1, 4)).tolist()

    plt.imshow(img / 255.)
    currentAxis = plt.gca()

    for i in range(top_conf.shape[0]):
        xmin = int(round(top_xmin[i] * img.shape[1]))
        ymin = int(round(top_ymin[i] * img.shape[0]))
        xmax = int(round(top_xmax[i] * img.shape[1]))
        ymax = int(round(top_ymax[i] * img.shape[0]))
        score = top_conf[i]
        label = int(top_label_indices[i])
        #label_name = voc_classes[label - 1]
        display_txt = '{:0.2f}, {}'.format(score, label)
        coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
        color = colors[label]
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':0.5})
    
    plt.show()
