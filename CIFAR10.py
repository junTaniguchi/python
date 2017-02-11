#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 19:22:24 2017

@author: JunTaniguchi
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

np.random.seed(20160704)
tf.set_random_seed(20160704)

#データファイルから画像イメージとラベルデータを読み取る関数を用意する。
def read_cifar10(filename_queue):
    class CIFAR10Record(object):
        pass
    
    result = CIFAR10Record()
    # 各画像部品
    label_bytes = 1
    result.height = 32
    result.width = 32
    result.depth = 3
    image_bytes = result.height * result.width * result.depth
    record_bytes = label_bytes + image_bytes
    # ファイルから固定長レコードを出力するリーダー
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    # リーダーが生成した次のレコード（キー、値のペア）を返す
    result.key, value = reader.read(filename_queue)
    # 文字列のバイトを数字のベクトルとして再解釈する。
    record_bytes = tf.decode_raw(value, tf.uint8)
    # tf.slice(input_, begin, size, name=None)
    # input_のbeginから指定されたsize分のテンソルを抽出する。
    result.label = tf.cast(tf.slice(record_bytes, [0], [label_bytes]), tf.int32)
    depth_major = tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]),
                             [result.depth, result.height, result.width])
    # Convert from [depth, height, width] to [height, width, depth].
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])

    return result

#それぞれのラベルについて、８個ずつの画像イメージを表示する。
sess = tf.InteractiveSession()
filename = '/Users/JunTaniguchi/Desktop/pythonPlayGround/study_tensorflow/cifar10_data/cifar-10-batches-bin/test_batch.bin'
# tf.FIFOQueue(capacity, dtypes, shapes)
q = tf.FIFOQueue(99, [tf.string], shapes=())
q.enqueue([filename]).run(session=sess)
q.close().run(session=sess)
result = read_cifar10(q)

samples = [[] for l in range(10)]
while(True):
    label, image = sess.run([result.label, result.uint8image])
    label = label[0]
    if len(samples[label]) < 8:
        samples[label].append(image)
    if all([len(samples[l]) >= 8 for l in range(10)]):
        break
        
fig = plt.figure(figsize=(8,10))
for l in range(10):
    for c in range(8):
        subplot = fig.add_subplot(10, 8, l*8+c+1)
        subplot.set_xticks([])
        subplot.set_yticks([])
        image = samples[l][c]
        subplot.imshow(image.astype(np.uint8))
        
sess.close()

#前処理を施した画像イメージを生成する関数を用意する。
def distorted_samples(image):

    reshaped_image = tf.cast(image, tf.float32)
    width, height = 24, 24
    float_images = []

    resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                           width, height)
    float_image = tf.image.per_image_whitening(resized_image)
    float_images.append(float_image)

    for _ in range(6):
        distorted_image = tf.random_crop(reshaped_image, [height, width, 3])
        distorted_image = tf.image.random_flip_left_right(distorted_image)
        distorted_image = tf.image.random_brightness(distorted_image,
                                                     max_delta=63)
        distorted_image = tf.image.random_contrast(distorted_image,
                                                   lower=0.2, upper=1.8)
        float_image = tf.image.per_image_whitening(distorted_image)
        float_images.append(float_image)

    return tf.concat(0,float_images)

#それぞれのラベルについて、オリジナル、及び、前処理を施した画像イメージを表示する。
sess = tf.InteractiveSession()
filename = '/Users/JunTaniguchi/Desktop/pythonPlayGround/study_tensorflow/cifar10_data/cifar-10-batches-bin/test_batch.bin'
q = tf.FIFOQueue(99, [tf.string], shapes=())
q.enqueue([filename]).run(session=sess)
q.close().run(session=sess)
result = read_cifar10(q)

fig = plt.figure(figsize=(8,10))
c = 0
original = {}
modified = {}

while len(original.keys()) < 10:
    label, orig, dists = sess.run([result.label,
                                   result.uint8image,
                                   distorted_samples(result.uint8image)])
    label = label[0]
    if not label in original.keys():
        original[label] = orig
        modified[label] = dists

for l in range(10):
    orig, dists = original[l], modified[l]
    c += 1
    subplot = fig.add_subplot(10, 8, c)
    subplot.set_xticks([])
    subplot.set_yticks([])
    subplot.imshow(orig.astype(np.uint8))

    for i in range(7):
        c += 1
        subplot = fig.add_subplot(10, 8, c)
        subplot.set_xticks([])
        subplot.set_yticks([])
        pos = i*24
        image = dists[pos:pos+24]*40+120
        subplot.imshow(image.astype(np.uint8))
        
sess.close()
