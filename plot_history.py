# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 00:02:52 2017

@author: JunTaniguchi
"""
import matplotlib.pyplot as plt
import os
path = "/Users/JunTaniguchi/study_tensorflow/keras_project/read_place"
os.chdir(path)

def plot_history(history_list):
    # print(history.history.keys())
    plt.figure
    for idx, history in enumerate(history_list):
        plt.subplot(idx,1,1)
        # 精度の履歴をプロット
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend(['acc', 'val_acc'], loc='lower right')
        # 損失の履歴をプロット
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['loss', 'val_loss'], loc='lower right')
    filename = "./plot/history_plot.png"
    plt.savefig(filename)