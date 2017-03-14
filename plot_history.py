# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 00:02:52 2017
@author: JunTaniguchi
"""
import os
path = "/Users/j13-taniguchi/study_tensorflow/keras_project/read_place"
os.chdir(path)

def plot_history(i, history):
    import matplotlib.pyplot as plt

    # 新規のウィンドウを描画
    fig = plt.figure()
    # サブプロットを追加
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)
    # 精度の履歴をプロット
    ax1.plot(history.history['acc'],"o-",label="accuracy")
    ax1.plot(history.history['val_acc'],"o-",label="val_acc")
    ax1.title('model accuracy')
    ax1.xlabel('epoch')
    ax1.ylabel('accuracy')
    ax1.legend(loc="lower right")
    ax1.show()
    # 損失の履歴をプロット
    ax2.plot(history.history['loss'],"o-",label="loss",)
    ax2.plot(history.history['val_loss'],"o-",label="val_loss")
    ax2.title('model loss')
    ax2.xlabel('epoch')
    ax2.ylabel('loss')
    ax2.legend(loc='lower right')
    plt.show()
    filename = "./plot/Learning_history_No%s.png" % str(i+1)
    plt.savefig(filename)
