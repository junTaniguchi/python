# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 16:16:06 2017

@author: j13-taniguchi
"""

import tesseract

api = tesseract.TessBaseAPI()
api.Init(".", "jpn", tesseract.OEM_DEFAULT)
api.SetPageSegMode(tesseract.PSM_AUTO)

#対象ファイルの読み込み
image_file = "/Users/j13-taniguchi/study_tensorflow/keras_project/image/J/n-MSGOTHIC.TTF-j--8.png"
image_pic  = open(image_file, "rb").read()

image_str = tesseract.ProcessPagesBuffer(image_pic, len(image_pic), api)
print(image_str)