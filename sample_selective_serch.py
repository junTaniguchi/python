# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 23:08:52 2017

@author: JunTaniguchi
"""
import os
import selectivesearch
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

os.chdir('/Users/JunTaniguchi/study_tensorflow/keras_project/read_place/project_rcnn/selectivesearch')

img = cv2.imread("hrei-sign105.png")
img_lbl, regions = selectivesearch.selective_search(img,
                                                    scale=500,
                                                    sigma=0.9,
                                                    min_size=10)

candidates = set()

for r in regions:
    # excluding same rectangle (with different segments)
    if r['rect'] in candidates:
        continue
    # excluding regions smaller than 2000 pixels
    if r['size'] < 500:
        continue
    # distorted rects
    x, y, w, h = r['rect']
    if w / h > 1.2 or h / w > 1.2:
        continue
    candidates.add(r['rect'])

# draw rectangles on the original image
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
ax.imshow(img)
for x, y, w, h in candidates:
    #print (x, y, w, h)
    rect = mpatches.Rectangle(
        (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
    ax.add_patch(rect)
plt.show()
