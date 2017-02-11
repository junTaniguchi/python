#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 21:01:54 2017

@author: JunTaniguchi
"""

import urllib.request, json
#from PIL import Image
#import matplotlib.pyplot as plt
#import numpy as np


def get_location(location, query):
    """
    google Place APIの機能を利用して検索ワード（query）の存在する場所の緯度経度のリストを取得する。
    location
      空のリスト
    query
      str型。検索ワード。
    """
    google_places_params = {
        "radius"   : 50000,
        "query"    : query,
        "key"      : "XXXX", #google APIsの認証キー
        "language" : "ja"
    }
    places_q = urllib.parse.urlencode(google_places_params)

    GOOGLE_PLACES_URL = "https://maps.googleapis.com/maps/api/place/textsearch/json?" + places_q
    place_html = urllib.request.urlopen(GOOGLE_PLACES_URL).read().decode("utf-8")        
    for j in range(0, len(json.loads(place_html)['results'])):
        lat = json.loads(place_html)['results'][j]['geometry']['location']['lat']
        lng = json.loads(place_html)['results'][j]['geometry']['location']['lng']
        location.append("%sx%s" % (str(lat), str(lng)))
    
    return location

def get_picture(location, i):
    """
    google map street view APIを利用し、locationの撮影場所を元に画像ファイルを取得する。
    location
      str型。"緯度x経度"のフォーマットのみを引数として認める。
    i
      int型。出力するファイル名の連番部分。
    """
    streetview_params = {
        "size"     : "600x600",
        "location" : location,
        "heading"  : 88,
        "pitch"    : 7,
        "fov"      : 0,
        "zoom"     : 3
    }
    streetview_q = urllib.parse.urlencode(streetview_params)
    
    STREET_VIEW_IMAGE_URL = 'https://maps.googleapis.com/maps/api/streetview?' + streetview_q
    streetview_html = urllib.request.urlopen(STREET_VIEW_IMAGE_URL).read() #.decode("utf-8")

    #画像の読み込み  
    f = open("/Users/JunTaniguchi/Desktop/pythonPlayGround/study_tensorflow/API/testdata%s.png" % (str(i + 1).zfill(4)), 'wb')
    f.write(streetview_html)
    f.close()


if __name__ == '__main__':
    i = 0
    location_list = []
    query = "看板"
    picture_locations = get_location(location_list, query)
    for picture_location in picture_locations:
        get_picture(picture_location, i)
        i += 1
    print('合計で％d件を取得できました。' % (i+1))
