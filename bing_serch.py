#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 21:01:54 2017
@author: JunTaniguchi
"""

import requests
from urllib.request import urlopen
from bs4 import BeautifulSoup

def get_html_dict(text):
    """
    flicker serch APIを利用し、テキスト検索でヒットした画像ファイルの情報をJSON形式で取得する。
      参照:https://www.flickr.com/services/api/flickr.photos.search.html
    -------------------------------------------------------------------------------
    text
      str型。検索する際のキーワードとして利用する。検索ワードを2つ以上使用したい場合は、各単語を+記号で繋ぐ。
      exmple) "東京+レストラン"
    """
    url = 'https://www.bing.com/images/search?q=%E9%81%93%E8%B7%AF%E6%A1%88%E5%86%85%E6%A8%99%E8%AD%98&qs=n&form=QBLH&scope=images&sp=-1&pq=%E9%81%93%E8%B7%AF%E6%A1%88%E5%86%85%E6%A8%99%E8%AD%98&sc=8-6&sk=&cvid=69379591511D45E2A24EF402B64656E3'
    html = urlopen(url)
    html_tag = BeautifulSoup(html.read(), "html.parser")
    script_tag = html_tag.find_all("script")
          html_tag.find_all("a", attrs={"class": "link", "href": "/link"})
    
    
    return FLICKER_dict


def get_flicker_picture(FLICKER_dict):
    """
    flicker serch APIを利用し、テキスト検索でヒットした画像ファイルの情報をJSON形式で取得する。
    その他の引数については下記参照。
    　　https://www.flickr.com/services/api/flickr.photos.search.html
    -------------------------------------------------------------------------------
    text
      str型。検索する際のキーワードとして利用する。
    """
    pct_url = 'https://farm%s.staticflickr.com/%s/%s_%s.jpg'
    count = 1
    for i in FLICKER_dict['photos']['photo']:
        img_url = pct_url % (i['farm'],i['server'],i['id'],i['secret'])
        FLICKER_img_req = requests.get(img_url)
        f = open("/Users/j13-taniguchi/study_tensorflow/pythonAPI/test_data/testdata%s.png" % (str(count).zfill(4)), 'wb')
        f.write(FLICKER_img_req.content)
        f.close()
        count += 1


if __name__ == '__main__':
    i = 0
    text = "日本語s"
    # flickerで画像データを検索し、結果をJSON形式で返す。
    FLICKER_dict = get_flicker_dict(text)
    # JSON形式のflicker検索結果を元に全画像データを出力する。
    get_flicker_picture(FLICKER_dict)
    print('取得できました。')
