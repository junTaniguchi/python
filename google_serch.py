#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 21:01:54 2017
@author: JunTaniguchi
"""

import requests
#import ssl
import urllib.request
from urllib.request import urlopen
from bs4 import BeautifulSoup

def get_serch_picture():
    """
    Beautiful Soupを利用してwebスクレイピングでgoogleから画像データを抽出する。
    -------------------------------------------------------------------------------
    text
      str型。検索する際のキーワードとして利用する。検索ワードを2つ以上使用したい場合は、各単語を+記号で繋ぐ。
      exmple) "東京+レストラン"
    """
    url = 'https://www.google.co.jp/search?q=%E4%BA%A4%E9%80%9A%E6%A1%88%E5%86%85%E6%A8%99%E8%AD%98&source=lnms&tbm=isch&sa=X&ved=0ahUKEwjPg9_hyo3SAhWMwbwKHfRSAQ8Q_AUICCgB&biw=1335&bih=760'

    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    soup = BeautifulSoup(urllib.request.urlopen(req).read(), "html.parser")
    
    soup_div = soup.find_all('div',{"class" : "rg_di rg_bx rg_el ivg-i"})
    count = 1
    for div in soup_div:
        print(count)
        if str(div).find('https') > 100:
            href = div.select('a[href^="http://"]')[0].get("href")
        elif str(div).find('https') <= 100:
            href = div.select('a[href^="https://"]')[0].get("href")
        else:
            break
        
        byte_streem = requests.get(href, verify=False).content

        f = open("XXXXXXXXXXXXXXXXXXX/testdata%s.png" % (str(count).zfill(4)), 'wb')
        f.write(byte_streem)
        f.close()
        count += 1


if __name__ == '__main__':
    
    get_serch_picture()
    print('取得できました。')
