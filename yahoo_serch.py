#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 21:01:54 2017
@author: JunTaniguchi
"""

import requests

from urllib.request import urlopen
from bs4 import BeautifulSoup

def get_serch_picture():
    """
    Beautiful Soupを利用してwebスクレイピングでyahooから画像データを抽出する。
    -------------------------------------------------------------------------------

    """
    url = 'http://image.search.yahoo.co.jp/search;_ylt=A2RCL6rBwaFYkQIAAWGU3uV7?p=%E4%BA%A4%E9%80%9A%E6%A1%88%E5%86%85%E6%A8%99%E8%AD%98&aq=-1&oq=&ei=UTF-8'
    html = urlopen(url)
    soup = BeautifulSoup(html.read(), "html.parser")
    
    soup_div = soup.find_all('div',{"class" : "SeR"})
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

        f = open("XXXXXXXXXX/testdata%s.png" % (str(count).zfill(4)), 'wb')
        f.write(byte_streem)
        f.close()
        count += 1


if __name__ == '__main__':
    
    get_serch_picture()
    print('取得できました。')
