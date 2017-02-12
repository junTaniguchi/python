#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 09:13:00 2017

@author: JunTaniguchi
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 21:01:54 2017
@author: JunTaniguchi
"""

import requests

def get_flicker_dict(text):
    """
    flicker serch APIを利用し、テキスト検索でヒットした画像ファイルの情報を
    JSON形式で取得する。
      参照:https://www.flickr.com/services/api/flickr.photos.search.html
    
    text
      str型。検索する際のキーワードとして利用する。日本語はNG（大抵の場合ヒットしないから）。
    """
    dict_url = 'https://api.flickr.com/services/rest/'
    flicker_argument = {
        'method'         : 'flickr.photos.search',
        "api_key"        : "2f65318bba9d731e608f4afe5e880f8e",
        "text"           : text,
        'nojsoncallback' : 1,
        'format'         : 'json',
        "accuracy"       : 2,  # 1:世界、 2〜3:国、 4〜6:地域、 7〜11:都市、12〜16ストリート
        "content_type"   : 7,  # 1:写真のみ、2:スクリーンショットのみ,3:その他のみ,7:全検索
        "geo_context"    : 0,  # 0:全て,1:屋内のみ,2:鴎外のみ
        "per_page"       : 100 # 1ページに返す写真の数。デフォルトは100。最大値は500です。
    }
    #argument_q = urllib.parse.urlencode(flicker_argument)
    FLICKER_json_req = requests.get(dict_url, params=flicker_argument)
    FLICKER_dict= FLICKER_json_req.json()
    
    return FLICKER_dict


def get_flicker_picture(FLICKER_dict):
    """
    flicker serch APIを利用し、テキスト検索でヒットした画像ファイルの情報を
    JSON形式で取得する。
    https://www.flickr.com/services/api/flickr.photos.search.html
    text
      str型。検索する際のキーワードとして利用する。
    """
    pct_url = 'https://farm%s.staticflickr.com/%s/%s_%s.jpg'
    count = 1
    for i in FLICKER_dict['photos']['photo']:
        img_url = pct_url % (i['farm'],i['server'],i['id'],i['secret'])
        FLICKER_img_req = requests.get(img_url)
        f = open("/Users/JunTaniguchi/Desktop/pythonPlayGround/study_tensorflow/API/testdata%s.png" % (str(count).zfill(4)), 'wb')
        f.write(FLICKER_img_req.content)
        f.close()
        count += 1


if __name__ == '__main__':
    i = 0
    text = "tokyo poster"
    # flickerで画像データを検索し、結果をJSON形式で返す。
    FLICKER_dict = get_flicker_dict(text)
    # JSON形式のflicker検索結果を元に全画像データを出力する。
    get_flicker_picture(FLICKER_dict)
    print('取得できました。')