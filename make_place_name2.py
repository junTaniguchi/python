# -*- encoding:utf8 -*-
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np

path = "/Users/j13-taniguchi/study_tensorflow/keras_project/read_place"
os.chdir(path)

#フォンントのリストを作成
with open("./param/japanese_font_list.txt", "r") as japanese_ttf:
    jp_font_list = japanese_ttf.readlines()
    jp_font_list = [jp_font.strip() for jp_font in jp_font_list]

#地名のリストを作成
with open("./param/place_tokyo.txt", "r") as place_file:
    place_list = place_file.readlines()
    place_list = [place_str.strip() for place_str in place_list]

# サンプル画像を出力するフォルダ
for place_name in place_list:
    url = "./image/" + place_name
    if not os.path.exists(url):
        os.makedirs(url)

X = []
Y = []
for idx1, jp_font in enumerate(jp_font_list):
    # フォントの指定。引数は順に「フォントのパス」「フォントのサイズ」「エンコード」
    # メソッド名からも分かるようにTure Typeのフォントを指定する
    font = ImageFont.truetype(jp_font, 20, encoding='unic')

    for idx2, place_name in enumerate(place_list):
        # 画像のピクセルを指定
        image = Image.new('RGB', (300, 300), "green")
        draw = ImageDraw.Draw(image)
        # 日本語の文字を入れてみる
        # 引数は順に「(文字列の左上のx座標, 文字列の左上のy座標)」「フォントの指定」「文字色」
        draw.text((25, 45), place_name, font=font, fill='#ffffff')
        # 画像ファイルを保存する
        image.save("./image/" + place_name + "/" + str(idx1) + "_upperleft_" + place_name + '.png', 'PNG')
        data = np.asarray(image)        
        X.append(data)
        Y.append(idx2)

for idx1, jp_font in enumerate(jp_font_list):
    # フォントの指定。引数は順に「フォントのパス」「フォントのサイズ」「エンコード」
    # メソッド名からも分かるようにTure Typeのフォントを指定する
    font = ImageFont.truetype(jp_font, 30, encoding='unic')

    for idx2, place_name in enumerate(place_list):
        # 画像のピクセルを指定
        image = Image.new('RGB', (300, 300), "blue")
        draw = ImageDraw.Draw(image)
        # 日本語の文字を入れてみる
        # 引数は順に「(文字列の左上のx座標, 文字列の左上のy座標)」「フォントの指定」「文字色」
        draw.text((170, 150), place_name, font=font, fill='#ffffff')
        # 画像ファイルを保存する
        image.save("./image/" + place_name + "/" + str(idx1) + "_lowerright_" + place_name + '.png', 'PNG')
        data = np.asarray(image)        
        X.append(data)
        Y.append(idx2)


X = np.array(X)
Y = np.array(Y)
np.savez("./param/place_name.npz", x=X, y=Y)
print("ok,", len(Y))
