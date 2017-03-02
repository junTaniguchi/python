# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 23:50:22 2017

@author: JunTaniguchi
"""
from gimpfu import *
# Python-Fu のサンプル・スクリプト
# GIMP の Python-Fu コンソールにコピペして実行してください

# 画像データの作成
## 指定したサイズで画像データを作成する
### width : 画像データの幅 (px)
### height : 画像データの高さ (px)
def create_image(width, height):
    # 画像データを生成
    return gimp.Image(width, height, RGB)

# レイヤーの追加
## 指定した名前のレイヤーを新規に作成し、画像データに挿入する
### image : レイヤーを追加する画像データ
### name : 新規に作成するレイヤーの名前（文字列）
def add_layer(image, name):
    # レイヤーの作成に必要なパラメータ
    width   = image.width
    height  = image.height
    type    = RGB_IMAGE
    opacity = 100
    mode    = NORMAL_MODE
    #
    # パラメータをもとにレイヤーを作成
    layer = gimp.Layer(image, name, width, height, type, opacity, mode)
    #
    # レイヤーを背景色で塗りつぶす（GIMP のデフォルトの挙動に合わせています）
    layer.fill(1)
    #
    # 画像データの 0 番目の位置にレイヤーを挿入する
    position = 0
    image.add_layer(layer, position)
    #
    return layer

# ペンシルツールで線を描く
## 配列に格納した座標列を結ぶ線を描画領域にペンシルツールで描く
### drawable : 描画領域（レイヤーなど）
### lines : 描画される線の座標列を格納した配列
def draw_pencil_lines(drawable, lines):
    # ペンシルツールで線を描画する
    pdb.gimp_pencil(drawable, len(lines), lines)

# ペンシルツールで矩形を描く
## 左上、右下座標をもとに描画領域に矩形を描く
### drawable : 描画領域（レイヤーなど）
### x1 : 左上の X 座標
### y1 : 左上の Y 座標
### x2 : 右下の X 座標
### y2 : 右下の Y 座標
def draw_rect(drawable, x1, y1, x2, y2):
    lines = [x1, y1, x2, y1, x2, y2, x1, y2, x1, y1]
    draw_pencil_lines(drawable, lines)

# エアブラシで線を描く
## 配列に格納した座標列を結ぶ線を描画領域にエアブラシで描く
### drawable : 描画領域（レイヤーなど）
### pressure : 筆圧 (0-100)
### lines : 描画される線の座標列を格納した配列
def draw_airbrush_lines(drawable, pressure, lines):
    # エアブラシで線を描画する
    pdb.gimp_airbrush(drawable, pressure, len(lines), lines)

# 文字列を描画する
## 指定した描画領域に文字列を描画します
### drawable : 描画領域（レイヤーなど）
### x : 文字列を描画する位置の X 座標
### y : 文字列を描画する位置の Y 座標
### size : フォントサイズ
### str : 描画する文字列
def draw_text(drawable, x, y, size, str):
    image = drawable.image
    border = -1
    antialias = True
    size_type = PIXELS
    fontname = '*'
    floating_sel = pdb.gimp_text_fontname(image, drawable, x, y, str, border,
                                          antialias, size, size_type, fontname)
    pdb.gimp_floating_sel_anchor(floating_sel)

# 描画する色を変更する
## パレットの前景色を変更して描画色を設定する
### r : 赤要素 (0-255)
### g : 緑要素 (0-255)
### b : 青要素 (0-255)
### a : 透明度 (0-1.0)
def set_color(r, g, b, a):
    color = (r, g, b, a)
    pdb.gimp_context_set_foreground(color)

# 描画する線の太さを変える
## ブラシのサイズを変更して線の太さを設定する
### width : 線の太さ
def set_line_width(width):
    pdb.gimp_context_set_brush_size(width)

# 画像の表示
## 新しいウィンドウを作成し、画像データを表示する
### image : 表示する画像データ
def display_image(image):
    gimp.Display(image)

def main():
    with open("/Users/JunTaniguchi/Desktop/pythonPlayGround/study_tensorflow/keras_project/japanese_lang/japanese_lang.txt", 'r') as japanese_lang:
        japanese_list = list(japanese_lang)
    
    # サンプル画像を出力するフォルダ
    for japanese_str in japanese_list:
        url = "/Users/JunTaniguchi/Desktop/pythonPlayGround/study_tensorflow/keras_project/image/" + japanese_str
        if not os.path.exists(url):
            os.makedires(url)
    
    image = create_image(640, 400)
    for japanese_str in japanese_list:
        url = "/Users/JunTaniguchi/Desktop/pythonPlayGround/study_tensorflow/keras_project/image/" + japanese_str
        layer = add_layer(image, japanese_str)
        draw_rect(layer, 390, 210, 490, 310)
        draw_text(layer, 200, 180, 20, japanese_str)
        lines = [110,90, 120,180, 130,110, 140,150]
        draw_airbrush_lines(layer, 75, lines)
        set_color(0,0,0,1.0)  # black
        set_line_width(1)
        draw_rect(layer, 420, 240, 520, 340)
        #display_image(image)
        with open(url + japanese_str + '001.png','bw') as f:
            f.write(image)

main()