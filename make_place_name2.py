# -*- encoding:utf8 -*-
import os
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

path = "/Users/JunTaniguchi/study_tensorflow/keras_project/read_place"
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
# １、フォントの標準文字
for idx1, jp_font in enumerate(jp_font_list):
    # フォントの指定。引数は順に「フォントのパス」「フォントのサイズ」「エンコード」
    # メソッド名からも分かるようにTure Typeのフォントを指定する
    font = ImageFont.truetype(jp_font, 20)

    for idx2, place_name in enumerate(place_list):
        # 画像のピクセルを指定
        image = Image.new('RGB', (300, 300), 'blue')
        draw = ImageDraw.Draw(image)
        # 日本語の文字を入れてみる
        # 引数は順に「(文字列の左上のx座標, 文字列の左上のy座標)」「フォントの指定」「文字色」
        draw.text((25, 45), place_name, font=font, fill='#ffffff')
        # 画像ファイルを保存する
        image.save("./image/" + place_name + "/" + str(idx1) + "_" + place_name + '.png', 'PNG')
        data = np.asarray(image)        
        X.append(data.astype(np.float64))
        Y.append(idx2)

# ２、ガウシアンフィルタでぼかす
for idx1, jp_font in enumerate(jp_font_list):
    # フォントの指定。引数は順に「フォントのパス」「フォントのサイズ」「エンコード」
    # メソッド名からも分かるようにTure Typeのフォントを指定する
    font = ImageFont.truetype(jp_font, 20)

    for idx2, place_name in enumerate(place_list):
        # 画像のピクセルを指定
        image = Image.new('RGB', (300, 300), 'blue')
        draw = ImageDraw.Draw(image)
        # 日本語の文字を入れてみる
        # 引数は順に「(文字列の左上のx座標, 文字列の左上のy座標)」「フォントの指定」「文字色」
        draw.text((25, 45), place_name, font=font, fill='#ffffff')
        # opencvのガウシアンフィルターを適応
        image_np = np.asarray(image)
        blur = cv2.GaussianBlur(image_np, (5, 5), 0)
        blur_image = Image.fromarray(blur)
        
        # 画像ファイルを保存する
        image.save("./image/" + place_name + "/" + str(idx1) + "_GaussianBlur_" + place_name + '.png', 'PNG')
        data = np.asarray(blur_image)       
        X.append(data.astype(np.float64))
        Y.append(idx2)

# ３、ガウシアンフィルタでぼかしつつ、アフィン変換で画面を回転させる
for idx1, jp_font in enumerate(jp_font_list):
    # フォントの指定。引数は順に「フォントのパス」「フォントのサイズ」「エンコード」
    # メソッド名からも分かるようにTure Typeのフォントを指定する
    font = ImageFont.truetype(jp_font, 20)

    for idx2, place_name in enumerate(place_list):
        # 画像のピクセルを指定
        image = Image.new('RGB', (300, 300), 'blue')
        draw = ImageDraw.Draw(image)
        # 日本語の文字を入れてみる
        # 引数は順に「(文字列の左上のx座標, 文字列の左上のy座標)」「フォントの指定」「文字色」
        draw.text((25, 45), place_name, font=font, fill='#ffffff')
        # opencvのガウシアンフィルターを適応
        image_np = np.asarray(image)
        blur = cv2.GaussianBlur(image_np, (5, 5), 0)
        # 画像の中心位置
        # 今回は画像サイズの中心をとっている
        center = tuple(np.array([blur.shape[1] * 0.5, blur.shape[0] * 0.5]))
        # 画像サイズの取得(横, 縦)
        size = tuple(np.array([blur.shape[1], blur.shape[0]]))
        # 回転させたい角度
        # ラジアンではなく角度(°)
        angle = 10.0
        # 拡大比率
        scale = 1.0
        # 回転変換行列の算出
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
        # アフィン変換
        blur_rot = cv2.warpAffine(blur, rotation_matrix, size, flags=cv2.INTER_CUBIC)
        affine_image = Image.fromarray(blur_rot)

        # 画像ファイルを保存する
        affine_image.save("./image/" + place_name + "/" + str(idx1) + "_GaussianBlur_Affine1_" + place_name + '.png', 'PNG')
        data = np.asarray(affine_image)        
        X.append(data.astype(np.float64))
        Y.append(idx2)

# ４、ガウシアンフィルタでぼかしつつ、アフィン変換で画面を回転させる
for idx1, jp_font in enumerate(jp_font_list):
    # フォントの指定。引数は順に「フォントのパス」「フォントのサイズ」「エンコード」
    # メソッド名からも分かるようにTure Typeのフォントを指定する
    font = ImageFont.truetype(jp_font, 20)

    for idx2, place_name in enumerate(place_list):
        # 画像のピクセルを指定
        image = Image.new('RGB', (300, 300), 'blue')
        draw = ImageDraw.Draw(image)
        # 日本語の文字を入れてみる
        # 引数は順に「(文字列の左上のx座標, 文字列の左上のy座標)」「フォントの指定」「文字色」
        draw.text((25, 45), place_name, font=font, fill='#ffffff')
        # opencvのガウシアンフィルターを適応
        image_np = np.asarray(image)
        blur = cv2.GaussianBlur(image_np, (5, 5), 0)
        # 画像の中心位置
        # 今回は画像サイズの中心をとっている
        center = tuple(np.array([blur.shape[1] * 0.5, blur.shape[0] * 0.5]))
        # 画像サイズの取得(横, 縦)
        size = tuple(np.array([blur.shape[1], blur.shape[0]]))
        # 回転させたい角度
        # ラジアンではなく角度(°)
        angle = 15.0
        # 拡大比率
        scale = 1.0
        # 回転変換行列の算出
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
        # アフィン変換
        blur_rot = cv2.warpAffine(blur, rotation_matrix, size, flags=cv2.INTER_CUBIC)
        affine_image = Image.fromarray(blur_rot)

        # 画像ファイルを保存する
        affine_image.save("./image/" + place_name + "/" + str(idx1) + "_GaussianBlur_Affine2_" + place_name + '.png', 'PNG')
        data = np.asarray(affine_image)        
        X.append(data.astype(np.float64))
        Y.append(idx2)

# ４、ガウシアンフィルタでぼかしつつ、アフィン変換で画面を回転させる
for idx1, jp_font in enumerate(jp_font_list):
    # フォントの指定。引数は順に「フォントのパス」「フォントのサイズ」「エンコード」
    # メソッド名からも分かるようにTure Typeのフォントを指定する
    font = ImageFont.truetype(jp_font, 20)

    for idx2, place_name in enumerate(place_list):
        # 画像のピクセルを指定
        image = Image.new('RGB', (300, 300), 'blue')
        draw = ImageDraw.Draw(image)
        # 日本語の文字を入れてみる
        # 引数は順に「(文字列の左上のx座標, 文字列の左上のy座標)」「フォントの指定」「文字色」
        draw.text((25, 45), place_name, font=font, fill='#ffffff')
        # opencvのガウシアンフィルターを適応
        image_np = np.asarray(image)
        blur = cv2.GaussianBlur(image_np, (5, 5), 0)
        # 画像の中心位置
        # 今回は画像サイズの中心をとっている
        center = tuple(np.array([blur.shape[1] * 0.5, blur.shape[0] * 0.5]))
        # 画像サイズの取得(横, 縦)
        size = tuple(np.array([blur.shape[1], blur.shape[0]]))
        # 回転させたい角度
        # ラジアンではなく角度(°)
        angle = 15.0
        # 拡大比率
        scale = 1.0
        # 回転変換行列の算出
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
        # アフィン変換
        blur_rot = cv2.warpAffine(blur, rotation_matrix, size, flags=cv2.INTER_CUBIC)
        affine_image = Image.fromarray(blur_rot)

        # 画像ファイルを保存する
        affine_image.save("./image/" + place_name + "/" + str(idx1) + "_GaussianBlur_Affine3_" + place_name + '.png', 'PNG')
        data = np.asarray(affine_image)        
        X.append(data.astype(np.float64))
        Y.append(idx2)

X = np.array(X)
Y = np.array(Y)
np.savez("./param/place_name.npz", x=X, y=Y)
print("ok,", len(Y))
