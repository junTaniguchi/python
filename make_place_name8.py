# -*- encoding:utf8 -*-
import os
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
from keras.utils import np_utils
import pickle

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

def gaussianBlur_image(img_name, 
                       place_name, 
                       image_np, 
                       idx1, 
                       idx2, 
                       X, 
                       Y, 
                       font_size, 
                       x_min_pixel, 
                       y_min_pixel,
                       x_max_pixel,
                       y_max_pixel,
                       file_name_list,
                       img_detail_cordinate,
                       img_detail_devision):
    # opencvのガウシアンフィルターを適応
    blur = cv2.GaussianBlur(image_np, (5, 5), 0)
    blur_image = Image.fromarray(blur)
    file_dir = "./image/"
    file_name = str(idx1) + img_name + "_" + place_name + "_" + str(x_min_pixel) + "_" + str(y_min_pixel) + "_" + str(x_max_pixel) + "_" + str(y_max_pixel) + '.png'
    img_file_name =  file_dir + file_name
    # 画像ファイルを保存する
    blur_image.save(img_file_name, 'PNG')
    xmin = x_min_pixel / 300
    ymin = y_min_pixel / 300
    xmax = x_max_pixel / 300
    ymax = y_max_pixel / 300
    # ラベルの作成
    img_detail = [xmin, ymin, xmax, ymax]
    data = np.asarray(blur_image)
    X.append(data.astype(np.float64))
    file_name_list.append(file_name)
    img_detail_cordinate.append(img_detail)
    img_detail_devision.append(idx2)
   
def make_img (img_name,
              jp_font_list, 
              place_list, 
              x_pixel, 
              y_pixel, 
              color):

    for idx1, jp_font in enumerate(jp_font_list):
        for font_size in range(y_pixel-250, y_pixel-249):
            # フォントの指定。引数は順に「フォントのパス」「フォントのサイズ」「エンコード」
            # メソッド名からも分かるようにTure Typeのフォントを指定する
            font = ImageFont.truetype(jp_font, font_size)
        
            for idx2, place_name in enumerate(place_list):

                # 表示文字のmarginを設定
                str_width = font_size * len(place_name)
                limit_x_draw_pixel = x_pixel - str_width
                limit_y_draw_pixel = y_pixel - font_size
                
                X = []
                file_name_list = []
                img_detail_cordinate = []
                img_detail_devision = []
                Y = []

                if limit_x_draw_pixel > 20:
                    x_range = np.arange(0, limit_x_draw_pixel, limit_x_draw_pixel / 2)
                    y_range = np.arange(0, limit_y_draw_pixel, limit_x_draw_pixel / 2)
                    for x_min_pixel in x_range:
                        for y_min_pixel in y_range:
                            # 画像のピクセルを指定
                            image = Image.new('RGB', (x_pixel, y_pixel), color)
                            draw = ImageDraw.Draw(image)
                            width =  str_width
                            height = font_size
                            x_max_pixel = x_min_pixel + width
                            y_max_pixel = y_min_pixel + height
                            # 日本語の文字を入れてみる
                            # 引数は順に「(文字列の左上のx座標, 文字列の左上のy座標)」「フォントの指定」「文字色」
                            draw.text((x_min_pixel, y_min_pixel), place_name, font=font, fill='#ffffff')
                            image_np = np.asarray(image)
                            gaussianBlur_image(img_name,
                                               place_name,
                                               image_np,
                                               idx1,
                                               idx2,
                                               X,
                                               Y,
                                               font_size,
                                               x_min_pixel,
                                               y_min_pixel,
                                               x_max_pixel,
                                               y_max_pixel,
                                               file_name_list,
                                               img_detail_cordinate,
                                               img_detail_devision)
    # Xをpickleファイルへ変換
    X = np.array(X)
    with open('./param/place_name_X.pkl', 'wb') as f:
        pickle.dump(X, f)
        f.close()
        
    img_detail_one_hot = np_utils.to_categorical(img_detail_devision, len(place_list)) 
    for idx, cordinate in enumerate(img_detail_cordinate):
        detail = np.hstack((np.reshape(np.array(cordinate),(1,4)), np.reshape(img_detail_one_hot[idx],(1,2))))
        Y.append(detail)
    Y = np.reshape(Y, (len(Y),6))
    
    print(Y)

    with open('./param/place_name_Y.pkl', 'wb') as f:
        pickle.dump(Y, f)
        f.close()
        #np.savez("./param/npz/place_name_%s_%s_%s.npz" % (str(font_size), str(x_pixel), str(y_pixel)), x=X, y=Y)
    print("ok,", len(Y))

for y_pixel in [300]:
    for x_pixel in [300]:#range(170, 180):
        make_img("1", jp_font_list, place_list, x_pixel=x_pixel, y_pixel=y_pixel, color="blue")
        #make_img("2", jp_font_list, place_list, x_pixel=x_pixel, y_pixel=y_pixel, color="green")