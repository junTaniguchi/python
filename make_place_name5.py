# -*- encoding:utf8 -*-
import os
from PIL import Image, ImageDraw, ImageFont
import cv2
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

def gaussianBlur_image(img_name, 
                       place_name, 
                       image_np, 
                       idx1, 
                       idx2, 
                       X, 
                       Y, 
                       font_size, 
                       axis_detail,
                       place_num):
    # opencvのガウシアンフィルターを適応
    blur = cv2.GaussianBlur(image_np, (5, 5), 0)
    blur_image = Image.fromarray(blur)
    img_file_name = "./image/" + place_name + "/" + str(idx1) + img_name + "_GaussianBlur_" + str(font_size)  + place_name + '.png'
    # 画像ファイルを保存する
    blur_image.save(img_file_name, 'PNG')
    '''
    print(place_num)
    print(axis_detail1[0])
    print(axis_detail1[1])
    print(axis_detail1[2])
    print(axis_detail1[3])
    print(axis_detail2[0])
    print(axis_detail2[1])
    print(axis_detail2[2])
    print(axis_detail2[3])
    '''
    data = np.asarray(blur_image)        

    X.append(data.astype(np.float64))
    img_detail = [idx2, place_num, axis_detail]
    Y.append(img_detail)
   
def make_img (img_name,
              jp_font_list, 
              place_list, 
              x_pixel, 
              y_pixel, 
              color):
    X = []
    Y = []
    
    for idx1, jp_font in enumerate(jp_font_list):
        for font_size in range(y_pixel-30, y_pixel):
            i = 0
            # フォントの指定。引数は順に「フォントのパス」「フォントのサイズ」「エンコード」
            # メソッド名からも分かるようにTure Typeのフォントを指定する
            font = ImageFont.truetype(jp_font, font_size)
        
            for idx2, place_name in enumerate(place_list):
                # 画像のピクセルを指定
                image = Image.new('RGB', (x_pixel, y_pixel), color)
                draw = ImageDraw.Draw(image)
                # 表示文字のmarginを設定
                str_width = font_size * len(place_name)
                limit_x_draw_pixel = x_pixel - str_width
                limit_y_draw_pixel = y_pixel - font_size
                if limit_x_draw_pixel > 20:
                    x_range = np.arange(0, limit_x_draw_pixel, 4)
                    y_range = np.arange(0, limit_y_draw_pixel, 4)
                    for x_draw_pixel in x_range:
                        for y_draw_pixel in y_range:
                            place_num = 0
                            width =  str_width
                            height = font_size
                            axis_detail = []
                            axis_detail.append([x_draw_pixel,
                                                y_draw_pixel,
                                                x_draw_pixel + width,
                                                y_draw_pixel + height])
                            
                            # 日本語の文字を入れてみる
                            # 引数は順に「(文字列の左上のx座標, 文字列の左上のy座標)」「フォントの指定」「文字色」
                            draw.text((x_draw_pixel, y_draw_pixel), place_name, font=font, fill='#ffffff')
                            place_num+=1
                            if x_draw_pixel % 2 == 0:
                                if y_draw_pixel < y_pixel / 2:
                                    y_special_draw_pixel = y_draw_pixel / 2
                                    axis_detail.append([x_draw_pixel,
                                                        y_special_draw_pixel,
                                                        x_draw_pixel + width,
                                                        y_special_draw_pixel + height])
                                    draw.text((x_draw_pixel, y_special_draw_pixel), "東京", font=font, fill='#ffffff')
                                    place_num+=1
                                if x_draw_pixel < x_pixel / 2:
                                    x_special_draw_pixel = x_draw_pixel / 2
                                    axis_detail.append([x_special_draw_pixel,
                                                        y_draw_pixel,
                                                        x_special_draw_pixel + width,
                                                        y_draw_pixel + height])
                                    draw.text((x_draw_pixel, y_special_draw_pixel), "名古屋", font=font, fill='#ffffff')
                                    place_num+=1
                                if y_draw_pixel < y_pixel / 2 and x_draw_pixel < x_pixel / 2:
                                    y_special_draw_pixel = y_draw_pixel / 2
                                    x_special_draw_pixel = x_draw_pixel / 2
                                    axis_detail.append([x_special_draw_pixel,
                                                        y_special_draw_pixel,
                                                        x_special_draw_pixel + width,
                                                        y_special_draw_pixel + height])
                                    draw.text((x_draw_pixel, y_special_draw_pixel), "大阪", font=font, fill='#ffffff')
                                    place_num+=1

                                    
                            image_np = np.asarray(image)
                            gaussianBlur_image(img_name, place_name, image_np, idx1, idx2, X, Y, font_size, axis_detail, place_num)
                            print(i)
                            i+=1
    X = np.array(X)
    np.save("./param/npz/place_name_%s_%s_%s.npy" % (str(font_size), str(x_pixel), str(y_pixel)), X)
    with open("./param/npz/label.txt", 'a') as y_file:
        y_file.write(Y + '\n')
    print("ok,", len(Y))

for y_pixel in [60]:
    for x_pixel in [150]:#range(150, 180):
        make_img("1", jp_font_list, place_list, x_pixel=x_pixel, y_pixel=y_pixel, color="blue")
        #make_img("2", jp_font_list, place_list, x_pixel=x_pixel, y_pixel=y_pixel, color="green")