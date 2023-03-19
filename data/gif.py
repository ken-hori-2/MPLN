# import glob

# from PIL import Image

# frames = []
# images = sorted(glob.glob("/Users/ken/Desktop/MP/Large/data/20230310_001258/*.png"))

# for image in images:
#     new_frame = Image.open(image)
#     frames.append(new_frame)

# frames[0].save('jpg_to_gif.gif',
#     format='GIF',
#     append_images=frames[1:],
#     save_all=True,
#     duration=500,
#     loop=0)




# from PIL import Image
# import os
# import glob
 
# # GIFアニメーションを作成
# def create_gif(in_dir, out_filename):
#     path_list = sorted(glob.glob(os.path.join(*[in_dir, '*']))) # ファイルパスをソートしてリストする
#     imgs = []                                                   # 画像をappendするための空配列を定義
 
#     # ファイルのフルパスからファイル名と拡張子を抽出
#     for i in range(len(path_list)):
#         img = Image.open(path_list[i])                          # 画像ファイルを1つずつ開く
#         imgs.append(img)                                        # 画像をappendで配列に格納していく
 
#     # appendした画像配列をGIFにする。durationで持続時間、loopでループ数を指定可能。
#     imgs[0].save(out_filename,
#                  save_all=True, append_images=imgs[1:], optimize=False, duration=100+200, loop=0)
 
# # GIFアニメーションを作成する関数を実行する
# create_gif(in_dir='/Users/ken/Desktop/MP/Large/data/20230310_001258', out_filename='animation2.gif')

#ライブラリのインポート
from PIL import Image
import glob
#画像を入れる箱を準備
pictures=[]

images = sorted(glob.glob("/Users/ken/Desktop/before/MPLN/data/20230318_023735_r/*.png"))
#画像を箱に入れていく
for i in range(len(images)+50):
    try:
        pic_name='/Users/ken/Desktop/before/MPLN/data/20230318_023735_r/' +str(i+1)+ '.png'
        img = Image.open(pic_name)
        pictures.append(img)
    except:
        pass
#gifアニメを出力する
pictures[0].save('anim.gif',save_all=True, append_images=pictures[1:],
optimize=True, duration=100, loop=0)