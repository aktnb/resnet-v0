import glob
from PIL import Image
import numpy as np
import os
import datetime


#   画像保存ディレクトリ
image_directory = "Vspo1/trimed_image"

#   画像ディレクトリから全ラベル取得
labels = list(map(lambda path: path.split('\\')[1], glob.glob(f'{image_directory}/**/')))
# labels = ['AngeKatrina', 'InuiToko', 'LizeHelesta']

print(labels)

#   タイムスタンプ習得
timestamp = datetime.datetime.now().strftime('%y%m%d-%H%M%S')
dataset_path = f'dataset/{timestamp}-train_test-vspo1.npy'

img_size = (72, 72)

X_train = []
y_train = []
X_test = []
y_test = []

for cnum, label in enumerate(labels):
    #   画像ディレクトリ
    image_dir = os.path.join(image_directory, label)

    count = 0

    #   画像ファイルの全パス
    files = glob.glob(image_dir + '/*.png')

    print(f'{len(files)} images in {label} directory')

    #   各画像ファイルに対して
    for i, file in enumerate(files):

        #   画像読み込み
        image = Image.open(file)
        image = image.convert('RGB')
        # image = image.convert('L')
        image = image.resize(img_size)

        for angle in range(-20, 25, 5):
            #   回転
            img_r = image.rotate(angle)
            data1 = np.asarray(img_r)
            if count % 5 == 0:
                X_test.append(data1)
                y_test.append(cnum)
            else:
                X_train.append(data1)
                y_train.append(cnum)
            count += 1

            img_f = img_r.transpose(Image.FLIP_LEFT_RIGHT)
            data2 = np.asarray(img_f)
            if count % 5 == 0:
                X_test.append(data2)
                y_test.append(cnum)
            else:
                X_train.append(data2)
                y_train.append(cnum)
            count += 1

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)


with open(dataset_path, 'wb') as f:
    np.save(f, X_train)
    np.save(f, X_test)
    np.save(f, y_train)
    np.save(f, y_test)
