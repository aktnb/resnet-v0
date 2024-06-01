import datetime
import glob
import os
import tensorflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU, MaxPooling2D, concatenate, GlobalAveragePooling2D, Dense
from keras.models import Model
from keras.utils import np_utils, plot_model
from keras.optimizers import SGD, Adam


#   タイムスタンプ習得
timestamp = datetime.datetime.now().strftime('%y%m%d-%H%M%S')

#   データセット
dataset_path = r'dataset\230915-213942-train_test-vspo1.npy'

#   epoch
epoch = 50
#   input size
input_size = 72
#   color type
isGray = False
#   learning rate
learning_rate=0.0001

#   追加情報
discription = 'vspo1-100-images-sgd'

#   画像ディレクトリ
img_dir = 'Vspo1/trimed_image'

#   出力先
out_dir = f'result/{timestamp}_{input_size}x{input_size}_{epoch}epoch_{discription}'
model_path = f'{out_dir}/{timestamp}_model.h5'
history_path = f'{out_dir}/{timestamp}_history.csv'
os.makedirs(out_dir)


labels = list(map(lambda path: path.split('\\')[1], glob.glob(f'{img_dir}/**/')))
# labels = ['AngeKatrina', 'InuiToko', 'LizeHelesta']


def residual_block(filters, img, strides=1):
    conv_0 = BatchNormalization()(img)
    conv_1 = LeakyReLU()(conv_0)
    conv_2 = Conv2D(filters, (3, 3), strides=strides, padding='same')(conv_1)
    conv_3 = BatchNormalization()(conv_2)
    conv_4 = LeakyReLU()(conv_3)
    conv_5 = Conv2D(filters, (3, 3), strides=1, padding='same')(conv_4)
    if strides == 1:
        return concatenate([img, conv_5])
    else:
        conv_6 = Conv2D(filters, (1, 1), strides=2, padding='same')(conv_0)
        return concatenate([conv_5, conv_6])


def resnet_18():
    input_layer = Input(shape=(input_size, input_size, 1 if isGray else 3))

    conv1 = Conv2D(64, (7, 7), strides=2, activation=LeakyReLU(), padding='same')(input_layer)

    conv2_1 = MaxPooling2D((3, 3), strides=2)(conv1)
    conv2_2 = residual_block(64, conv2_1, 1)
    conv2_3 = residual_block(64, conv2_2, 1)

    conv3_1 = residual_block(128, conv2_3, 2)
    conv3_2 = residual_block(128, conv3_1, 1)

    conv4_1 = residual_block(256, conv3_2, 2)
    conv4_2 = residual_block(256, conv4_1, 1)

    conv5_1 = residual_block(512, conv4_2, 2)
    conv5_2 = residual_block(512, conv5_1, 1)

    avepool = GlobalAveragePooling2D()(conv5_2)
    output = Dense(len(labels), activation='softmax')(avepool)

    return Model(input_layer, output)


model = resnet_18()

model.compile(optimizer=SGD(learning_rate=learning_rate, momentum=0.9), loss='categorical_crossentropy',metrics=['accuracy'])

with open(dataset_path, 'rb') as f:
    X_train = np.load(f)
    X_test = np.load(f)
    y_train = np.load(f)
    y_test = np.load(f)
X_train = X_train.astype('float') / 255
X_test = X_test.astype('float') / 255
y_train = np_utils.to_categorical(y_train, len(labels))
y_test = np_utils.to_categorical(y_test, len(labels))

plot_model(model, to_file=f'{out_dir}/model_img.png')

model.summary()

print('fitting...')
his = model.fit(X_train, y_train, epochs = epoch, shuffle=True, validation_data=(X_test, y_test))

print('saving...')
model.save(model_path)

hist_df = pd.DataFrame(his.history)
hist_df.to_csv(history_path)

plt.plot(his.history['loss'])
plt.plot(his.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig(f'{out_dir}/{timestamp}-loss.png')
plt.show()
plt.clf()

plt.plot(his.history['accuracy'])
plt.plot(his.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig(f'{out_dir}/{timestamp}-acc.png')
plt.show()
plt.clf()
