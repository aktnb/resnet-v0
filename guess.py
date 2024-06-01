from keras.models import load_model
import numpy as np
from PIL import Image
import glob


#   画像保存ディレクトリ
image_directory = "Vspo1/trimed_image"

#   画像ディレクトリから全ラベル取得
labels = list(map(lambda path: path.split('\\')[1], glob.glob(f'{image_directory}/**/')))
# labels = ['AngeKatrina', 'InuiToko', 'LizeHelesta']

print(labels)

model_path = r'result\230914-024424_72x72_30epoch_vspo1-100-images-sgd\230914-031646_model.h5'
input_size = 72



while True:
    print()
    print('Image path >>')
    img_path = input()

    if img_path.startswith('"'):
        img_path = img_path[1:-1]

    model = load_model(model_path)
    image = Image.open(img_path)
    # image = image.convert('RGB')
    image = image.convert('L')
    image = image.resize((input_size, input_size))
    data = np.asarray(image)
    X = np.array([data])
    X = X.astype('float')/255

    output = model.predict([X])[0]
    predict = output.argmax()
    print(output)
    print(f'{labels[predict]}')
