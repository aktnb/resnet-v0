import cv2
import numpy as np
import os
import datetime

#   ウィンドウの最大サイズ
WINDOW_HEIGHT = 900
WINDOW_WIDTH = 1600

class TrimingTool:


    def __init__(self, img_path, saving_dir):

        self.img_path = img_path
        self.saving_dir = saving_dir

        #   画像読み込み
        self.img = cv2.imread(img_path)
        #   初期化
        self.pos1 = None
        self.saveings = list()

        #   保存用ディレクトリがなければ作成
        os.makedirs(saving_dir, exist_ok=True)

        #   読み込んだ画像の高さと幅
        height = self.img.shape[0]
        width  = self.img.shape[1]

        #   最大ウィンドウサイズに収まるように係数を算出
        if height < WINDOW_HEIGHT and width < WINDOW_WIDTH:
            factor = 1
        elif height/width > WINDOW_HEIGHT/WINDOW_WIDTH:
            factor = WINDOW_HEIGHT / height
        else:
            factor = WINDOW_WIDTH / width

        #   係数をかけてリサイズ
        self.resized_height = int(factor*height)
        self.resized_width = int(factor*width)
        self.resized_img = cv2.resize(self.img, (self.resized_width, self.resized_height))
        #   描画用にバッファ
        self.buffer_img = np.copy(self.resized_img)

        #   描画
        cv2.imshow(self.img_path, self.resized_img)
        #   マウスイベントのコールバックを設定
        cv2.setMouseCallback(self.img_path, self.mouse_callback)
        key = cv2.waitKey(0)
        if key == 113:
            raise Exception()
        cv2.destroyAllWindows()

        #   タイムスタンプ習得
        timestamp = datetime.datetime.now().strftime('%y%m%d-%H%M%S')
        #   各切り出しポイントについて
        for idx, ((x1, y1), (x2, y2)) in enumerate(self.saveings):
            #   画像名
            img_name = f'{timestamp}-{idx}.png'
            #   切り出した画像
            trimed_img = self.resized_img[min(y1,y2):max(y1,y2),min(x1,x2):max(x1,x2)]
            #   保存
            cv2.imwrite(os.path.join(save_dir, img_name), trimed_img)


    #   マウスイベントのコールバック
    def mouse_callback(self, event, x, y, flags, params):

        x = min(x, self.resized_width-1)
        y = min(y, self.resized_height-1)
        #   マウスが動かされた場合
        if event == cv2.EVENT_MOUSEMOVE:

            #   バッファをコピー
            img2 = np.copy(self.buffer_img)

            #   補助線描画
            #   x軸
            for u in range(self.resized_width):
                img2[y][u][0] = 255 - img2[y][u][0]
                img2[y][u][1] = 255 - img2[y][u][1]
                img2[y][u][2] = 255 - img2[y][u][2]
            #   y軸
            for v in range(self.resized_height):
                img2[v][x][0] = 255 - img2[v][x][0]
                img2[v][x][1] = 255 - img2[v][x][1]
                img2[v][x][2] = 255 - img2[v][x][2]

            #   選択中の矩形を描画
            if self.pos1 is not None:
                cv2.rectangle(img2, self.pos1, (x, y), (0, 0, 255))

            #   表示
            cv2.imshow(self.img_path, img2)
            return

        #   左クリック
        elif event == cv2.EVENT_LBUTTONDOWN:
            #   1つ目なら頂点を記録
            if self.pos1 is None:
                self.pos1 = (x, y)
            #   2つ目なら矩形を記録
            else:
                self.saveings.append((self.pos1, (x, y)))
                self.pos1 = None

        #   右クリック
        elif event == cv2.EVENT_RBUTTONDOWN:
            #   1つ目が記録されていたら消す
            if self.pos1 is not None:
                self.pos1 = None
            #   それ以外なら、切り出し区間を1つずつ消す
            elif len(self.saveings) > 0:
                self.saveings.pop()

        #   バッファをコピー
        img2 = np.copy(self.resized_img)

        #   選択済の矩形を描画
        for saving in self.saveings:
            cv2.rectangle(img2, saving[0], saving[1], (0, 255, 0))

        #   バッファに保存
        self.buffer_img = np.copy(img2)

        #   表示
        cv2.imshow(self.img_path, img2)


if __name__ == '__main__':
    import sys, glob
    file = None
    if len(sys.argv) > 2:
        file = sys.argv[1]
        print()
        print('Please Name >>')
        name = input()
    else:
        print()
        print('Please Name >>')
        name = input()

    img_dir = os.path.join('image_source', name)
    save_dir = os.path.join('trimed_image', name)
    img_paths = glob.glob(os.path.join(img_dir, '*.png'))
    print(f'There is {len(img_paths)} images in {img_dir}')

    if file is not None:
        TrimingTool(file, save_dir)
    else:
        try:
            for img_path in img_paths:
                TrimingTool(img_path, save_dir)
        except:
            print('exit')
