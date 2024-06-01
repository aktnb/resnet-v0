import datetime
import glob
import os
import time

from PIL import ImageGrab, Image
import imagehash


def watch_clipboard(save_dir, prefix) -> None:

    #   ディレクトリがなければ作成
    os.makedirs(save_dir, exist_ok=True)

    # hash値を記録しておく
    old_imgs = set()

    #   既存の画像を取得
    existings = glob.glob(os.path.join(save_dir, '*.png'))

    #   各画像に対して
    for existing in existings:
        #   Hashを計算
        existing_hash = imagehash.average_hash(Image.open(existing))
        #   既にあるHashと一致した場合
        if existing_hash in old_imgs:
            #   画像削除
            os.remove(existing)
        #   どのHashとも一致しなかった場合
        else:
            #   Hashを記録
            old_imgs.add(existing_hash)

    existings = glob.glob(os.path.join(save_dir, '*.png'))
    print(f'There is {len(existings)} imgs in {save_dir}')

    print('Start clipboard monitoring')

    initial = True

    #   ループ
    while True:
        try:
            #   クリップボード読み込み
            cur_img = ImageGrab.grabclipboard()
        except:
            continue

        #   画像以外の場合
        if cur_img is None or type(cur_img) is list:
            time.sleep(1)
            initial = False
            continue

        #   画像のHashを計算
        cur_hash = imagehash.average_hash(cur_img)

        if initial:
            old_imgs.add(cur_hash)
            initial = False
            continue

        #   既にあるHashと一致した場合ｊ
        if cur_hash in old_imgs:
            time.sleep(1)
            continue

        #   画像保存
        img_name = prefix + '-' + datetime.datetime.now().strftime('%y%m%d-%H%M%S') + '.png'
        cur_img.save(
            os.path.join(save_dir, img_name))
        print(f'saving image as {img_name} ({cur_hash})')

        #   Hash追加
        old_imgs.add(cur_hash)

        time.sleep(0.5)


if __name__ == '__main__':
    import sys
    args = sys.argv
    if len(args) < 2:
        print()
        print('Please Name >>')
        name = input()
    else:
        name = args[1]
    watch_clipboard(os.path.join('image_source', name), name.lower())