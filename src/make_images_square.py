import math
import os
import sys

import numpy as np
from PIL import Image

from utils.file import make_list_in_dir


def make_image_square(img, width, height):
    '''
    Args: 画像ファイルのパス
    Returns: 余白を入れて正方形にした画像
    '''
    diag = np.sqrt(width ** 2 + height ** 2)
    diag = int(math.ceil(diag))

    margin_w = (diag - width) // 2
    margin_h = (diag - height) // 2

    color = (0, 0, 0)
    square_img = Image.new(img.mode, (diag, diag), color)
    square_img.paste(img, (margin_w, margin_h))

    return square_img


def make_images_in_directory_square(image_path, output_path=None):
    '''
    Args:
        image_path:
            回転させたい画像が入っているディレクトリのパス
        output_path:
            回転させた画像を保存するディレクトリ
            Noneの場合元のファイルを上書きする
    Returns:
        指定ディレクトリ内の画像を回転して保存する
    '''
    img_list = make_list_in_dir(image_path)

    for img_name in img_list:
        image_path = output_path if output_path else image_path
        img_path = os.path.join(image_path, img_name)
        img = Image.open(img_path)

        width, height = img.size
        square_img = make_image_square(img, width, height)

        square_img.save(img_path)


def main():
    args = sys.argv

    IMAGE_DIR_PATH = args[1]
    OUTPUT_PATH = args[2] if len(args) == 3 else None

    make_images_in_directory_square(IMAGE_DIR_PATH, OUTPUT_PATH)


if __name__ == '__main__':
    main()
