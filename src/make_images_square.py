import math
import os
import sys

import numpy as np
from PIL import Image

args = sys.argv
IMAGE_DIR = args[1] if len(args) == 2 else './'


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


def get_image_size(img):
    '''
    Args: 画像
    Returns: (width, height)
    '''
    return img.size
    

def main():
    img_list = os.listdir(IMAGE_DIR)

    for img_name in img_list:
        img_path = os.path.join(IMAGE_DIR, img_name)
        img = Image.open(img_path)

        width, height = get_image_size(img)
        square_img = make_image_square(img, width, height)

        square_img.save(img_path)


if __name__ == '__main__':
    main()
