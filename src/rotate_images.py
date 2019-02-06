import os
import sys

from PIL import Image

from settings import Settings
from utils.file import make_list_in_dir


def create_rotate_images(image_path, deg_split=10):
    img = Image.open(image_path)
    img_dir_name = image_path.split('/').rstrip('.jpg')
    os.mkdir(img_dir_name)

    deg_num = 360 // deg_split
    for i in range(deg_num):
        img_rotate = img.rotate(deg_split * i)
        jpg_path = os.path.join(
                       img_dir_name,
                       '{}_rotate{:03}.jpg'.format(img_dir_name, deg_split * i)
                       )
        img_rotate.save(jpg_path)


def main():
    args = sys.argv

    IMAGE_PATH = args[1]
    OUTPUT_PATH = os.path.join(IMAGE_PATH, 'rotated_images')
    if not os.path.isdir(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)

    DEG_SPLIT = Settings.DEG_SPLIT

    img_list = make_list_in_dir(IMAGE_PATH, expanded='jpg')

    for img_name in img_list:
        image_path = os.path.join(IMAGE_PATH, img_name)
        create_rotate_images(image_path, deg_split=DEG_SPLIT)


if __name__ == '__main__':
    main()
