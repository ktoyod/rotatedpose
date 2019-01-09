import os
import sys

from PIL import Image

args = sys.argv
IMAGE_DIR = args[1] if len(args) == 2 else './'
OUTPUT_DIR = os.path.join(IMAGE_DIR, 'rotated_images')
if not os.path.isdir(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)


def create_rotate_images(img_name, deg_split=10):
    img = Image.open(os.path.join(IMAGE_DIR, img_name))
    img_dir_name = os.path.join(OUTPUT_DIR, img_name.rstrip(".jpg"))
    os.mkdir(img_dir_name)

    deg_num = 360 // deg_split
    for i in range(deg_num):
        img_rotate = img.rotate(deg_split * i)
        jpg_path = os.path.join(img_dir_name, '{}_rotate{:03}.jpg'.format(img_name.rstrip(".jpg"), deg_split * i))
        img_rotate.save(jpg_path)


img_list = os.listdir(IMAGE_DIR)
img_list.sort()
img_list = [img for img in img_list if '.jpg' in img]

for img_name in img_list:
    print(img_name)
    create_rotate_images(img_name)
