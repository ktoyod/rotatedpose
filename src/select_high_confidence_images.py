import json
import os
import sys

import numpy as np
from PIL import Image


def calc_confidence(keypoints_list):
    '''
    Args: キーポイントのリスト
    Returns: 信頼度の平均
    '''
    keypoints_array = np.array(keypoints_list)
    confidence_array = keypoints_array[2::3]

    return np.mean(confidence_array)


def calc_confidence_without_face_joints(keypoints_list):
    '''
    Args: キーポイントのリスト
    Returns: 信頼度の平均
    '''
    keypoints_array = np.array(keypoints_list)
    confidence_array = keypoints_array[2:14 * 3:3]

    return np.mean(confidence_array)


def calc_confidence_without_untrusted_joints(keypoints_list):
    '''
    Args:
        keypoints_list: list
            キーポイントのリスト
    Returns:
        confidence 0のジョイント以外の平均値
    '''
    keypoints_array = np.array(keypoints_list)
    confidence_array = keypoints_array[2::3]

    trusted_confidence_array = confidence_array[np.nonzero(confidence_array)]

    return np.mean(trusted_confidence_array)


def create_confidence_array(json_dir_path):
    '''
    Args:
        json_dir: jsonファイルが置いてあるディレクトリ名
    Returns:
        各画像の信頼度の平均が格納されたnumpy.array
    '''
    json_list = make_list_in_dir(json_dir_path)

    confidence_list = []
    for file_name in json_list:
        json_file_path = os.path.join(json_dir_path, file_name)
        with open(json_file_path) as f:
            json_dic = json.load(f)
        if json_dic['people']:
            keypoints_list = json_dic['people'][0]['pose_keypoints_2d']
            confidence_list.append(calc_confidence(keypoints_list))
        else:
            confidence_list.append(0)
    return np.array(confidence_list)


def create_confidence_array_without_face_joints(json_dir_path):
    '''
    Args:
        json_dir: jsonファイルが置いてあるディレクトリ名
    Returns:
        各画像の信頼度の平均が格納されたnumpy.array
    '''
    json_list = make_list_in_dir(json_dir_path)

    confidence_list = []
    for file_name in json_list:
        json_file_path = os.path.join(json_dir_path, file_name)
        with open(json_file_path) as f:
            json_dic = json.load(f)
        if json_dic['people']:
            keypoints_list = json_dic['people'][0]['pose_keypoints_2d']
            confidence_list.append(calc_confidence_without_face_joints(keypoints_list))
        else:
            confidence_list.append(0)
    return np.array(confidence_list)


def create_confidence_array_without_untrusted_joints(json_dir_path):
    '''
    Args:
        json_dir: jsonファイルが置いてあるディレクトリ名
    Returns:
        各画像の信頼度の平均が格納されたnumpy.array
    '''
    json_list = make_list_in_dir(json_dir_path)

    confidence_list = []
    for file_name in json_list:
        json_file_path = os.path.join(json_dir_path, file_name)
        with open(json_file_path) as f:
            json_dic = json.load(f)
        if json_dic['people']:
            keypoints_list = json_dic['people'][0]['pose_keypoints_2d']
            confidence_list.append(calc_confidence_without_untrusted_joints(keypoints_list))
        else:
            confidence_list.append(0)
    return np.array(confidence_list)


def make_list_in_dir(dir_name):
    '''
    Args: ディレクトリの名前
    Returns: 指定ディレクトリ内のディレクトリorファイルの名前のリスト
    '''
    list_in_dir = os.listdir(dir_name)
    list_in_dir.sort()
    return list_in_dir


def save_rotate_image(image_path, save_path, deg):
    '''
    画像の回転を戻して保存する
    '''
    img = Image.open(image_path)
    img_rotate = img.rotate(deg)
    img_rotate.save(save_path)


def get_max_confidence_and_idx(json_dir_path):
    '''
    /path/to/images/hoge,/path/to/json/hoge
    を渡すとconfidenceが最大となるインデックスを返す
    '''
    confidence_array = create_confidence_array(json_dir_path)

    max_confidence = np.max(confidence_array)
    max_confidence_idx = np.argmax(confidence_array) if confidence_array.any() else 0

    return max_confidence, max_confidence_idx


def get_max_confidence_and_idx_without_face_joints(json_dir_path):
    '''
    /path/to/images/hoge,/path/to/json/hoge
    を渡すとconfidenceが最大となるインデックスを返す
    '''
    confidence_array = create_confidence_array_without_face_joints(json_dir_path)

    max_confidence = np.max(confidence_array)
    max_confidence_idx = np.argmax(confidence_array) if confidence_array.any() else 0

    return max_confidence, max_confidence_idx


def get_max_confidence_and_idx_without_untrusted_joints(json_dir_path):
    '''
    /path/to/images/hoge,/path/to/json/hoge
    を渡すとconfidenceが最大となるインデックスを返す
    '''
    confidence_array = create_confidence_array_without_untrusted_joints(json_dir_path)

    max_confidence = np.max(confidence_array)
    max_confidence_idx = np.argmax(confidence_array) if confidence_array.any() else 0

    return max_confidence, max_confidence_idx


def main():
    args = sys.argv
    OPENPOSE_OUTPUT_DIR = args[1] if len(args) == 2 else './'
    IMAGE_DIR = os.path.join(OPENPOSE_OUTPUT_DIR, 'images')
    JSON_DIR = os.path.join(OPENPOSE_OUTPUT_DIR, 'json')
    FOR_VIDEO_DIR = os.path.join(OPENPOSE_OUTPUT_DIR, 'for_confidence_video')
    if not os.path.isdir(FOR_VIDEO_DIR):
        os.mkdir(FOR_VIDEO_DIR)

    image_dir_list = make_list_in_dir(IMAGE_DIR)
    json_dir_list = make_list_in_dir(JSON_DIR)

    for image_dir_name, json_dir_name in zip(image_dir_list, json_dir_list):
        image_dir_path = os.path.join(IMAGE_DIR, image_dir_name)
        json_dir_path = os.path.join(JSON_DIR, json_dir_name)

        max_confidence, max_confidence_idx = get_max_confidence_and_idx(json_dir_path)

        image_list = make_list_in_dir(image_dir_path)

        deg_split = os.getenv('DEG_SPLIT')
        rotate_deg = -1 * max_confidence_idx * deg_split
        max_confidence_image_path = os.path.join(image_dir_path, image_list[max_confidence_idx])
        save_path = os.path.join(FOR_VIDEO_DIR, '{}.png'.format(image_dir_name))
        save_rotate_image(max_confidence_image_path, save_path, rotate_deg)

        # print('{}---------------------------------------------------'.format(image_dir_name))
        # print('    confidence_array: {}'.format(confidence_array))
        # if confidence_array.any():
        #     print('    max_confidence: {}'.format(confidence_array[max_confidence_idx]))
        #     print('    image_name: {}'.format(image_list[max_confidence_idx]))
        # else:
        #     print('    max_confidence: None')
        #     print('    image_name: image_list[0]')


if __name__ == '__main__':
    main()
