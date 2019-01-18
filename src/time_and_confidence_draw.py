import json
import os
import sys

import numpy as np

from PIL import Image

from openposedraw import draw_joints
from select_high_confidence_images import (get_max_confidence_and_idx,
                                           save_rotate_image)


def rotate_xy(x, y, deg, rot_center_x=0, rot_center_y=0):
    '''
    Args:
        x, y: int
            回転前の点の座標
        deg: int, float
            回転させる角度
        rot_center_x, rot_center_y: int, float
            回転中心
    Returns:
        rotated_corrd: tuple
            回転後の座標
    '''
    corrd = np.array([x - rot_center_x, y - rot_center_y])

    rad = np.radians(deg)
    cos = np.cos(rad)
    sin = np.sin(rad)

    rot_matrix = np.array([[cos, -1 * sin], [sin, cos]])

    rotated_corrd = np.dot(rot_matrix, corrd) + np.array([rot_center_x, rot_center_y])

    return rotated_corrd


def extract_keypoints_from_json(json_path):
    '''
    Args:
        json_path: str
            OpenPoseの結果jsonファイルのパス
    Returns:
        keypoints_array: numpy.array
            キーポイントが格納されたnp.array
            人を認識できていなかった場合要素が全て0のnp.arrayを返す
    '''
    with open(json_path) as f:
        json_dic = json.load(f)
    if json_dic['people']:
        keypoints_array = np.array(json_dic['people'][0]['pose_keypoints_2d'])
    else:
        keypoints_array = np.zeros(54)

    return keypoints_array


def euclidean_distance(pre_keypoints_array, keypoints_array):
    '''
    Args:
        pre_keypoints_array: numpy.array
            １フレーム前のキーポイントのnp.array
        keypoints_array: numpy.array
            現在のフレームのキーポイントのnp.array
    Returns:
        euclidean_dist: float
            与えられたキーポイント間のユークリッド距離
            頭の部分は無視して計算している(0~13番目のキーポイントのみ)
    '''
    euclidean_dist = 0
    for i in range(14):
        pre_xy = pre_keypoints_array[i * 3: i * 3 + 1]
        xy = keypoints_array[i * 3: i * 3 + 1]
        euclidean_dist += np.linalg.norm(pre_xy - xy)

    return euclidean_dist


def rotate_keypoints_array(keypoints_array, deg, rot_center_x=0, rot_center_y=0):
    '''
    Args:
        keypoints_array: numpy.array
            キーポイントが格納されたnp.array
        deg: int, float
            回転させる角度
        rot_center_x, rot_center_y: int, float
            回転中心の座標
    Returns:
        rotated_keypoints_array: numpy.array
            与えられたキーポイントをdeg度だけ回転させたnp.array
    '''
    rotated_keypoints_array = np.array([])

    for i in range(18):
        x = keypoints_array[i * 3]
        y = keypoints_array[i * 3 + 1]
        confidence = keypoints_array[i * 3 + 2]
        rotated_xy = rotate_xy(x, y, deg, rot_center_x, rot_center_y)
        rotated_keypoints_array = np.append(rotated_keypoints_array, rotated_xy)
        rotated_keypoints_array = np.append(rotated_keypoints_array, confidence)

    return rotated_keypoints_array


def make_list_in_dir(dir_path):
    '''
    Args:
        dir_path: str
            ディレクトリの名前
    Returns:
        list_in_dir: list
            引数で指定したディレクトリ内のディレクトリorファイルの名前のソート済みリスト
    '''
    list_in_dir = os.listdir(dir_path)
    list_in_dir.sort()

    return list_in_dir


def get_rot_center(img_path):
    '''
    Args:
        img_path: str
            画像のパス
    Returns:
        center_x, center_y: int, float
            画像の中心座標
    '''
    img = Image.open(img_path)
    width, height = img.size
    center_x, center_y = width / 2, height / 2

    return center_x, center_y


def main():
    args = sys.argv

    BASE_PATH = args[1]
    JSON_PATH = os.path.join(BASE_PATH, 'json')
    IMGS_PATH = os.path.join(BASE_PATH, 'images')
    MAX_DIST = os.getenv('DIST_THRESHOLD')
    TIME_AND_CONFIDENCE_DRAW_PATH = os.path.join(
                                        BASE_PATH,
                                        'for_time_and_confidence_draw_video_{}'.format(MAX_DIST)
                                    )
    if not os.path.isdir(TIME_AND_CONFIDENCE_DRAW_PATH):
        os.mkdir(TIME_AND_CONFIDENCE_DRAW_PATH)

    INPUT_IMAGES_PATH = args[2]
    input_imgs_list = make_list_in_dir(INPUT_IMAGES_PATH)
    input_imgs_list = [img for img in input_imgs_list if 'jpg' in img]

    log_file_path = os.path.join(TIME_AND_CONFIDENCE_DRAW_PATH, 'log.txt')
    f = open(log_file_path, 'w')

    # OpenPoseの結果jsonと画像が入っているディレクトリの名前のリスト
    # image000001, image000002, ...
    json_dir_list = make_list_in_dir(JSON_PATH)
    imgs_dir_list = make_list_in_dir(IMGS_PATH)

    exists_first_keypoints = False

    imgs_dir_path = os.path.join(IMGS_PATH, imgs_dir_list[0])
    imgs_name_list = make_list_in_dir(imgs_dir_path)
    img_path = os.path.join(imgs_dir_path, imgs_name_list[0])
    rot_center_x, rot_center_y = get_rot_center(img_path)

    for json_dir, imgs_dir, input_img in zip(json_dir_list, imgs_dir_list, input_imgs_list):
        # OpenPoseの結果jsonと画像が格納されているディレクトリ
        json_dir_path = os.path.join(JSON_PATH, json_dir)
        imgs_dir_path = os.path.join(IMGS_PATH, imgs_dir)

        # jsonと画像のファイル名を格納したリスト
        # 10度ごとに回転させたものが入っている
        # image000001_rotate000_keypoints.json, image000001_rotate010_keypoints.json, ...
        json_name_list = make_list_in_dir(json_dir_path)
        imgs_name_list = make_list_in_dir(imgs_dir_path)

        input_img_path = os.path.join(INPUT_IMAGES_PATH, input_img)

        # 最初のキーポイントが定まっていなかった時はconfidenceで判断
        if not exists_first_keypoints:
            print('{} ===================================='.format(json_dir))
            print('    method: confidence')
            f.write('{} ====================================\n'.format(json_dir))
            f.write('    method: confidence\n')

            max_confidence, max_confidence_idx = get_max_confidence_and_idx(json_dir_path)
            max_confidence_json_path = os.path.join(json_dir_path,
                                                    json_name_list[max_confidence_idx])
            max_confidence_keypoints = extract_keypoints_from_json(max_confidence_json_path)

            max_image_name = imgs_name_list[max_confidence_idx]
            max_image_path = os.path.join(imgs_dir_path, max_image_name)
            if max_confidence_keypoints.any():
                exists_first_keypoints = True
                pre_keypoints_array = rotate_keypoints_array(max_confidence_keypoints,
                                                             max_confidence_idx * 10,
                                                             rot_center_x=rot_center_x,
                                                             rot_center_y=rot_center_y)
                reshaped_pre_keypoints_array = pre_keypoints_array.reshape([18, 3])

                OUTPUT_PATH = os.path.join(TIME_AND_CONFIDENCE_DRAW_PATH, '{}.png'.format(imgs_dir))
                draw_joints(input_img_path, OUTPUT_PATH, reshaped_pre_keypoints_array)
            else:
                OUTPUT_PATH = os.path.join(TIME_AND_CONFIDENCE_DRAW_PATH, '{}.png'.format(imgs_dir))
                save_rotate_image(max_image_path, OUTPUT_PATH, 0)

            print('    confidence score: {}'.format(max_confidence))
            f.write('    confidence score: {}\n'.format(max_confidence))
        # 最初のキーポイントが定まっているときは時系列を考慮して
        else:
            euclidean_dist_list = np.array([])

            for i in range(len(json_name_list)):
                deg = i * 10

                json_name = json_name_list[i]
                json_name_path = os.path.join(json_dir_path, json_name)

                keypoints_array = extract_keypoints_from_json(json_name_path)

                if keypoints_array.any():
                    rotated_keypoints_array = rotate_keypoints_array(keypoints_array,
                                                                     deg,
                                                                     rot_center_x=rot_center_x,
                                                                     rot_center_y=rot_center_y)

                    euclidean_dist = euclidean_distance(pre_keypoints_array,
                                                        rotated_keypoints_array)
                    euclidean_dist_list = np.append(euclidean_dist_list, euclidean_dist)
                else:
                    euclidean_dist_list = np.append(euclidean_dist_list, np.inf)

            if np.min(euclidean_dist_list) > MAX_DIST:
                print('{} ===================================='.format(json_dir))
                print('    method: confidence')
                f.write('{} ====================================\n'.format(json_dir))
                f.write('    method: confidence\n')
                f.write('    dist: {}\n'.format(np.min(euclidean_dist_list)))

                max_confidence, max_confidence_idx = get_max_confidence_and_idx(json_dir_path)
                max_confidence_json_path = os.path.join(json_dir_path,
                                                        json_name_list[max_confidence_idx])
                max_confidence_keypoints = extract_keypoints_from_json(max_confidence_json_path)

                max_image_name = imgs_name_list[max_confidence_idx]
                max_image_path = os.path.join(imgs_dir_path, max_image_name)
                if max_confidence_keypoints.any():
                    exists_first_keypoints = True
                    pre_keypoints_array = rotate_keypoints_array(max_confidence_keypoints,
                                                                 max_confidence_idx * 10,
                                                                 rot_center_x=rot_center_x,
                                                                 rot_center_y=rot_center_y)
                    reshaped_pre_keypoints_array = pre_keypoints_array.reshape([18, 3])

                    OUTPUT_PATH = os.path.join(TIME_AND_CONFIDENCE_DRAW_PATH,
                                               '{}.png'.format(imgs_dir))
                    draw_joints(input_img_path, OUTPUT_PATH, reshaped_pre_keypoints_array)
                else:
                    OUTPUT_PATH = os.path.join(TIME_AND_CONFIDENCE_DRAW_PATH,
                                               '{}.png'.format(imgs_dir))
                    save_rotate_image(max_image_path, OUTPUT_PATH, 0)

                print('    confidence score: {}'.format(max_confidence))
                f.write('    confidence score: {}\n'.format(max_confidence))

            else:
                print('{} ===================================='.format(json_dir))
                print('    method: time series')
                f.write('{} ====================================\n'.format(json_dir))
                f.write('    method: time series\n')

                sorted_euclidean_dist_list = np.sort(euclidean_dist_list)
                f.write('    1: {}\n'.format(sorted_euclidean_dist_list[0]))
                f.write('    2: {}\n'.format(sorted_euclidean_dist_list[1]))
                f.write('    3: {}\n'.format(sorted_euclidean_dist_list[2]))

                nearest_idx = np.argmin(euclidean_dist_list)
                # nearest_image_name = imgs_name_list[nearest_idx]
                # nearest_image_path = os.path.join(imgs_dir_path, nearest_image_name)

                nearest_json_path = os.path.join(json_dir_path, json_name_list[nearest_idx])
                nearest_keypoints_array = extract_keypoints_from_json(nearest_json_path)
                pre_keypoints_array = rotate_keypoints_array(nearest_keypoints_array,
                                                             nearest_idx * 10,
                                                             rot_center_x=rot_center_x,
                                                             rot_center_y=rot_center_y)
                reshaped_pre_keypoints_array = pre_keypoints_array.reshape([18, 3])

                OUTPUT_PATH = os.path.join(TIME_AND_CONFIDENCE_DRAW_PATH, '{}.png'.format(imgs_dir))
                draw_joints(input_img_path, OUTPUT_PATH, reshaped_pre_keypoints_array)

    f.close()


if __name__ == '__main__':
    main()
