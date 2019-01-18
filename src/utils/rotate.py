import json
import os

import numpy as np

from PIL import Image


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
