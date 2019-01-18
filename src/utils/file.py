import json
import os

import numpy as np


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
