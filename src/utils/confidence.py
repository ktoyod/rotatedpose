import json
import os

import numpy as np

from utils.file import make_list_in_dir


def calc_confidence(keypoints_list, with_face=False):
    '''
    Args: キーポイントのリスト
    Returns: 信頼度の平均
    '''
    keypoints_array = np.array(keypoints_list)
    confidence_array = keypoints_array[2::3] if with_face else keypoints_array[2:14*3:3]

    return np.mean(confidence_array)


def create_confidence_array(json_dir_path, with_face=False):
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
            confidence_list.append(calc_confidence(keypoints_list, with_face=with_face))
        else:
            confidence_list.append(0)
    return np.array(confidence_list)


def get_max_confidence_and_idx(json_dir_path, with_face=False):
    '''
    /path/to/images/hoge,/path/to/json/hoge
    を渡すとconfidenceが最大となるインデックスを返す
    '''
    confidence_array = create_confidence_array(json_dir_path, with_face=with_face)

    max_confidence = np.max(confidence_array)
    max_confidence_idx = np.argmax(confidence_array) if confidence_array.any() else 0

    return max_confidence, max_confidence_idx
