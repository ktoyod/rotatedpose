import numpy as np


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
