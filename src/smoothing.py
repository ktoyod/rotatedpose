import numpy as np


def smoothing(keypoints_array, pre_keypoints_array, w_cnt=0.9):
    # weight for current and previous frame
    w_cnt = w_cnt
    w_pre = 1.0 - w_cnt

    joints_num = int(len(keypoints_array) / 3)
    reconst_keypoints_array = np.array([])
    for i in range(joints_num):
        if keypoints_array[3 * i + 2] == 0 or pre_keypoints_array[3 * i + 2] == 0:
            reconst_keypoints_array = np.append(reconst_keypoints_array, keypoints_array[3 * i])
            reconst_keypoints_array = np.append(reconst_keypoints_array, keypoints_array[3 * i + 1])
            reconst_keypoints_array = np.append(reconst_keypoints_array, keypoints_array[3 * i + 2])
        else:
            _reconst_keypoint_1 = pre_keypoints_array[3 * i] * w_pre + \
                    keypoints_array[3 * i] * w_cnt
            _reconst_keypoint_2 = pre_keypoints_array[3 * i + 1] * w_pre + \
                    keypoints_array[3 * i + 1] * w_cnt
            reconst_keypoints_array = np.append(reconst_keypoints_array, _reconst_keypoint_1)
            reconst_keypoints_array = np.append(reconst_keypoints_array, _reconst_keypoint_2)
            reconst_keypoints_array = np.append(reconst_keypoints_array, keypoints_array[3 * i + 2])

    return reconst_keypoints_array
