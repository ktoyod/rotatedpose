import os
import sys

import numpy as np

from select_high_confidence_images import (get_max_confidence_and_idx,
                                           save_rotate_image)
from utils.file import get_keypoints_array_from_json, make_list_in_dir
from utils.rotate import rotate_keypoints_array, get_rot_center
from utils.distance import euclidean_distance


def main():
    args = sys.argv

    DEG_SPLIT = os.getenv('DEG_SPLIT')
    BASE_PATH = args[1] if len(args) == 2 else './'
    JSON_PATH = os.path.join(BASE_PATH, 'json')
    IMGS_PATH = os.path.join(BASE_PATH, 'images')
    CONFIDENCE_WITH_DIST_INFO = os.path.join(BASE_PATH,
                                             'for_confidence_video_with_dist_info')
    if not os.path.isdir(CONFIDENCE_WITH_DIST_INFO):
        os.mkdir(CONFIDENCE_WITH_DIST_INFO)

    log_file_path = os.path.join(CONFIDENCE_WITH_DIST_INFO, 'conf_log_with_dist_info.txt')
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

    for json_dir, imgs_dir in zip(json_dir_list, imgs_dir_list):
        # OpenPoseの結果jsonと画像が格納されているディレクトリ
        json_dir_path = os.path.join(JSON_PATH, json_dir)
        imgs_dir_path = os.path.join(IMGS_PATH, imgs_dir)

        # jsonと画像のファイル名を格納したリスト
        # 10度ごとに回転させたものが入っている
        # image000001_rotate000_keypoints.json, image000001_rotate010_keypoints.json, ...
        json_name_list = make_list_in_dir(json_dir_path)
        imgs_name_list = make_list_in_dir(imgs_dir_path)

        # 最初のキーポイントが定まっていなかった時はconfidenceで判断
        if not exists_first_keypoints:
            print('{} ===================================='.format(json_dir))
            print('    method: confidence')
            f.write('{} ====================================\n'.format(json_dir))
            f.write('    method: confidence\n')

            max_confidence, max_confidence_idx = get_max_confidence_and_idx(json_dir_path)
            max_confidence_json_path = os.path.join(json_dir_path,
                                                    json_name_list[max_confidence_idx])
            max_confidence_keypoints = get_keypoints_array_from_json(max_confidence_json_path)

            max_image_name = imgs_name_list[max_confidence_idx]
            max_image_path = os.path.join(imgs_dir_path, max_image_name)
            if max_confidence_keypoints.any():
                exists_first_keypoints = True
                pre_keypoints_array = rotate_keypoints_array(max_confidence_keypoints,
                                                             max_confidence_idx * (-10),
                                                             rot_center_x=rot_center_x,
                                                             rot_center_y=rot_center_y)

                OUTPUT_PATH = os.path.join(CONFIDENCE_WITH_DIST_INFO, '{}.png'.format(imgs_dir))
                save_rotate_image(max_image_path, OUTPUT_PATH, max_confidence_idx * (-10))
            else:
                OUTPUT_PATH = os.path.join(CONFIDENCE_WITH_DIST_INFO, '{}.png'.format(imgs_dir))
                save_rotate_image(max_image_path, OUTPUT_PATH, 0)

            print('    confidence score: {}'.format(max_confidence))
            f.write('    confidence score: {}\n'.format(max_confidence))
        # 最初のキーポイントが定まっているときは時系列を考慮して
        else:
            euclidean_dist_list = np.array([])

            for i in range(len(json_name_list)):
                deg = i * DEG_SPLIT

                json_name = json_name_list[i]
                json_name_path = os.path.join(json_dir_path, json_name)

                keypoints_array = get_keypoints_array_from_json(json_name_path)

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

            print('{} ===================================='.format(json_dir))
            print('    method: confidence')
            f.write('{} ====================================\n'.format(json_dir))
            f.write('    method: confidence\n')

            max_confidence, max_confidence_idx = get_max_confidence_and_idx(json_dir_path)
            max_confidence_json_path = os.path.join(json_dir_path,
                                                    json_name_list[max_confidence_idx])
            max_confidence_keypoints = get_keypoints_array_from_json(max_confidence_json_path)

            max_image_name = imgs_name_list[max_confidence_idx]
            max_image_path = os.path.join(imgs_dir_path, max_image_name)
            if max_confidence_keypoints.any():
                exists_first_keypoints = True
                pre_keypoints_array = rotate_keypoints_array(max_confidence_keypoints,
                                                             max_confidence_idx * (-10),
                                                             rot_center_x=rot_center_x,
                                                             rot_center_y=rot_center_y)

                OUTPUT_PATH = os.path.join(CONFIDENCE_WITH_DIST_INFO, '{}.png'.format(imgs_dir))
                save_rotate_image(max_image_path, OUTPUT_PATH, max_confidence_idx * (-10))
            else:
                OUTPUT_PATH = os.path.join(CONFIDENCE_WITH_DIST_INFO, '{}.png'.format(imgs_dir))
                save_rotate_image(max_image_path, OUTPUT_PATH, 0)

            print('    confidence score: {}'.format(max_confidence))
            f.write('    confidence score: {}\n'.format(max_confidence))
            f.write('    dist: {}\n'.format(euclidean_dist_list[max_confidence_idx]))

    f.close()


if __name__ == '__main__':
    main()
