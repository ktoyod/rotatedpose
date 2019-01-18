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
    MAX_DIST = os.getenv('DIST_THRESHOLD')
    TIME_AND_CONFIDENCE_PATH = os.path.join(BASE_PATH,
                                            'for_time_and_confidence_video_{}'.format(MAX_DIST))
    if not os.path.isdir(TIME_AND_CONFIDENCE_PATH):
        os.mkdir(TIME_AND_CONFIDENCE_PATH)

    log_file_path = os.path.join(TIME_AND_CONFIDENCE_PATH, 'log.txt')
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

                OUTPUT_PATH = os.path.join(TIME_AND_CONFIDENCE_PATH, '{}.png'.format(imgs_dir))
                save_rotate_image(max_image_path, OUTPUT_PATH, max_confidence_idx * (-10))
            else:
                OUTPUT_PATH = os.path.join(TIME_AND_CONFIDENCE_PATH, '{}.png'.format(imgs_dir))
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

            if np.min(euclidean_dist_list) > MAX_DIST:
                print('{} ===================================='.format(json_dir))
                print('    method: confidence')
                f.write('{} ====================================\n'.format(json_dir))
                f.write('    method: confidence\n')
                f.write('    dist: {}\n'.format(np.min(euclidean_dist_list)))

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

                    OUTPUT_PATH = os.path.join(TIME_AND_CONFIDENCE_PATH, '{}.png'.format(imgs_dir))
                    save_rotate_image(max_image_path, OUTPUT_PATH, max_confidence_idx * (-10))
                else:
                    OUTPUT_PATH = os.path.join(TIME_AND_CONFIDENCE_PATH, '{}.png'.format(imgs_dir))
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
                nearest_image_name = imgs_name_list[nearest_idx]
                nearest_image_path = os.path.join(imgs_dir_path, nearest_image_name)

                nearest_json_path = os.path.join(json_dir_path, json_name_list[nearest_idx])
                nearest_keypoints_array = get_keypoints_array_from_json(nearest_json_path)
                pre_keypoints_array = rotate_keypoints_array(nearest_keypoints_array,
                                                             nearest_idx * (-10),
                                                             rot_center_x=rot_center_x,
                                                             rot_center_y=rot_center_y)

                OUTPUT_PATH = os.path.join(TIME_AND_CONFIDENCE_PATH, '{}.png'.format(imgs_dir))
                save_rotate_image(nearest_image_path, OUTPUT_PATH, nearest_idx * (-10))

    f.close()


if __name__ == '__main__':
    main()
