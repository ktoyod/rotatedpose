import os
import sys

import numpy as np

from openposedraw import draw_joints
from select_high_confidence_images import (get_max_confidence_and_idx_without_untrusted_joints,
                                           save_rotate_image,
                                           create_confidence_array_without_untrusted_joints)
from utils.file import get_keypoints_array_from_json, make_list_in_dir
from utils.rotate import rotate_keypoints_array, get_rot_center
from utils.distance import euclidean_distance


def main():
    args = sys.argv

    BASE_PATH = args[1]
    JSON_PATH = os.path.join(BASE_PATH, 'json')
    IMGS_PATH = os.path.join(BASE_PATH, 'images')
    DEG_SPLIT = os.getenv('DEG_SPLIT')
    MAX_DIST = os.getenv('DIST_THRESHOLD')
    TIME_AND_CONFIDENCE_DRAW_PATH = os.path.join(
                                        BASE_PATH,
                                        'for_confidence_with_dist_draw_trusted_video_{}'
                                        .format(MAX_DIST)
                                    )
    if not os.path.isdir(TIME_AND_CONFIDENCE_DRAW_PATH):
        os.mkdir(TIME_AND_CONFIDENCE_DRAW_PATH)
    CONFIDENCE_AVE_TEXT_PATH = os.path.join(TIME_AND_CONFIDENCE_DRAW_PATH,
                                            'confidence_ave_{}.txt'.format(MAX_DIST))

    INPUT_IMAGES_PATH = args[2]
    input_imgs_list = make_list_in_dir(INPUT_IMAGES_PATH)
    input_imgs_list = [img for img in input_imgs_list if 'jpg' in img]

    log_file_path = os.path.join(TIME_AND_CONFIDENCE_DRAW_PATH,
                                 'log_confidence_with_dist_draw.txt')
    f = open(log_file_path, 'w')
    confidence_f = open(CONFIDENCE_AVE_TEXT_PATH, 'w')

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

            max_confidence, max_confidence_idx = \
                get_max_confidence_and_idx_without_untrusted_joints(json_dir_path)
            max_confidence_json_path = os.path.join(json_dir_path,
                                                    json_name_list[max_confidence_idx])
            max_confidence_keypoints = get_keypoints_array_from_json(max_confidence_json_path)

            max_image_name = imgs_name_list[max_confidence_idx]
            max_image_path = os.path.join(imgs_dir_path, max_image_name)

            confidence_f.write(str(max_confidence) + '\n')
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

                valid_euclidean_dist_list = np.array([
                    dist for dist in euclidean_dist_list if dist <= MAX_DIST])
                valid_dist_len = len(valid_euclidean_dist_list)
                max_len = 5
                valid_dist_len = valid_dist_len if valid_dist_len <= max_len else max_len

            if valid_dist_len == 0:
                print('{} ===================================='.format(json_dir))
                print('    method: confidence')
                f.write('{} ====================================\n'.format(json_dir))
                f.write('    method: confidence\n')
                f.write('    dist: {}\n'.format(np.min(euclidean_dist_list)))

                max_confidence, max_confidence_idx = \
                    get_max_confidence_and_idx_without_untrusted_joints(json_dir_path)
                max_confidence_json_path = os.path.join(json_dir_path,
                                                        json_name_list[max_confidence_idx])
                max_confidence_keypoints = get_keypoints_array_from_json(max_confidence_json_path)

                max_image_name = imgs_name_list[max_confidence_idx]
                max_image_path = os.path.join(imgs_dir_path, max_image_name)

                confidence_f.write(str(max_confidence) + '\n')

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
                print('    method: time series and confidence')
                f.write('{} ====================================\n'.format(json_dir))
                f.write('    method: time series and confidence\n')

                confidence_array = create_confidence_array_without_untrusted_joints(json_dir_path)

                argsort_idx_list = np.argsort(euclidean_dist_list)
                argsort_idx_list = argsort_idx_list[:valid_dist_len]

                best_idx = 0
                max_confidence = -1
                for argsort_idx in argsort_idx_list:
                    f.write('    idx - {}:\n'.format(argsort_idx))
                    f.write('        dist -> {}\n'.format(euclidean_dist_list[argsort_idx]))
                    f.write('        confidence -> {}\n'.format(confidence_array[argsort_idx]))
                    if confidence_array[argsort_idx] > max_confidence:
                        best_idx = argsort_idx
                        max_confidence = confidence_array[argsort_idx]

                confidence_f.write(str(max_confidence) + '\n')

                best_json_path = os.path.join(json_dir_path, json_name_list[best_idx])
                best_keypoints_array = get_keypoints_array_from_json(best_json_path)
                pre_keypoints_array = rotate_keypoints_array(best_keypoints_array,
                                                             best_idx * 10,
                                                             rot_center_x=rot_center_x,
                                                             rot_center_y=rot_center_y)
                reshaped_pre_keypoints_array = pre_keypoints_array.reshape([18, 3])

                OUTPUT_PATH = os.path.join(TIME_AND_CONFIDENCE_DRAW_PATH, '{}.png'.format(imgs_dir))
                draw_joints(input_img_path, OUTPUT_PATH, reshaped_pre_keypoints_array)

    f.close()
    confidence_f.close()


if __name__ == '__main__':
    main()
