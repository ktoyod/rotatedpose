import sys
from os import mkdir
from os.path import isdir, join

import numpy as np
from PIL import Image

from openposedraw import draw_joints_on_image
from settings import Settings
from utils.confidence import get_confidence_and_idx
from utils.file import get_keypoints_array_from_json, make_list_in_dir
from utils.rotate import get_rot_center, rotate_keypoints_array, save_rotate_image
from utils.smoothing import smoothing


def get_rot_center_from_path(imgs_path, imgs_dir_list):
    imgs_dir_path = join(imgs_path, imgs_dir_list[0])
    imgs_name_list = make_list_in_dir(imgs_dir_path)
    img_path = join(imgs_dir_path, imgs_name_list[0])
    rot_center_x, rot_center_y = get_rot_center(img_path)

    return rot_center_x, rot_center_y


def main():
    '''
    OpenPoseのconfidenceが最も高いものを
    最適なjointとして選択する
    '''
    args = sys.argv

    DEG_SPLIT = Settings.DEG_SPLIT
    IS_SELF_DRAWING = Settings.IS_SELF_DRAWING
    IS_SMOOTHED = Settings.IS_SMOOTHED
    W_CNT = Settings.W_CNT if IS_SMOOTHED else 1

    # ベースとなるパス
    BASE_PATH = args[1]
    JSON_PATH = join(BASE_PATH, 'json')
    IMGS_PATH = join(BASE_PATH, 'images')
    FOR_VIDEO_PATH = join(BASE_PATH,
                          'for_video_deg{}_w_cnt{}_confidence'
                          .format(DEG_SPLIT, int(W_CNT * 100)))
    if not isdir(FOR_VIDEO_PATH):
        mkdir(FOR_VIDEO_PATH)

    # ファイルのパス
    LOG_FILE_PATH = join(FOR_VIDEO_PATH, 'log.txt')
    ANGLES_FILE_PATH = join(FOR_VIDEO_PATH, 'angles.txt')
    CONFIDENCE_MEAN_TEXT_PATH = join(FOR_VIDEO_PATH, 'confidence_mean.txt')
    # ファイルを開く
    log_f = open(LOG_FILE_PATH, 'w')
    angles_f = open(ANGLES_FILE_PATH, 'w')
    confidence_f = open(CONFIDENCE_MEAN_TEXT_PATH, 'w')

    # 元画像のパス
    INPUT_IMAGES_PATH = args[2]
    input_imgs_list = make_list_in_dir(INPUT_IMAGES_PATH, expanded='jpg')

    # OpenPoseの結果jsonと画像が入っているディレクトリの名前のリスト
    # image000001, image000002, ...
    json_dir_list = make_list_in_dir(JSON_PATH)
    imgs_dir_list = make_list_in_dir(IMGS_PATH)

    # 画像の中心を計算
    rot_center_x, rot_center_y = get_rot_center_from_path(IMGS_PATH, imgs_dir_list)

    for json_dir, imgs_dir, input_img in zip(json_dir_list, imgs_dir_list, input_imgs_list):
        # OpenPoseの結果jsonと画像が格納されているディレクトリ
        json_dir_path = join(JSON_PATH, json_dir)
        imgs_dir_path = join(IMGS_PATH, imgs_dir)

        # jsonと画像のファイル名を格納したリスト
        # 10度ごとに回転させたものが入っている
        # image000001_rotate000_keypoints.json, image000001_rotate010_keypoints.json, ...
        json_name_list = make_list_in_dir(json_dir_path)
        imgs_name_list = make_list_in_dir(imgs_dir_path)

        input_img_path = join(INPUT_IMAGES_PATH, input_img)

        output_img_path = join(FOR_VIDEO_PATH, '{}.png'.format(imgs_dir))

        log_f.write('{} ====================================\n'.format(json_dir))
        log_f.write('    method: confidence\n')

        # confidenceの最大とそのインデックスから
        # confidence最大の時のキーポイントのnp.arrayを得る
        max_confidence, max_confidence_idx = get_confidence_and_idx(json_dir_path)
        max_confidence_json_path = join(json_dir_path, json_name_list[max_confidence_idx])
        max_confidence_keypoints = get_keypoints_array_from_json(max_confidence_json_path)

        pre_keypoints_array = np.array([])

        confidence_f.write(str(max_confidence) + '\n')

        if max_confidence_keypoints.any():
            deg = max_confidence_idx * DEG_SPLIT

            cnt_keypoints_array = rotate_keypoints_array(max_confidence_keypoints,
                                                         deg,
                                                         rot_center_x=rot_center_x,
                                                         rot_center_y=rot_center_y)
            if IS_SMOOTHED and pre_keypoints_array.any():
                cnt_keypoints_array = smoothing(cnt_keypoints_array, pre_keypoints_array, W_CNT)

            reshaped_pre_keypoints_array = cnt_keypoints_array.reshape([18, 3])
            pre_keypoints_array = cnt_keypoints_array

            if IS_SELF_DRAWING:
                draw_joints_on_image(input_img_path, output_img_path,
                                     reshaped_pre_keypoints_array)
            else:
                max_image_name = imgs_name_list[max_confidence_idx]
                max_image_path = join(imgs_dir_path, max_image_name)
                save_rotate_image(max_image_path, output_img_path, deg)
        else:
            img = Image.open(input_img_path)
            img.save(output_img_path)

        log_f.write('    confidence score: {}\n'.format(max_confidence))
        angles_f.write(str(deg) + '\n')

    log_f.close()
    angles_f.close()
    confidence_f.close()


if __name__ == '__main__':
    main()
