import os
import sys

import matplotlib.pyplot as plt

from select_high_confidence_images import create_confidence_array_without_face_joints


def plot_confidence():
    pass


def get_openpose_confidence_list(openpose_json_path):
    openpose_confidence_array = create_confidence_array_without_face_joints(openpose_json_path)

    return list(openpose_confidence_array)


def get_our_method_confidence_list(confidence_text_path):
    with open(confidence_text_path, 'r') as f:
        our_method_confidence_str_list = f.readlines()
        our_method_confidence_str_list = [conf.strip('')
                                          for conf in our_method_confidence_str_list]
        our_method_confidence_list = list(map(float, our_method_confidence_str_list))

    return our_method_confidence_list


def save_confidence_plot(openpose_confidence_list, our_method_confidence_list, output_path):
    x_list = list(range(1, len(openpose_confidence_list) + 1))

    plt.plot(x_list, openpose_confidence_list, label='openpose')
    plt.plot(x_list, our_method_confidence_list, label='our_method')
    plt.xlabel('frame in sequence')
    plt.ylabel('mean of confidence')
    plt.xlim(1, len(x_list))
    plt.ylim(0.0, 1.0)

    plt.legend()
    plt.savefig(os.path.join(output_path, 'confidence.png'))


def main():
    args = sys.argv

    OPENPOSE_JSON_DIR_PATH = args[1]  # 普通にOpenPoseかけた結果のjsonが格納されたディレクトリ
    CONFIDENCE_TEXT_PATH = args[2]  # 提案手法によるconfidenceが格納されたテキスト
    OUTPUT_PATH = args[3]

    openpose_confidence_list = get_openpose_confidence_list(OPENPOSE_JSON_DIR_PATH)
    our_method_confidence_list = get_our_method_confidence_list(CONFIDENCE_TEXT_PATH)

    save_confidence_plot(openpose_confidence_list, our_method_confidence_list, OUTPUT_PATH)


if __name__ == '__main__':
    main()
