import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from utils.confidence import create_confidence_array


def get_angles_list(angles_file_path):
    with open(angles_file_path, 'r') as f:
        angles_list = f.readlines()
        angles_list = [angle.strip() for angle in angles_list]
        angles_list = list(map(int, angles_list))

    return angles_list


def get_openpose_confidence_list(openpose_json_path):
    openpose_confidence_array = create_confidence_array(openpose_json_path)

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


def save_angles_plot(angles_file_path, output_path, start_idx=0, stop_idx=None, file_name=None):
    angles_list = get_angles_list(angles_file_path)
    angles_list = angles_list[start_idx:stop_idx] if stop_idx else angles_list[start_idx:]
    x_list = list(range(1, len(angles_list) + 1))

    plt.plot(x_list, angles_list)
    plt.xlabel('frame in sequence')
    plt.ylabel('angle')
    plt.xlim(1, len(x_list))
    plt.ylim(0, 360)

    file_name = file_name if file_name else 'angles.png'
    plt.savefig(os.path.join(output_path, file_name))


def save_angles_sin_plot(angles_file_path, output_path):
    angles_list = get_angles_list(angles_file_path)
    angles_sin_list = [np.sin(np.radians(angle)) for angle in angles_list]
    x_list = list(range(1, len(angles_list) + 1))

    plt.plot(x_list, angles_sin_list)
    plt.xlabel('frame in sequence')
    plt.ylabel('sin')
    plt.xlim(1, len(x_list))
    plt.ylim(-1.0, 1.0)

    plt.savefig(os.path.join(output_path, 'angles_sin.png'))


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
