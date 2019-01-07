import json
import os
import sys

import numpy as np

args = sys.argv
JSON_DIR = args[1] if len(args) == 2 else './'


def calc_confidence(keypoints_list):
    keypoints_array = np.array(keypoints_list)
    confidence_array = keypoints_array[2::3]
    return np.mean(confidence_array)


json_list = os.listdir(JSON_DIR)
json_list.sort()
json_list = [json_file for json_file in json_list if 'json' in json_file]

confidence_list = []
for file_name in json_list:
    with open(f'{JSON_DIR}/{file_name}') as f:
        json_dic = json.load(f)
    if json_dic['people']:
        keypoints_list = json_dic['people'][0]['pose_keypoints_2d']
        confidence_list.append(calc_confidence(keypoints_list))
    else:
        confidence_list.append(None)

with open(f'{JSON_DIR}/confidence-list.txt', 'w') as f:
    for i, confidence_score in enumerate(confidence_list):
        f.write(f'{i * 10}: {confidence_score}\n')
