def smoothing(keypoints_array, pre_keypoints_array, w_cnt=0.8):
    # weight for current and previous frame
    w_cnt = w_cnt
    w_pre = 1.0 - w_cnt

    # pose reconstruction
    reconst_keypoints_array = pre_keypoints_array * w_pre + keypoints_array * w_cnt

    return reconst_keypoints_array
