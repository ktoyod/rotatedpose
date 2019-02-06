import os


class Settings(object):
    '''環境変数を管理するクラス'''

    DIST_THRESHOLD = os.getenv('DIST_THRESHOLD', 500)
    W_CNT = os.getenv('W_CNT', 0.8)
    DEG_SPLIT = os.getenv('DEG_SPLIT', 10)
    MAX_NUM_IN_THRESHOLD = os.getenv('MAX_NUM_IN_THRESHOLD', 5)
    IS_SMOOTHED = os.getenv('IS_SMOOTHED', True)
    IS_SELF_DRAWING = os.getenv('IS_SELF_DRAWING', True)
