import os


class Path(object):

    CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))

    # リポジトリがOpenPose直下に置かれる
    OPENPOSE_PATH = os.path.normpath(os.path.join(CURRENT_PATH, os.pardir))
