# rotatedpose
- paper -> [Post-Data Augmentation to Improve Deep Pose Estimation of Extreme and Wild Motions - arxiv](https://arxiv.org/abs/1902.04250)

## How to use
```bash
$ git clone https://github.com/ktoyod/rotatedpose.git

$ cp rotatedpose/shell/*.sh /path/to/openpose
# rotate_openpose.shの"NAME"、pathは自分の環境に合わせて変更する

$ cd /path/to/openpose
$ mkdir -p openpose_input/movie_name
# 上記で作成した'openpose_input/movie_name'に'movie_name.mp4'を置く
$ mkdir oprnpose_output

$ ./rotate_openpose.sh  # 画像を回転させてOpenPoseかける
$ ./rotatedpose.sh      # 時系列とconfidenceを考慮してフレーム選択して動画作成
$ ./raw_openpose.sh     # rotaedpose.shと同様の条件でOpenPoseかけて自前で描画
$ ./plot_confidence.sh  # confidenceのグラフを作成
```

## ディレクトリ構成(入出力データ)
- image\_dir: OpenPoseの入力画像
  - rotated\_images: image\_dir直下の画像を回転させた画像。各フレームに対してサブディレクトリが存在しそのサブディレクトリ内に各回転角で回転させた画像が格納される
- openpose\_output: OpenPoseの出力
  - for\_video: 最終的に動画にする画像と完成した動画
  - images: openposeの出力画像
  - json: openposeの出力json
    - 上記２つは各フレームごとのサブディレクトリが存在しそのサブディレクトリ内に各回転角に対する結果ファイルが格納される
```
openpose_output
 |
movie_name
 ├── movie_name.mp4
 └── images
     ├── image000001.jpg
     ├── image000002.jpg
     ├── ...
     ├── image000xxx.jpg
     └── rotated_images
         ├── image000001
         │   ├── image000001_rotate000.jpg
         │   ├── image000001_rotate010.jpg
         │   ├── ...
         │   └── image000001_rotate350.jpg
         ├── image000002
         ├── ...
         └── image000xxx

openpose_output
|
movie_name
├── for_video
│   ├── image000001.png
│   ├── image000002.png
│   ├── ...
│   ├── image000xxx.png
│   └── rotated_video.mp4
├── images
│   ├── image000001
|   |   ├── image000001_rotate000_rendered.png
|   │   ├── image000001_rotate010_rendered.png
|   │   ├── ...
|   │   └── image000001_rotate350_rendered.png
│   ├── image000002
│   ├──...
│   └── image000xxx
└── json
   ├── image000001
   |   ├── image000001_rotate000_keypoints.json
   │   ├── image000001_rotate010_keypoints.json
   │   ├── ...
   │   └── image000001_rotate350_keypoints.json
   ├── image000002
   ├── ...
   └── image000xxx
```

## 環境変数

| 環境変数名 | 意味 |
|:-:|:-:|
| DIST\_THRESHOLD | フレーム間のジョイント距離の閾値 |
| W\_CNT | 平滑化の際の現在のフレームに対する重み |
| DEG\_SPLIT | 角度の刻み幅 |
| MAX\_NUM\_IN\_THRESHOLD | 閾値内で考慮する個数 |
| IS\_SMOOTHED | 平滑化するか |
| IS\_SELF\_DRAWING | jsonを自分で画像に描画するか |
