# rotated-pose

# ディレクトリ構成
- image\_dir: OpenPoseの入力画像
  - rotated\_images: image\_dir直下の画像を回転させた画像。各フレームに対してサブディレクトリが存在しそのサブディレクトリ内に各回転角で回転させた画像が格納される
- openpose\_output: OpenPoseの出力
  - for\_video: 最終的に動画にする画像と完成した動画
  - images: openposeの出力画像
  - json: openposeの出力json
    - 上記２つは各フレームごとのサブディレクトリが存在しそのサブディレクトリ内に各回転角に対する結果ファイルが格納される
```
image_dir
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

# 環境変数

| 環境変数名 | 意味 |
|:-:|:-:|
| DIST\_THRESHOLD | フレーム間のジョイント距離の閾値 |
| W\_CNT | 平滑化の際の現在のフレームに対する重み |
