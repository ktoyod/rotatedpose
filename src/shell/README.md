## Shell Scripts
- OpenPoseをかける処理があるスクリプトに関しては`openpose/`直下に置く

## 各ファイルの説明
- `rotate\_openpose.sh`: 動画を分割、回転してOpenPoseをかける(images / jsonができる)
- `rotatedpose.sh`: 前後フレームでのジョイントの距離、ジョイントのconfidenceを考慮した上で最もよいポーズを選択し、動画を作成
- `raw_openpose.sh`: 元動画に対してOpenPoseをかける
- `plot\_confidence.sh`: 各フレームにおけるジョイントのconfidenceをプロットする
