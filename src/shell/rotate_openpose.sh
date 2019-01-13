# ここは実行前に設定する
# name=hogehoge
NAME=""
# INPUT_PATH=openpose_input/hogehoge
INPUT_PATH="./openpose_input/$NAME"
INPUT_IMAGES_PATH="${INPUT_PATH%/}/images"
if [ ! -e $INPUT_IMAGES_PATH ]; then mkdir $INPUT_IMAGES_PATH ; fi
INPUT_ROTATED_PATH="${INPUT_IMAGES_PATH%/}/rotated_images"
# OUTPUT_PATH=openpose_output/hogehoge
OUTPUT_PATH="./openpose_output/$NAME"
if [ ! -e $OUTPUT_PATH ]; then mkdir $OUTPUT_PATH ; fi
OUTPUT_IMAGES_PATH="${OUTPUT_PATH%/}/images"
if [ ! -e $OUTPUT_IMAGES_PATH ]; then mkdir $OUTPUT_IMAGES_PATH ; fi
OUTPUT_JSON_PATH="${OUTPUT_PATH%/}/json"
if [ ! -e $OUTPUT_JSON_PATH ]; then mkdir $OUTPUT_JSON_PATH ; fi

# 動画のフレームレート
FRAME_RATE=30

# 動画を静止画に分割する
ffmpeg -i "${INPUT_PATH%/}/${NAME}.mp4" -r $FRAME_RATE "${INPUT_IMAGES_PATH%/}/image%06d.jpg"
 
# 回転で見切れないように画像を正方形にする
python3 /path/to/make_images_square.py $INPUT_IMAGES_PATH

# 画像を回転させる
python3 /path/to/rotate_images.py $INPUT_IMAGES_PATH

# 分割後の画像の枚数
IMAGE_NUM=`ls -l ${INPUT_ROTATED_PATH} | grep ^d | wc -l`

# 回転させた画像に対してOpenPoseをかけて出力を保存する
for i in `seq 1 $IMAGE_NUM`; do
if [ $i -lt 10 ]; then
  image_name="image00000$i"
elif [ $i -lt 100 ]; then
  image_name="image0000$i"
elif [ $i -lt 1000 ]; then
  image_name="image000$i"
else
  image_name="image00$i"
fi
  image_dir="${INPUT_ROTATED_PATH%/}/${image_name}"
  output_image_dir="${OUTPUT_IMAGES_PATH%/}/${image_name}"
  output_json_dir="${OUTPUT_JSON_PATH%/}/${image_name}"
  if [ -d $output_image_dir ]; then
    mkdir $output_image_dir
  fi
  if [ -d $output_json_dir ]; then
    mkdir $output_json_dir
  fi
  ./build/examples/openpose/openpose.bin --image_dir $image_dir --write_images $output_image_dir --write_json $output_json_dir --model_pose COCO --num_gpu_start 1 --display 0
done
