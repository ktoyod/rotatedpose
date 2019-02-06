# 動画の名前
NAME=""

# INPUT_PATH=openpose_input/hogehoge
INPUT_PATH="./openpose_input/$NAME"
# INPUT_VIDEO_PATH="${INPUT_PATH%/}/${NAME}.mp4"
INPUT_IMAGES_DIR="${INPUT_PATH}/images"
# OUTPUT_PATH=openpose_output/hogehoge
OUTPUT_PATH="./openpose_output/$NAME"
if [ ! -e $OUTPUT_PATH ]; then mkdir $OUTPUT_PATH ; fi
OUTPUT_RAW_PATH="${OUTPUT_PATH%/}/raw_openpose"
if [ ! -e $OUTPUT_RAW_PATH ]; then mkdir $OUTPUT_RAW_PATH ; fi
OUTPUT_JSON_PATH="${OUTPUT_RAW_PATH%/}/json"
if [ ! -e $OUTPUT_JSON_PATH ]; then mkdir $OUTPUT_JSON_PATH ; fi
OUTPUT_IMAGES_PATH="${OUTPUT_RAW_PATH%/}/images"
if [ ! -e $OUTPUT_IMAGES_PATH ]; then mkdir $OUTPUT_IMAGES_PATH ; fi
OUTPUT_VIDEO_PATH="${OUTPUT_RAW_PATH%/}/${NAME}_raw_openpose.mp4"

FRAME_RATE=30

./build/examples/openpose/openpose.bin --image_dir $INPUT_IMAGES_DIR --write_json $OUTPUT_JSON_PATH --write_images $OUTPUT_IMAGES_PATH --model_pose COCO --num_gpu_start 1 --display 0

ffmpeg -y -r $FRAME_RATE -i "${OUTPUT_IMAGES_PATH%/}/image%6d_rendered.png" -c:v libx264 -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" "${OUTPUT_VIDEO_PATH}"
