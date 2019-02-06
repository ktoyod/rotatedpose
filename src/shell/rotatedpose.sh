NAME=""

FRAME_RATE=30

DEG_SPLIT=10
DIST_THRESHOLD=500
W_CNT=80

OUTPUT_PATH="./openpose_output/$NAME"
FOR_VIDEO_PATH="${OUTPUT_PATH%/}/for_video_deg${DEG_SPLIT}_w_cnt${W_CNT}_dist${DIST_THRESHOLD}"
INPUT_IMAGES_PATH="./openpose_input/$NAME/images"

python3 /path/to/rotatedpose.py "$OUTPUT_PATH" "$INPUT_IMAGES_PATH"
ffmpeg -y -r $FRAME_RATE -i "${FOR_VIDEO_PATH%/}/image%6d.png" -c:v libx264 -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" "${FOR_VIDEO_PATH%/}/${NAME}_deg${DEG_SPLIT}_w_cnt${W_CNT}_dist${DIST_THRESHOLD}.mp4"
