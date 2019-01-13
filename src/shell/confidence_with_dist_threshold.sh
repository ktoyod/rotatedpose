# ノイズなし
FRAME_RATE=30
# フレーム間のジョイントの距離の閾値
MAX_DIST=500

NAME=""
OUTPUT_PATH="./openpose_output/$NAME"
OUTPUT_TIME_AND_CONFIDENCE_VIDEO_PATH="${OUTPUT_PATH%/}/for_confidence_with_dist_draw_without_face_and_noise_video_${MAX_DIST}"
INPUT_IMAGES_PATH="./openpose_input/$NAME/images"

python3 /path/to/confidence_with_dist_draw_without_face_and_noise.py "$OUTPUT_PATH" "$INPUT_IMAGES_PATH"
ffmpeg -y -r $FRAME_RATE -i "${OUTPUT_TIME_AND_CONFIDENCE_VIDEO_PATH%/}/image%6d.png" -c:v libx264 -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" "${OUTPUT_TIME_AND_CONFIDENCE_VIDEO_PATH%/}/${NAME}_confidence_with_dist_draw_withour_face_and_noise_${MAX_DIST}.mp4"
