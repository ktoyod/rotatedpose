NAME=""
MAX_DIST=500
OUTPUT_PATH="./openpose_output/${NAME}"
OPENPOSE_JSON_PATH="${OUTPUT_PATH}/raw_openpose/json"
CONFIDENCE_TEXT_PATH="${OUTPUT_PATH}/for_confidence_with_dist_draw_without_face_video_${MAX_DIST}/confidence_ave_without_face_${MAX_DIST}.txt"
PLOT_OUTPUT_PATH="${OUTPUT_PATH}/for_confidence_with_dist_draw_without_face_video_${MAX_DIST}"

python3 /path/to/visualize.py "$OPENPOSE_JSON_PATH" "$CONFIDENCE_TEXT_PATH" "$PLOT_OUTPUT_PATH"
