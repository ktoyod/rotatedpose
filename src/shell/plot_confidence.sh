NAME=""

DEG_SPLIT=10
DIST_THRESHOLD=500
W_CNT=80

OUTPUT_PATH="./openpose_output/${NAME}"
OPENPOSE_JSON_PATH="${OUTPUT_PATH}/raw_openpose/json"
CONFIDENCE_TEXT_PATH="${OUTPUT_PATH}/for_video_deg${DEG_SPLIT}_w_cnt${W_CNT}_dist${DIST_THRESHOLD}/confidence_mean.txt"
PLOT_OUTPUT_PATH="${OUTPUT_PATH}/for_video_deg${DEG_SPLIT}_w_cnt${W_CNT}_dist${DIST_THRESHOLD}"

python3 /path/to/visualize.py "$OPENPOSE_JSON_PATH" "$CONFIDENCE_TEXT_PATH" "$PLOT_OUTPUT_PATH"
