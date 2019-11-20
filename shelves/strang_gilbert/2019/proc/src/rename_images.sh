# Path to this script:
SCRIPT_PATH=$(realpath -s "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
# Recall: readlink resolves relative paths, realpath finds relative paths
IMG_SOURCE_DIR=$(readlink -f "$SCRIPT_DIR/../img/")

PAGE_FILES=($IMG_SOURCE_DIR/20191120_*.jpg) # Array of page files
START_PAGE=420 # Offset for renaming the files

for i in ${!PAGE_FILES[@]}; do
  PAGE_NUM=$(echo "$START_PAGE + $i" | bc)
  FILE_PATH="${PAGE_FILES[$i]}"
  FILE_NAME="$(basename $FILE_PATH)"
  echo "Moving $FILE_NAME" to "$IMG_SOURCE_DIR/$PAGE_NUM.jpg"
  mv "$FILE_PATH" "$IMG_SOURCE_DIR/$PAGE_NUM.jpg"
done
