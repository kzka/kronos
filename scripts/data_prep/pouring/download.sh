#!/bin/bash

# Example:
#
#   download.sh ./tmp
#
# will download all of the pouring files into "./tmp".

ARGC="$#"
OUTPUT_DIR="./multiview-pouring"
if [ "${ARGC}" -ge 1 ]; then
  OUTPUT_DIR=$1
fi

echo "OUTPUT_DIR=$OUTPUT_DIR"

mkdir "${OUTPUT_DIR}"

function download_file {
  ID=$1
  FILE=multiview-pouring.tgz${ID}
  BUCKET="https://storage.googleapis.com/brain-robotics-data/pouring-multiview/${FILE}"
  URL="${BUCKET}"
  OUTPUT_FILE="${OUTPUT_DIR}/${FILE}"
  DIRECTORY=`dirname ${OUTPUT_FILE}`
  echo "downloading $URL to $OUTPUT_FILE"
  mkdir -p "${DIRECTORY}"
  curl --output ${OUTPUT_FILE} ${URL}
}

# Download
for i in `seq -f "%02g" 0 19`; do
  download_file $i
done

# Unpack
echo "Unpacking data..."
cd $OUTPUT_DIR
cat multiview-pouring.tgz* | tar xz
