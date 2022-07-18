#!/bin/bash
set -ex

if [ ! -d ${NNMODELS_PATH} ]; then
  echo "[Warning] nnmodles does not exist; Skip nnmodels tests."
  exit 0
fi

DIR="$( cd "$(dirname "$0")" ; pwd -P )"

$DIR/run_mobilenet_v2.sh
$DIR/run_resnet50_v2.sh
#$DIR/run_resnet34_ssd1200.sh
#$DIR/run_yolov5s.sh
