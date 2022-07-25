#!/bin/bash
set -ex

if [ ! -d ${MODEL_ZOO_PATH} ] && [ ! -d ${NNMODELS_PATH} ]; then
  echo "[Warning] model-zoo does not exist; Skip model-zoo tests."
  exit 0
fi

DIR="$( cd "$(dirname "$0")" ; pwd -P )"

$DIR/run_mobilenet_v2.sh
$DIR/run_resnet50_v2.sh
$DIR/run_vgg16-12.sh
#$DIR/run_resnet34_ssd1200.sh
#$DIR/run_yolov5s.sh

