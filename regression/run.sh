#!/bin/bash
set -ex
rm -rf tmp
./run_deploy.sh
./run_onnx.sh
./run_tflite.sh

if [ -d ${PROJECT_ROOT}/../nnmodels/ ]; then
  ./run_mobilenet_onnx.sh
  ./run_resnet50-v1_onnx.sh
else
  echo "[Warning] nnmodles does not exist; Skip some tests."
fi
