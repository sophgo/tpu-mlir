#!/bin/bash
set -ex

mkdir -p yolov5s
pushd yolov5s

model_transform.py \
    --model_name yolov5s \
    --model_def  ${NNMODELS_PATH}/onnx_models/yolov5s.onnx \
    --input_shapes [[1,3,640,640]] \
    --output_names 397,458,519 \
    --mlir yolov5s.mlir

popd
