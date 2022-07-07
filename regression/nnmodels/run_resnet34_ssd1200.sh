#!/bin/bash
set -ex

mkdir -p resnet34_ssd1200
pushd resnet34_ssd1200

model_transform.py \
    --model_name resnet34_ssd1200 \
    --model_def  ${NNMODELS_PATH}/onnx_models/resnet34-ssd1200.onnx \
    --input_shapes [[1,3,1200,1200]] \
    --output_names Concat_470,Concat_471 \
    --mlir resnet34_ssd1200.mlir

popd
