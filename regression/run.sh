#!/bin/bash
set -ex
mkdir -p tmp
pushd tmp
model_transform.py \
    --model_type onnx \
    --model_name resnet18 \
    --model_def  ../resnet18.onnx \
    --input ../resnet18_in_f32.npz \
    --mlir resnet18.mlir

popd
