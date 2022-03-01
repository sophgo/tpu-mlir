#!/bin/bash
set -ex
mkdir -p tmp
pushd tmp
model_transform.py \
    --model_type onnx \
    --model_name resnet18 \
    --model_def  ../resnet18.onnx \
    --tolerance 0.99,0.99,0.96 \
    --mlir resnet18.mlir

tpuc-opt resnet18.mlir \
    --canonicalize \
    -o resnet18_opt.mlir

model_runner.py \
    --input ../resnet18_in_fp32.npz \
    --model resnet18_opt.mlir \
    --output resnet18_out.npz

npz_tool.py compare resnet18_blobs.npz resnet18_out.npz

popd
