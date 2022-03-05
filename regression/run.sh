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

#calibration
mkdir -p dataset
cp ../resnet18_in_f32.npz dataset/
run_calibration.py resnet18.mlir \
    --dataset dataset \
    --input_num 1 \
    -o resnet18_cali_table

popd
