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

# do calibration
mkdir -p dataset
cp ../resnet18_in_f32.npz dataset/
run_calibration.py resnet18.mlir \
    --dataset dataset \
    --input_num 1 \
    -o resnet18_cali_table

# import calibration
sophgo-opt resnet18.mlir \
    --import-calibration-table='file=resnet18_cali_table asymmetric=false' \
    --save-weight \
    -o resnet18_cali.mlir

# quantize mlir
sophgo-opt resnet18_cali.mlir \
    --quantize="mode=INT8 asymmetric=false chip=bm1684" \
    --save-weight \
    -o resnet18_int8.mlir

# tpu weight reorder
sophgo-opt resnet18_int8.mlir \
    --weight-reorder \
    --save-weight \
    -o resnet18_int8_reorder.mlir

popd
