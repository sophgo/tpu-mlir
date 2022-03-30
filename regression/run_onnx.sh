#!/bin/bash
set -ex

INPUT=../resnet18_in_f32_b4.npz
mkdir -p tmp
pushd tmp
model_transform.py \
    --model_type onnx \
    --model_name resnet18 \
    --input_shapes [[4,3,224,224]] \
    --model_def  ../resnet18.onnx \
    --input $INPUT \
    --mlir resnet18.mlir

# do calibration
mkdir -p dataset
cp $INPUT dataset/
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

model_runner.py --model resnet18_int8.mlir --input $INPUT --dump_all_tensors --output resnet18_int8_outputs.npz
#npz_tool.py compare resnet18_int8_outputs.npz resnet18_ref_outputs.npz -v

# tpu weight reorder
sophgo-opt resnet18_int8.mlir \
    --weight-reorder \
    --save-weight \
    -o resnet18_int8_reorder.mlir

# tpu divide to subnets
sophgo-opt resnet18_int8_reorder.mlir \
    --subnet-divide \
    --save-weight \
    -o resnet18_int8_divide.mlir

# tpu address asign
sophgo-opt resnet18_int8_divide.mlir \
    --address-asign \
    --save-weight \
    -o resnet18_int8_addr.mlir

sophgo-opt resnet18_int8_addr.mlir \
    --codegen="model_file=resnet18_int8.bmodel" > /dev/null

popd
