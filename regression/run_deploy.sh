#!/bin/bash
set -ex

#########################
# transform to top mlir
#########################
INPUT=../resnet18_in_f32.npz
mkdir -p tmp
pushd tmp
model_transform.py \
    --model_type onnx \
    --model_name resnet18 \
    --input_shapes [[1,3,224,224]] \
    --model_def  ../resnet18.onnx \
    --test_input $INPUT \
    --test_result resnet18_top_outputs.npz \
    --mlir resnet18.mlir

#########################
# deploy to float bmodel
#########################
model_deploy.py \
  --mlir resnet18.mlir \
  --quantize F32 \
  --chip bm1686 \
  --test_input $INPUT \
  --test_reference resnet18_top_outputs.npz \
  --tolerance 0.99,0.99 \
  --model resnet18_1686_f32.bmodel

# model_deploy.py \
#   --mlir resnet18.mlir \
#   --quantize F16 \
#   --chip bm1686 \
#   --test_input $INPUT \
#   --test_reference resnet18_top_outputs.npz \
#   --tolerance 0.99,0.99 \
#   --model resnet18_1686_f16.bmodel

# model_deploy.py \
#   --mlir resnet18.mlir \
#   --quantize BF16 \
#   --chip bm1686 \
#   --test_input $INPUT \
#   --test_reference resnet18_top_outputs.npz \
#   --tolerance 0.99,0.98 \
#   --model resnet18_1686_bf16.bmodel

#########################
# deploy to int8 bmodel
#########################
mkdir -p dataset
cp $INPUT dataset/
# run calibration first
run_calibration.py resnet18.mlir \
    --dataset dataset \
    --input_num 1 \
    -o resnet18_cali_table

# to symmetric
# model_deploy.py \
#   --mlir resnet18.mlir \
#   --quantize INT8 \
#   --calibration_table resnet18_cali_table \
#   --chip bm1686 \
#   --tolerance 0.97,0.75 \
#   --model resnet18_1686_int8_sym.bmodel
#   --test_input $INPUT \
#   --test_reference resnet18_top_outputs.npz \

# to asymmetric
# model_deploy.py \
#   --mlir resnet18.mlir \
#   --quantize INT8 \
#   --asymmetric \
#   --calibration_table resnet18_cali_table \
#   --chip bm1686 \
#   --tolerance 0.97,0.75 \
#   --model resnet18_1686_int8_asym.bmodel
  # --test_input $INPUT \
  # --test_reference resnet18_top_outputs.npz \

popd
