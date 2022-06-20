#!/bin/bash
set -ex

#########################
# transform to top mlir
#########################
mkdir -p tmp
pushd tmp
model_transform.py \
  --model_type onnx \
  --model_name resnet18 \
  --model_def  ../resnet18.onnx \
  --input_shapes [[1,3,224,224]] \
  --resize_dims 256,256 \
  --mean 123.675,116.28,103.53 \
  --scale 0.0171,0.0175,0.0174 \
  --pixel_format rgb \
  --test_input ../image/cat.jpg \
  --test_result resnet18_top_outputs.npz \
  --mlir resnet18.mlir

#########################
# deploy to float bmodel
#########################
model_deploy.py \
  --mlir resnet18.mlir \
  --quantize F32 \
  --chip bm1684x \
  --test_input resnet18_in_f32.npz \
  --test_reference resnet18_top_outputs.npz \
  --tolerance 0.99,0.99 \
  --model resnet18_1684x_f32.bmodel

# model_deploy.py \
#   --mlir resnet18.mlir \
#   --quantize F16 \
#   --chip bm1684x \
#   --test_input resnet18_in_f32.npz \
#   --test_reference resnet18_top_outputs.npz \
#   --tolerance 0.99,0.99 \
#   --model resnet18_1684x_f16.bmodel

# model_deploy.py \
#   --mlir resnet18.mlir \
#   --quantize BF16 \
#   --chip bm1684x \
#   --test_input resnet18_in_f32.npz \
#   --test_reference resnet18_top_outputs.npz \
#   --tolerance 0.99,0.98 \
#   --model resnet18_1684x_bf16.bmodel

#########################
# deploy to int8 bmodel
#########################
run_calibration.py resnet18.mlir \
  --dataset ../image \
  --input_num 2 \
  -o resnet18_cali_table

# to symmetric
model_deploy.py \
  --mlir resnet18.mlir \
  --quantize INT8 \
  --calibration_table resnet18_cali_table \
  --chip bm1684x \
  --test_input resnet18_in_f32.npz \
  --test_reference resnet18_top_outputs.npz \
  --tolerance 0.98,0.85 \
  --correctness 0.99,0.95 \
  --model resnet18_1684x_int8_sym.bmodel

# to asymmetric
model_deploy.py \
  --mlir resnet18.mlir \
  --quantize INT8 \
  --asymmetric \
  --calibration_table resnet18_cali_table \
  --chip bm1684x \
  --test_input resnet18_in_f32.npz \
  --test_reference resnet18_top_outputs.npz \
  --tolerance 0.98,0.90 \
  --correctness 0.99,0.95 \
  --model resnet18_1684x_int8_asym.bmodel

popd
