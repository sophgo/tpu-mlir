#!/bin/bash
set -ex

INPUT=../mobilenet_in_f32.npz

NNMODELS=${PROJECT_ROOT}/../nnmodels/onnx_models
mkdir -p tmp
pushd tmp

model_transform.py \
    --model_type onnx \
    --model_name resnet50-v1-7-onnx \
    --model_def  ${NNMODELS}/resnet50-v1-7.onnx \
    --input_shapes [[1,3,224,224]] \
    --resize_dims 256,256 \
    --mean 123.675,116.28,103.53 \
    --scale 0.0171,0.0175,0.0174 \
    --pixel_format rgb \
    --test_input ../image/cat.jpg \
    --test_result resnet50-v1-7_fp32_outputs.npz \
    --mlir resnet50-v1-7-onnx.mlir \


#########################
# deploy to float bmodel
#########################
model_deploy.py \
  --mlir resnet50-v1-7-onnx.mlir \
  --quantize F32 \
  --chip bm1684x \
  --test_input resnet50-v1-7-onnx_in_f32.npz \
  --test_reference resnet50-v1-7_fp32_outputs.npz \
  --tolerance 0.99,0.99 \
  --model resnet50-v1-7-onnx_1686_f32.bmodel


#########################
# deploy to int8 bmodel
#########################
run_calibration.py resnet50-v1-7-onnx.mlir \
    --dataset ../image \
    --input_num 2 \
    -o resnet50-v1-7-onnx_cali_table

# to symmetric
model_deploy.py \
  --mlir resnet50-v1-7-onnx.mlir \
  --quantize INT8 \
  --calibration_table resnet50-v1-7-onnx_cali_table \
  --chip bm1684x \
  --test_input resnet50-v1-7-onnx_in_f32.npz \
  --test_reference resnet50-v1-7_fp32_outputs.npz \
  --tolerance 0.95,0.72 \
  --correctness 0.99,0.85 \
  --model resnet50-v1-7-onnx_1684x_int8_sym.bmodel

# to asymmetric
model_deploy.py \
  --mlir resnet50-v1-7-onnx.mlir \
  --quantize INT8 \
  --asymmetric \
  --calibration_table resnet50-v1-7-onnx_cali_table \
  --chip bm1684x \
  --test_input resnet50-v1-7-onnx_in_f32.npz \
  --test_reference resnet50-v1-7_fp32_outputs.npz \
  --tolerance 0.97,0.75 \
  --correctness 0.99,0.85 \
  --model resnet50-v1-7-onnx_1684x_int8_asym.bmodel

popd
