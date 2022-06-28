#!/bin/bash
set -ex

INPUT=../mobilenet_in_f32.npz

mkdir -p resnet50_v1
pushd resnet50_v1

model_transform.py \
    --model_name resnet50_v1 \
    --model_def  ${NNMODELS_PATH}/onnx_models/resnet50-v1-7.onnx \
    --input_shapes [[1,3,224,224]] \
    --resize_dims 256,256 \
    --mean 123.675,116.28,103.53 \
    --scale 0.0171,0.0175,0.0174 \
    --pixel_format rgb \
    --test_input ${REGRESSION_PATH}/image/cat.jpg \
    --test_result resnet50_v1_top_outputs.npz \
    --mlir resnet50_v1.mlir


#########################
# deploy to float bmodel
#########################
model_deploy.py \
  --mlir resnet50_v1.mlir \
  --quantize F32 \
  --chip bm1684x \
  --test_input resnet50_v1_in_f32.npz \
  --test_reference resnet50_v1_top_outputs.npz \
  --tolerance 0.99,0.99 \
  --model resnet50_v1_1684x_f32.bmodel


#########################
# deploy to int8 bmodel
#########################
run_calibration.py resnet50_v1.mlir \
    --dataset $REGRESSION_PATH/image \
    --input_num 2 \
    -o resnet50_v1_cali_table

# to symmetric
model_deploy.py \
  --mlir resnet50_v1.mlir \
  --quantize INT8 \
  --calibration_table resnet50_v1_cali_table \
  --chip bm1684x \
  --test_input resnet50_v1_in_f32.npz \
  --test_reference resnet50_v1_top_outputs.npz \
  --tolerance 0.95,0.72 \
  --correctness 0.99,0.95 \
  --model resnet50_v1_1684x_int8_sym.bmodel

# to asymmetric
model_deploy.py \
  --mlir resnet50_v1.mlir \
  --quantize INT8 \
  --asymmetric \
  --calibration_table resnet50_v1_cali_table \
  --chip bm1684x \
  --test_input resnet50_v1_in_f32.npz \
  --test_reference resnet50_v1_top_outputs.npz \
  --tolerance 0.97,0.75 \
  --correctness 0.99,0.95 \
  --model resnet50_v1_1684x_int8_asym.bmodel

popd
