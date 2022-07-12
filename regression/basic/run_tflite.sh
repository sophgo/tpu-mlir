#!/bin/bash
set -ex

mkdir -p tflite
pushd tflite
model_transform.py \
    --model_name resnet50 \
    --model_def  $REGRESSION_PATH/model/resnet50_quant_int8.tflite \
    --input_shapes [[1,3,224,224]] \
    --resize_dims 256,256 \
    --mean 123.675,116.28,103.53 \
    --scale 0.0171,0.0175,0.0174 \
    --pixel_format rgb \
    --test_input ${REGRESSION_PATH}/image/cat.jpg \
    --test_result resnet50_top_outputs.npz \
    --mlir resnet50_tflite.mlir

#########################
# TFLite to TPU BM1684x
#########################
model_deploy.py \
  --mlir resnet50_tflite.mlir \
  --chip bm1684x \
  --quantize INT8 \
  --asymmetric \
  --test_input resnet50_in_f32.npz \
  --test_reference resnet50_top_outputs.npz \
  --tolerance 0.95,0.71 \
  --correctness 0.99,0.92 \
  --model resnet50_tflite_1684x_int8_asym.bmodel

popd
