#!/bin/bash
set -ex


mkdir -p tmp
pushd tmp
model_transform.py \
    --model_type tflite \
    --model_name resnet50 \
    --input_shapes [[4,3,224,224]] \
    --model_def  ../resnet50_quant_int8.tflite \
    --mlir resnet50_tflite.mlir
popd
