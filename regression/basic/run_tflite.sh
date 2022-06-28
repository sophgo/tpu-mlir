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
tpuc-opt resnet50_tflite.mlir \
    --toptflite-to-tpu \
    --canonicalize \
    --save-weight \
    -o resnet50_int8_1684x.mlir

#########################
# TFLiteTPU BM1684x check
#########################
model_runner.py \
    --model resnet50_int8_1684x.mlir \
    --input resnet50_in_f32.npz \
    --dump_all_tensors \
    --output resnet50_int8_outputs_1684x.npz

npz_tool.py compare \
    resnet50_int8_outputs_1684x.npz \
    resnet50_top_outputs.npz \
    --tolerance 0.89,0.50 -v

popd
