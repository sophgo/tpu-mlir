#!/bin/bash
set -ex

mkdir -p resnet50_tf
pushd resnet50_tf
model_transform.py \
    --model_name resnet50_tf \
    --model_def  $REGRESSION_PATH/model/resnet50_int8.tflite \
    --input_shapes [[1,3,224,224]] \
    --mean 103.939,116.779,123.68 \
    --scale 1.0,1.0,1.0 \
    --pixel_format bgr \
    --test_input ${REGRESSION_PATH}/image/cat.jpg \
    --test_result resnet50_tf_top_outputs.npz \
    --mlir resnet50_tf.mlir

#########################
# TFLite to TPU BM1684X
#########################
model_deploy.py \
    --mlir resnet50_tf.mlir \
    --chip bm1684x \
    --quantize INT8 \
    --test_input resnet50_tf_in_f32.npz \
    --test_reference resnet50_tf_top_outputs.npz \
    --tolerance 0.95,0.71 \
    --model resnet50_tf_1684x.bmodel

popd
