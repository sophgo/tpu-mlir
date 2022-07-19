#!/bin/bash
set -ex

mkdir -p resnet34_ssd1200
pushd resnet34_ssd1200

# https://github.com/onnx/models/tree/main/vision/object_detection_segmentation/ssd#preprocessing-steps
model_transform.py \
    --model_name resnet34_ssd1200 \
    --model_def  ${NNMODELS_PATH}/onnx_models/resnet34-ssd1200.onnx \
    --input_shapes [[1,3,1200,1200]] \
    --resize_dims 1200,1200 \
    --mean 123.675,116.28,103.53 \
    --pixel_format rgb \
    --scale 0.01712475,0.017507,0.01742919 \
    --test_input ${REGRESSION_PATH}/image/cat.jpg \
    --output_names Concat_470,Concat_471 \
    --test_result resnet34_ssd1200_top_outputs.npz \
    --mlir resnet34_ssd1200.mlir


#########################
# deploy to float bmodel
#########################
model_deploy.py \
  --mlir resnet34_ssd1200.mlir \
  --quantize F32 \
  --chip bm1684x \
  --test_input resnet34_ssd1200_in_f32.npz \
  --test_reference resnet34_ssd1200_top_outputs.npz \
  --tolerance 0.99,0.99 \
  --model resnet34_ssd1200_1684x_f32.bmodel

popd
