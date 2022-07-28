#!/bin/bash
set -ex

mkdir -p yolov5s
pushd yolov5s

model_transform.py \
    --model_name yolov5s \
    --model_def ${REGRESSION_PATH}/model/yolov5s.onnx \
    --input_shapes [[1,3,640,640]] \
    --mean 0.0,0.0,0.0 \
    --scale 0.0039216,0.0039216,0.0039216 \
    --keep_aspect_ratio \
    --pixel_format rgb \
    --output_names 350,498,646 \
    --test_input ${REGRESSION_PATH}/image/dog.jpg \
    --test_result yolov5s_top_outputs.npz \
    --mlir yolov5s.mlir

#########################
# deploy to float bmodel
#########################
model_deploy.py \
  --mlir yolov5s.mlir \
  --quantize F32 \
  --chip bm1684x \
  --test_input yolov5s_in_f32.npz \
  --test_reference yolov5s_top_outputs.npz \
  --tolerance 0.99,0.99 \
  --model yolov5s_1684x_f32.bmodel


#########################
# deploy to int8 bmodel
#########################
# only once
CALI_TABLE=${REGRESSION_PATH}/cali_tables/yolov5s_cali_table
# run_calibration.py yolov5s.mlir \
#     --dataset ${REGRESSION_PATH}/dataset/COCO2017/ \
#     --input_num 100 \
#     -o $CALI_TABLE

# to symmetric
model_deploy.py \
  --mlir yolov5s.mlir \
  --quantize INT8 \
  --calibration_table $CALI_TABLE \
  --chip bm1684x \
  --test_input yolov5s_in_f32.npz \
  --test_reference yolov5s_top_outputs.npz \
  --tolerance 0.85,0.43 \
  --correctness 0.99,0.90 \
  --model yolov5s_1684x_int8_sym.bmodel

# to asymmetric: support later
# model_deploy.py \
#   --mlir yolov5s.mlir \
#   --quantize INT8 \
#   --asymmetric \
#   --calibration_table $CALI_TABLE \
#   --chip bm1684x \
#   --test_input yolov5s_in_f32.npz \
#   --test_reference yolov5s_top_outputs.npz \
#   --tolerance 0.83,0.27 \
#   --correctness 0.99,0.93 \
#   --model yolov5s_1684x_int8_asym.bmodel

#########################
# detect image by detect_yolov5.py
#########################

# by onnx
detect_yolov5.py \
  --input ${REGRESSION_PATH}/image/dog.jpg \
  --model ${REGRESSION_PATH}/model/yolov5s.onnx \
  --output dog_onnx.jpg

# by f32 bmodel
detect_yolov5.py \
  --input ${REGRESSION_PATH}/image/dog.jpg \
  --model yolov5s_1684x_f32.bmodel \
  --output dog_f32.jpg

# by int8 symmetric bmodel
detect_yolov5.py \
  --input ${REGRESSION_PATH}/image/dog.jpg \
  --model yolov5s_1684x_int8_sym.bmodel \
  --output dog_int8_sym.jpg

# by int8 asymmetric bmodel
# detect_yolov5.py \
#   --input ${REGRESSION_PATH}/image/dog.jpg \
#   --model yolov5s_1684x_int8_asym.bmodel \
#   --output dog_int8_asym.jpg

popd
