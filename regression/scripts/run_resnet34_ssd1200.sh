#!/bin/bash
set -ex

mkdir -p resnet34_ssd1200
pushd resnet34_ssd1200

MODEL_PATH=${MODEL_ZOO_PATH}/vision/detection/ssd-resnet34/resnet34-ssd1200.onnx
if [ ! -f ${MODEL_PATH} ]; then
  MODEL_PATH=${NNMODELS_PATH}/onnx_models/resnet34-ssd1200.onnx
fi

model_transform.py \
    --model_name resnet34_ssd1200 \
    --model_def  ${MODEL_PATH} \
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


#########################
# deploy to int8 bmodel
#########################
# CALI_TABLE=${REGRESSION_PATH}/cali_tables/resnet34_ssd1200_cali_table
# run_calibration.py resnet34_ssd1200.mlir \
#   --dataset ${REGRESSION_PATH}/dataset/COCO2017/ \
#   --input_num 100 \
#   -o $CALI_TABLE

CALI_TABLE=resnet34_ssd1200_cali_table
run_calibration.py resnet34_ssd1200.mlir \
  --dataset ${REGRESSION_PATH}/image \
  --input_num 2 \
  -o $CALI_TABLE

# to symmetric
model_deploy.py \
  --mlir resnet34_ssd1200.mlir \
  --quantize INT8 \
  --calibration_table $CALI_TABLE \
  --chip bm1684x \
  --test_input resnet34_ssd1200_in_f32.npz \
  --test_reference resnet34_ssd1200_top_outputs.npz \
  --tolerance 0.98,0.79 \
  --correctness 0.99,0.99 \
  --model resnet34_ssd1200_1684x_int8_sym.bmodel


# to symmetric
# model_deploy.py \
#   --mlir resnet34_ssd1200.mlir \
#   --quantize INT8 \
#   --asymmetric \
#   --calibration_table resnet34_ssd1200_cali_table \
#   --chip bm1684x \
#   --test_input resnet34_ssd1200_in_f32.npz \
#   --test_reference resnet34_ssd1200_top_outputs.npz \
#   --tolerance 0.98,0.84 \
#   --correctness 0.99,0.95 \
#   --model resnet34_ssd1200_1684x_int8_asym.bmodel

popd
