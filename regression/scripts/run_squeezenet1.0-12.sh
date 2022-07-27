#!/bin/bash
set -ex

mkdir -p squeezenet1.0
pushd squeezenet1.0

MODEL_PATH=${MODEL_ZOO_PATH}/vision/classification/squeezenet1.0/squeezenet1.0-12.onnx
if [ ! -f ${MODEL_PATH} ]; then
  MODEL_PATH=${NNMODELS_PATH}/onnx_models/squeezenet1.0-12.onnx
fi

model_transform.py \
    --model_name squeezenet1.0 \
    --model_def  $MODEL_PATH \
    --input_shapes [[1,3,224,224]] \
    --mean 123.675,116.28,103.53 \
    --scale 0.0171,0.0175,0.0174 \
    --pixel_format rgb \
    --test_input ${REGRESSION_PATH}/image/cat.jpg \
    --test_result squeezenet1.0_top_outputs.npz \
    --mlir squeezenet1.0.mlir

#########################
# deploy to float bmodel
#########################
model_deploy.py \
  --mlir squeezenet1.0.mlir \
  --quantize F32 \
  --chip bm1684x \
  --test_input squeezenet1.0_in_f32.npz \
  --test_reference squeezenet1.0_top_outputs.npz \
  --tolerance 0.99,0.99 \
  --model squeezenet1.0_1684x_f32.bmodel

#########################
# deploy to int8 bmodel
#########################
# CALI_TABLE=${REGRESSION_PATH}/cali_tables/squeezenet1.0_cali_table
# run_calibration.py squeezenet1.0.mlir \
#     --dataset ${REGRESSION_PATH}/dataset/ILSVRC2012/ \
#     --input_num 100 \
#     -o $CALI_TABLE
CALI_TABLE=squeezenet1.0_cali_table
run_calibration.py squeezenet1.0.mlir \
  --dataset ${REGRESSION_PATH}/image \
  --input_num 2 \
  -o $CALI_TABLE


# to symmetric
model_deploy.py \
  --mlir squeezenet1.0.mlir \
  --quantize INT8 \
  --calibration_table $CALI_TABLE \
  --chip bm1684x \
  --test_input squeezenet1.0_in_f32.npz \
  --test_reference squeezenet1.0_top_outputs.npz \
  --tolerance 0.96,0.74 \
  --correctness 0.99,0.90 \
  --model squeezenet1.0_1684x_int8_sym.bmodel

# to asymmetric
model_deploy.py \
  --mlir squeezenet1.0.mlir \
  --quantize INT8 \
  --asymmetric \
  --calibration_table $CALI_TABLE \
  --chip bm1684x \
  --test_input squeezenet1.0_in_f32.npz \
  --test_reference squeezenet1.0_top_outputs.npz \
  --tolerance 0.98,0.82 \
  --correctness 0.99,0.93 \
  --model squeezenet1.0_1684x_int8_asym.bmodel

popd
