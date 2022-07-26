#!/bin/bash
set -ex

mkdir -p squeezenet1.0-12
pushd squeezenet1.0-12

model_transform.py \
    --model_name squeezenet1.0-12 \
    --model_def  ${NNMODELS_PATH}/onnx_models/squeezenet1.0-12.onnx \
    --input_shapes [[1,3,224,224]] \
    --mean 123.675,116.28,103.53 \
    --scale 0.0171,0.0175,0.0174 \
    --pixel_format rgb \
    --test_input ${REGRESSION_PATH}/image/cat.jpg \
    --test_result squeezenet1.0-12_top_outputs.npz \
    --mlir squeezenet1.0-12.mlir

#########################
# deploy to float bmodel
#########################
model_deploy.py \
  --mlir squeezenet1.0-12.mlir \
  --quantize F32 \
  --chip bm1684x \
  --test_input squeezenet1.0-12_in_f32.npz \
  --test_reference squeezenet1.0-12_top_outputs.npz \
  --tolerance 0.99,0.99 \
  --model squeezenet1.0-12_1684x_f32.bmodel

#########################
# deploy to int8 bmodel
#########################
run_calibration.py squeezenet1.0-12.mlir \
    --dataset $REGRESSION_PATH/image \
    --input_num 2 \
    -o squeezenet1.0-12_cali_table

# to symmetric
model_deploy.py \
  --mlir squeezenet1.0-12.mlir \
  --quantize INT8 \
  --calibration_table squeezenet1.0-12_cali_table \
  --chip bm1684x \
  --test_input squeezenet1.0-12_in_f32.npz \
  --test_reference squeezenet1.0-12_top_outputs.npz \
  --tolerance 0.96,0.74 \
  --correctness 0.99,0.90 \
  --model squeezenet1.0-12_1684x_int8_sym.bmodel

# to asymmetric
model_deploy.py \
  --mlir squeezenet1.0-12.mlir \
  --quantize INT8 \
  --asymmetric \
  --calibration_table squeezenet1.0-12_cali_table \
  --chip bm1684x \
  --test_input squeezenet1.0-12_in_f32.npz \
  --test_reference squeezenet1.0-12_top_outputs.npz \
  --tolerance 0.98,0.82 \
  --correctness 0.99,0.93 \
  --model squeezenet1.0-12_1684x_int8_asym.bmodel

popd
