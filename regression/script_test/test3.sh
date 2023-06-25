#!/bin/bash
# test case: test torch input int32 or int16 situation
set -ex

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

#
python3 $DIR/test3.py

# test for int32
model_transform.py \
  --model_name test3 \
  --model_def test3.pt \
  --input_shapes [[4,64,8,8]] \
  --input_types int32 \
  --test_input=test3_input.npz \
  --test_result=test3_top_outputs.npz \
  --mlir test3.mlir

model_deploy.py \
  --mlir test3.mlir \
  --quantize F16 \
  --chip bm1684x \
  --test_input test3_input.npz \
  --test_reference test3_top_outputs.npz \
  --compare_all \
  --model test3_f32.bmodel

run_calibration.py test3.mlir \
  --data_list data3_list \
  -o test3_cali_table

model_deploy.py \
  --mlir test3.mlir \
  --quantize INT8 \
  --chip bm1684x \
  --calibration_table test3_cali_table \
  --test_input test3_input.npz \
  --test_reference test3_top_outputs.npz \
  --compare_all \
  --model test3_1684x_int8.bmodel

model_transform.py \
  --model_name test3_i16 \
  --model_def test3.pt \
  --input_shapes [[4,64,8,8]] \
  --input_types int16 \
  --test_input=test3_input.npz \
  --test_result=test3_top_outputs.npz \
  --mlir test3_i16.mlir

model_deploy.py \
  --mlir test3_i16.mlir \
  --quantize F16 \
  --chip bm1684x \
  --test_input test3_input.npz \
  --test_reference test3_top_outputs.npz \
  --compare_all \
  --model test3_i16.bmodel

model_deploy.py \
  --mlir test3_i16.mlir \
  --quantize INT8 \
  --chip bm1684x \
  --calibration_table test3_cali_table \
  --test_input test3_input.npz \
  --test_reference test3_top_outputs.npz \
  --compare_all \
  --model test3_i16_1684x_int8.bmodel
