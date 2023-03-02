#!/bin/bash
set -ex

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

#
python3 $DIR/test2.py

model_transform.py \
  --model_name test \
  --model_def test.onnx \
  --test_input=test_input.npz \
  --test_result=test_top_outputs.npz \
  --mlir test.mlir

model_deploy.py \
  --mlir test.mlir \
  --quantize F32 \
  --chip bm1684x \
  --test_input test_input.npz \
  --test_reference test_top_outputs.npz \
  --compare_all \
  --model test_f32.bmodel

run_calibration.py test.mlir \
  --data_list data_list \
  -o test_cali_table

# run_qtable.py test.mlir \
#   --data_list data_list \
#   --calibration_table test_cali_table \
#   --chip bm1684x \
#   -o test_qtable

model_deploy.py \
  --mlir test.mlir \
  --quantize INT8 \
  --chip bm1684x \
  --calibration_table test_cali_table \
  --test_input test_input.npz \
  --test_reference test_top_outputs.npz \
  --compare_all \
  --model test_1684x_int8.bmodel
