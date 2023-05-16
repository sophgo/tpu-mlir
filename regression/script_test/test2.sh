#!/bin/bash
# test case: test quantization with qtable
set -ex

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

#
python3 $DIR/test2.py

model_transform.py \
  --model_name test2 \
  --model_def test2.onnx \
  --test_input=test2_input.npz \
  --test_result=test2_top_outputs.npz \
  --mlir test2.mlir

model_deploy.py \
  --mlir test2.mlir \
  --quantize F32 \
  --chip bm1684x \
  --test_input test2_input.npz \
  --test_reference test2_top_outputs.npz \
  --compare_all \
  --model test2_f32.bmodel

run_calibration.py test2.mlir \
  --data_list data2_list \
  -o test2_cali_table

touch test2_qtable
run_qtable.py test2.mlir \
  --data_list data2_list \
  --calibration_table test2_cali_table \
  --chip bm1684x \
  -o test2_qtable

model_deploy.py \
  --mlir test2.mlir \
  --quantize INT8 \
  --chip bm1684x \
  --calibration_table test2_cali_table \
  --quantize_table test2_qtable \
  --test_input test2_input.npz \
  --test_reference test2_top_outputs.npz \
  --tolerance 0.9,0.5 \
  --compare_all \
  --model test2_1684x_int8.bmodel
