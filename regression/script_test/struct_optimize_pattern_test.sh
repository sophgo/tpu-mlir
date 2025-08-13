#!/bin/bash
set -ex

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

mkdir -p struct_optimize_pattern_test
pushd struct_optimize_pattern_test
# export ONNX and NPZ into current workdir
OUT_DIR="$(pwd)" python3 $DIR/struct_optimize_pattern_test.py

# model_transform
model_transform.py \
  --model_name struct_optimize_pattern_test \
  --model_def struct_optimize_pattern_test.onnx \
  --test_input struct_optimize_pattern_test_input.npz \
  --test_result struct_optimize_pattern_test_top_outputs.npz \
  --mlir struct_optimize_pattern_test.mlir \
  --excepts \sq_Squeeze \
  --struct_optimize 1

# model_deploy (float32)
model_deploy.py \
  --mlir struct_optimize_pattern_test.mlir \
  --quantize F32 \
  --chip bm1684x \
  --test_input struct_optimize_pattern_test_input.npz \
  --test_reference struct_optimize_pattern_test_top_outputs.npz \
  --model struct_optimize_pattern_test_f32.bmodel

rm -rf *.npz

popd
