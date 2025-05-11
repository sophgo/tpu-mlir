#!/bin/bash
# test case: test core parallel
set -ex

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

mkdir -p test7
pushd test7
#
python3 $DIR/test7.py

model_transform.py \
  --model_name test7 \
  --model_def test7.onnx \
  --test_input=test7_input.npz \
  --test_result=test7_top_outputs.npz \
  --mlir test7.mlir

# test one core
model_deploy.py \
  --mlir test7.mlir \
  --quantize F16 \
  --chip bm1688 \
  --num_core 1 \
  --test_input test7_input.npz \
  --test_reference test7_top_outputs.npz \
  --model test7_f16_core1.bmodel

# test two core
model_deploy.py \
  --mlir test7.mlir \
  --quantize F16 \
  --chip bm1688 \
  --num_core 2 \
  --test_input test7_input.npz \
  --test_reference test7_top_outputs.npz \
  --model test7_f16_core2.bmodel

rm -rf *.npz

popd
