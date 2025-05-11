#!/bin/bash
# test case: test torch input int32 or int16 situation
set -ex

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

mkdir -p test10
pushd test10
#
python3 $DIR/test10.py

model_transform.py \
  --model_name test10_a \
  --model_def test10_a.onnx \
  --mlir test10_a.mlir

model_deploy.py \
  --mlir test10_a.mlir \
  --quantize W4F16 \
  --q_group_size 32 \
  --chip bm1684x \
  --model test10_a.bmodel

model_transform.py \
  --model_name test10_b \
  --model_def test10_b.onnx \
  --mlir test10_b.mlir

model_deploy.py \
  --mlir test10_b.mlir \
  --quantize W4F16 \
  --q_group_size 32 \
  --chip bm1684x \
  --model test10_b.bmodel

# update a weight by b weight
model_tool --update_weight test10_a.bmodel test10_a 0x0 test10_b.bmodel test10_b 0x0

model_runner.py --input test10_input.npz --model test10_a.bmodel --output a_out.npz
model_runner.py --input test10_input.npz --model test10_b.bmodel --output b_out.npz

npz_tool.py compare a_out.npz b_out.npz --tolerance 1.0,1.0

rm -rf *.npz

popd
