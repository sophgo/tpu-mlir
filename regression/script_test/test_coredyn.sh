#!/bin/bash
# test case: test torch input int32 or int16 situation
set -ex

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

mkdir -p test_coredyn
pushd test_coredyn
#
python3 $DIR/test_coredyn.py

# test for int32
model_transform.py \
  --model_name test_coredyn \
  --model_def test_coredyn.pt \
  --input_shapes [[4,512]] \
  --test_input=test_coredyn_input.npz \
  --test_result=test_coredyn_top_outputs.npz \
  --mlir test_coredyn.mlir

model_deploy.py \
  --mlir test_coredyn.mlir \
  --quantize F16 \
  --chip bm1688 \
  --num_core 2 \
  --dynamic \
  --test_input test_coredyn_input.npz \
  --test_reference test_coredyn_top_outputs.npz \
  --compare_all \
  --debug \
  --model test_coredyn_f16.bmodel


popd
