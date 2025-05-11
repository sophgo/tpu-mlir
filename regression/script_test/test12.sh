#!/bin/bash
# test case: test mlir file convert workflow
set -ex

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
mkdir -p test12
pushd test12

python3 $DIR/test12.py

model_transform.py \
  --model_name resnetquant \
  --model_def resnetquant_origin.mlir \
  --mlir resnetquant.mlir

model_deploy.py \
  --mlir resnetquant.mlir \
  --quantize INT8 \
  --chip bm1684x \
  --model resnetquant.bmodel

rm -rf *.npz

popd
