#!/bin/bash
# test case: test addr_mode, base, io_tag, io_alone
set -ex

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

mkdir -p test9
pushd test9
# convert by batch 1
model_transform.py \
  --model_name mobilenet_v2_b1 \
  --model_def ${REGRESSION_PATH}/model/mobilenet_v2_deploy.prototxt \
  --model_data ${REGRESSION_PATH}/model/mobilenet_v2.caffemodel \
  --input_shapes=[[1,3,224,224]] \
  --resize_dims=256,256 \
  --mean=103.94,116.78,123.68 \
  --scale=0.017,0.017,0.017 \
  --pixel_format=bgr \
  --test_input=${REGRESSION_PATH}/image/cat.jpg \
  --test_result=mobilenet_v2_top_outputs.npz \
  --mlir mobilenet_v2_b1.mlir

# 1684x io alone
model_deploy.py \
  --mlir mobilenet_v2_b1.mlir \
  --quantize F16 \
  --chip bm1684x \
  --addr_mode io_alone \
  --test_input  mobilenet_v2_b1_in_f32.npz \
  --test_reference mobilenet_v2_top_outputs.npz \
  --model mobilenet_v2_1684x_io_alone.bmodel

# 1688 io tag
model_deploy.py \
  --mlir mobilenet_v2_b1.mlir \
  --quantize F16 \
  --chip bm1688 \
  --addr_mode io_tag \
  --test_input  mobilenet_v2_b1_in_f32.npz \
  --test_reference mobilenet_v2_top_outputs.npz \
  --model mobilenet_v2_1688_io_tag.bmodel

# 1688 io alone
model_deploy.py \
  --mlir mobilenet_v2_b1.mlir \
  --quantize F16 \
  --chip bm1688 \
  --addr_mode io_alone \
  --test_input  mobilenet_v2_b1_in_f32.npz \
  --test_reference mobilenet_v2_top_outputs.npz \
  --model mobilenet_v2_1688_io_alone.bmodel

rm -rf *.npz

popd
