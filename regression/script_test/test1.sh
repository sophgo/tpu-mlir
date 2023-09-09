#!/bin/bash
# test case: test batch 4, calibration by npz, preprocess, tosa convert
set -ex

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

# convert by batch 4
model_transform.py \
  --model_name mobilenet_v2 \
  --model_def ${REGRESSION_PATH}/model/mobilenet_v2_deploy.prototxt \
  --model_data ${REGRESSION_PATH}/model/mobilenet_v2.caffemodel \
  --input_shapes=[[4,3,224,224]] \
  --resize_dims=256,256 \
  --mean=103.94,116.78,123.68 \
  --scale=0.017,0.017,0.017 \
  --pixel_format=bgr \
  --test_input=${REGRESSION_PATH}/image/cat.jpg \
  --test_result=mobilenet_v2_top_outputs.npz \
  --mlir mobilenet_v2.mlir

# input to npy
npz_tool.py to_npy mobilenet_v2_in_f32.npz data

# cali with batch 4
run_calibration.py mobilenet_v2.mlir \
  --dataset ${REGRESSION_PATH}/dataset/ILSVRC2012 \
  --input_num 6 \
  -o mobilenet_v2_cali_table

# cali by npz
mkdir -p npz_data
cp mobilenet_v2_in_f32.npz npz_data
run_calibration.py mobilenet_v2.mlir \
  --dataset npz_data \
  --input_num 1 \
  -o mobilenet_v2_cali_table_by_npz

# cali by npy
mkdir -p npy_data
cp data.npy npy_data
run_calibration.py mobilenet_v2.mlir \
  --dataset npy_data \
  --input_num 1 \
  -o mobilenet_v2_cali_table_by_npy

# cali by list
echo data.npy >data.list
run_calibration.py mobilenet_v2.mlir \
  --data_list data.list \
  -o mobilenet_v2_cali_table_by_list

run_qtable.py mobilenet_v2.mlir \
  --dataset ${REGRESSION_PATH}/dataset/ILSVRC2012 \
  --calibration_table mobilenet_v2_cali_table \
  --chip bm1684x \
  -o mobilenet_qtable

# do fuse preprocess
# f32
model_deploy.py \
  --mlir mobilenet_v2.mlir \
  --quantize F32 \
  --chip bm1684x \
  --fuse_preprocess \
  --test_input ${REGRESSION_PATH}/image/cat.jpg \
  --test_reference mobilenet_v2_top_outputs.npz \
  --compare_all \
  --model mobilenet_v2_1684x_f32_fuse.bmodel
# f16
model_deploy.py \
  --mlir mobilenet_v2.mlir \
  --quantize F16 \
  --chip bm1684x \
  --fuse_preprocess \
  --test_input ${REGRESSION_PATH}/image/cat.jpg \
  --test_reference mobilenet_v2_top_outputs.npz \
  --compare_all \
  --model mobilenet_v2_1684x_f16_fuse.bmodel
# int8
model_deploy.py \
  --mlir mobilenet_v2.mlir \
  --quantize INT8 \
  --chip bm1684x \
  --calibration_table mobilenet_v2_cali_table \
  --fuse_preprocess \
  --test_input ${REGRESSION_PATH}/image/cat.jpg \
  --test_reference mobilenet_v2_top_outputs.npz \
  --compare_all \
  --model mobilenet_v2_1684x_int8_fuse.bmodel

# no test
model_deploy.py \
  --mlir mobilenet_v2.mlir \
  --quantize INT8 \
  --chip bm1684x \
  --calibration_table mobilenet_v2_cali_table \
  --fuse_preprocess \
  --model mobilenet_v2_1684x_int8_fuse2.bmodel

# convert to tosa without weight
# tpuc-opt mobilenet_v2.mlir \
#   --chip-top-optimize \
#   --convert-top-to-tosa="includeWeight=False" \
#   -o mobilenet_v2_tosa_no_weight.mlir

tpuc-opt mobilenet_v2.mlir \
  --chip-top-optimize \
  --convert-top-to-tosa="includeWeight=True" \
  -o mobilenet_v2_tosa.mlir

# mlir-opt mobilenet_v2_tosa.mlir \
#   --pass-pipeline="func.func(\
#     tosa-to-linalg-named, \
#     tosa-to-linalg, \
#     tosa-to-arith, \
#     tosa-to-scf, \
#     canonicalize, \
#     linalg-bufferize, \
#     convert-linalg-to-affine-loops, \
#     affine-loop-fusion, \
#     affine-simplify-structures, \
#     lower-affine)" \
#   -o mobilenet_v2_affine.mlir

# mlir-opt mobilenet_v2_affine.mlir \
#   --func-bufferize --tensor-bufferize --arith-expand --arith-bufferize \
#   --normalize-memrefs --convert-scf-to-cf --convert-math-to-llvm \
#   --convert-arith-to-llvm --llvm-request-c-wrappers --convert-func-to-llvm \
#   --convert-cf-to-llvm --memref-expand --canonicalize \
#   --llvm-legalize-for-export --reconcile-unrealized-casts |
#   mlir-translate --mlir-to-llvmir |
#   llc -mtriple=x86_64-unknown-linux-gnu --filetype=obj \
#     -o mobilenet_v2.o
