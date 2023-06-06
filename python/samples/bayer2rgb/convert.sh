#!/bin/bash

mkdir -p tmp
pushd tmp

python3 ../model.py

model_transform.py \
    --model_name bayer2rgb \
    --model_def bayer2rgb.pt \
    --input_shapes [[1,1,1024,1024]] \
    --test_input input.npz \
    --test_result bayer2rgb_top_outputs.npz \
    --mlir bayer2rgb.mlir

mkdir -p dataset
cp input.npz dataset/

run_calibration.py bayer2rgb.mlir \
  --dataset dataset/ \
  --input_num 1 \
  -o bayer2rgb_cali_table

model_deploy.py \
  --mlir bayer2rgb.mlir \
  --quantize INT8 \
  --calibration_table bayer2rgb_cali_table\
  --chip bm1684x \
  --quant_input \
  --quant_output \
  --test_input input.npz \
  --test_reference bayer2rgb_top_outputs.npz \
  --compare_all \
  --disable_layer_group \
  --model bayer2rgb.bmodel

popd

