#!/bin/bash
# test case: test yolov5s preprocess and postprocess
set -ex

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

model_transform.py \
  --model_name yolov5s \
  --model_def ${REGRESSION_PATH}/model/yolov5s.onnx \
  --input_shapes=[[1,3,192,1024]] \
  --output_names=350,498,646 \
  --scale=0.0039216,0.0039216,0.0039216 \
  --pixel_format=rgb \
  --test_input=${REGRESSION_PATH}/image/dog.jpg \
  --test_result=yolov5s_top_outputs.npz \
  --post_handle_type=yolo \
  --mlir yolov5s.mlir

run_calibration.py yolov5s.mlir \
  --dataset ${REGRESSION_PATH}/dataset/COCO2017 \
  --input_num 100 \
  -o yolov5s_cali_table

model_deploy.py \
  --mlir yolov5s.mlir \
  --quantize INT8 \
  --chip bm1684x \
  --calibration_table yolov5s_cali_table \
  --fuse_preprocess \
  --test_input ${REGRESSION_PATH}/image/dog.jpg \
  --test_reference yolov5s_top_outputs.npz \
  --except "yolo_post" \
  --compare_all \
  --model yolov5s_int8.bmodel
