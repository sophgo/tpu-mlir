#!/bin/bash
# test case: test yolov5s preprocess and postprocess
set -ex

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

mkdir -p yolov5s_1output
pushd yolov5s_1output

# yolov5s 1 output
model_transform.py \
  --model_name yolov5s_1o \
  --model_def ${REGRESSION_PATH}/model/yolov5s.onnx \
  --input_shapes=[[1,3,192,1024]] \
  --output_names=output \
  --keep_aspect_ratio \
  --scale=0.0039216,0.0039216,0.0039216 \
  --pixel_format=rgb \
  --test_input=${REGRESSION_PATH}/image/dog.jpg \
  --test_result=yolov5s_1o_top_outputs.npz \
  --add_postprocess=yolov5 \
  --mlir yolov5s_1o.mlir

run_calibration.py yolov5s_1o.mlir \
  --dataset ${REGRESSION_PATH}/dataset/COCO2017 \
  --input_num 100 \
  --tune_num 20 \
  -o yolov5s_1o_cali_table

run_qtable.py \
  yolov5s_1o.mlir \
	--dataset ${REGRESSION_PATH}/dataset/COCO2017 \
	--chip bm1684x \
	-o yolov5s_1o_qtable \
	--calibration_table yolov5s_1o_cali_table

# last 4 concat should be float
matches=$(sed -nE 's/.*\("([^)]+_Concat)"\).*/\1/p' "yolov5s_1o.mlir" | tail -n 4)
new_matches=""
for match in $matches; do
  new_matches+="$match F16\n"
done
echo -e $new_matches >> yolov5s_1o_qtable

model_deploy.py \
  --mlir yolov5s_1o.mlir \
  --quantize INT8 \
  --chip bm1684x \
  --calibration_table yolov5s_1o_cali_table \
  --quantize_table yolov5s_1o_qtable \
  --fuse_preprocess \
  --test_input ${REGRESSION_PATH}/image/dog.jpg \
  --test_reference yolov5s_1o_top_outputs.npz \
  --compare_all \
  --except "yolo_post" \
  --debug \
  --model yolov5s_1o_int8.bmodel

detect_yolov5.py \
  --input ${REGRESSION_PATH}/image/dog.jpg \
  --model yolov5s_1o_int8.bmodel \
  --net_input_dims 192,1024 \
  --fuse_postprocess \
  --out dog_out.jpg

popd

mkdir -p yolov5s_3output
pushd yolov5s_3output

# yolov5s 3 output
model_transform.py \
  --model_name yolov5s_3o \
  --model_def ${REGRESSION_PATH}/model/yolov5s.onnx \
  --input_shapes=[[1,3,192,1024]] \
  --output_names=326,474,622 \
  --keep_aspect_ratio \
  --scale=0.0039216,0.0039216,0.0039216 \
  --pixel_format=rgb \
  --test_input=${REGRESSION_PATH}/image/dog.jpg \
  --test_result=yolov5s_3o_top_outputs.npz \
  --add_postprocess=yolov5 \
  --mlir yolov5s_3o.mlir

run_calibration.py yolov5s_3o.mlir \
  --dataset ${REGRESSION_PATH}/dataset/COCO2017 \
  --input_num 100 \
  --tune_num 20 \
  -o yolov5s_3o_cali_table

model_deploy.py \
  --mlir yolov5s_3o.mlir \
  --quantize INT8 \
  --chip bm1684x \
  --calibration_table yolov5s_3o_cali_table \
  --fuse_preprocess \
  --test_input ${REGRESSION_PATH}/image/dog.jpg \
  --test_reference yolov5s_3o_top_outputs.npz \
  --compare_all \
  --except "yolo_post" \
  --debug \
  --model yolov5s_3o_int8.bmodel

detect_yolov5.py \
  --input ${REGRESSION_PATH}/image/dog.jpg \
  --model yolov5s_3o_int8.bmodel \
  --net_input_dims 192,1024 \
  --fuse_postprocess \
  --out dog_out.jpg

popd

mkdir -p yolov3
pushd yolov3

# yolov3
model_transform.py \
  --model_name yolov3 \
  --model_def ${NNMODELS_PATH}/onnx_models/yolov3-10.onnx \
  --input_shapes=[[1,3,416,416]] \
  --output_names=convolution_output2,convolution_output1,convolution_output \
  --keep_aspect_ratio \
  --scale=0.0039216,0.0039216,0.0039216 \
  --pixel_format=rgb \
  --test_input=${REGRESSION_PATH}/image/dog.jpg \
  --test_result=yolov3_top_outputs.npz \
  --add_postprocess=yolov3 \
  --mlir yolov3.mlir

model_deploy.py \
  --mlir yolov3.mlir \
  --quantize F32 \
  --chip bm1684x \
  --fuse_preprocess \
  --test_input ${REGRESSION_PATH}/image/dog.jpg \
  --test_reference yolov3_top_outputs.npz \
  --compare_all \
  --except "yolo_post" \
  --debug \
  --model yolov3_f32.bmodel

detect_yolov3.py \
  --input ${REGRESSION_PATH}/image/dog.jpg \
  --model yolov3_f32.bmodel \
  --net_input_dims 416,416 \
  --fuse_preprocess \
  --fuse_postprocess \
  --out dog_out.jpg

popd
