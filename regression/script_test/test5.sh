#!/bin/bash
# test case: test doc 07_mix_precision.rst
set -ex

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

mkdir -p test5
pushd test5

YOLO_PATH=${NNMODELS_PATH}/onnx_models/tiny-yolov3-11.onnx
if [ ! -f ${YOLO_PATH} ]; then
    echo "tiny-yolov3-11.onnx not found, please download it first."
    exit 1
fi
DATASET=${REGRESSION_PATH}/dataset/COCO2017


mkdir workspace && cd workspace

model_transform.py \
       --model_name yolov3_tiny \
       --model_def ${YOLO_PATH} \
       --input_shapes [[1,3,416,416]] \
       --scale 0.0039216,0.0039216,0.0039216 \
       --pixel_format rgb \
       --keep_aspect_ratio \
       --pad_value 128 \
       --output_names=convolution_output1,convolution_output \
       --mlir yolov3_tiny.mlir

run_calibration.py yolov3_tiny.mlir \
       --dataset ${DATASET} \
       --input_num 100 \
       -o yolov3_cali_table


run_calibration.py yolov3_tiny.mlir \
       --dataset ${DATASET} \
       --input_num 100 \
       --search search_qtable\
       --expected_cos 0.9999 \
       --quantize_method_list KL \
       --inference_num 10 \
       --chip bm1684x \
       --calibration_table yolov3_cali_table \
       --quantize_table yolov3_qtable


model_deploy.py \
       --mlir yolov3_tiny.mlir \
       --quantize INT8 \
       --quantize_table yolov3_qtable \
       --calibration_table yolov3_cali_table \
       --chip bm1684x \
       --model yolov3_mix.bmodel


detect_yolov3.py \
        --model yolov3_mix.bmodel \
        --input ${DATASET}/000000366711.jpg \
        --output yolov3_mix.jpg

if ! grep -q "person" yolov3_mix.bmodel_image_dir_result
then
    echo "mix_precision failed."
    exit 1
fi

rm -rf *.npz

popd
