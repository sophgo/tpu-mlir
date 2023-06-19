#!/bin/bash
# test case: test doc 07_mix_precision.rst
set -ex

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

mkdir -p yolov3_tiny
pushd yolov3_tiny

cp ${NNMODELS_PATH}/onnx_models/tiny-yolov3-11.onnx .
cp -rf ${DIR}/../dataset/COCO2017 .
mkdir workspace && cd workspace

model_transform.py \
       --model_name yolov3_tiny \
       --model_def ../tiny-yolov3-11.onnx \
       --input_shapes [[1,3,416,416]] \
       --scale 0.0039216,0.0039216,0.0039216 \
       --pixel_format rgb \
       --keep_aspect_ratio \
       --pad_value 128 \
       --output_names=convolution_output1,convolution_output \
       --mlir yolov3_tiny.mlir

run_calibration.py yolov3_tiny.mlir \
       --dataset ../COCO2017 \
       --input_num 100 \
       -o yolov3_cali_table


run_qtable.py yolov3_tiny.mlir \
       --dataset ../COCO2017 \
       --calibration_table yolov3_cali_table \
       --chip bm1684x \
       --min_layer_cos 0.999 \
       --expected_cos 0.9999 \
       -o yolov3_qtable


model_deploy.py \
       --mlir yolov3_tiny.mlir \
       --quantize INT8 \
       --quantize_table yolov3_qtable \
       --calibration_table yolov3_cali_table \
       --chip bm1684x \
       --model yolov3_mix.bmodel


detect_yolov3.py \
        --model yolov3_mix.bmodel \
        --input ../COCO2017/000000366711.jpg \
        --output yolov3_mix.jpg

if ! grep -q "person" yolov3_mix.bmodel_image_dir_result
then
    echo "mix_precision failed."
    exit 1
fi

popd
