#!/bin/bash
# test case: test doc 10_sensitive_layer.rst
set -ex

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

mkdir -p mobilenet_v2
pushd mobilenet_v2

cp ${NNMODELS_PATH}/pytorch_models/accuracy_test/classification/mobilenet_v2.pt .
cp -rf ${DIR}/../dataset/ILSVRC2012 .
mkdir workspace && cd workspace

model_transform.py \
       --model_name mobilenet_v2 \
       --model_def ../mobilenet_v2.pt \
       --input_shapes [[1,3,224,224]] \
       --resize_dims 256,256 \
       --mean 123.675,116.28,103.53 \
       --scale 0.0171,0.0175,0.0174 \
       --pixel_format rgb \
       --mlir mobilenet_v2.mlir

run_calibration.py mobilenet_v2.mlir \
       --dataset ../ILSVRC2012 \
       --input_num 100 \
       -o mobilenet_v2_cali_table


run_sensitive_layer.py mobilenet_v2.mlir \
       --dataset ../ILSVRC2012 \
       --calibration_table mobilenet_v2_cali_table \
       --input_num 100 \
       --inference_num 30 \
       --chip bm1684 \
       --expected_cos 0.9999 \
       -o mobilenet_v2_qtable


model_deploy.py \
       --mlir mobilenet_v2.mlir \
       --quantize INT8 \
       --quantize_table mobilenet_v2_qtable \
       --calibration_table new_cali_table.txt \
       --chip bm1684 \
       --model mobilenet_v2_mix.bmodel


classify_mobilenet_v2.py \
        --model_def mobilenet_v2_mix.bmodel \
        --input ../ILSVRC2012/n01440764_9572.JPEG \
        --output mobilnet_v2_mix.jpg \
        --category_file ../ILSVRC2012/synset_words.txt

popd
