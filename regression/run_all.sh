#!/bin/bash
set -ex

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

model_list_all=(
# classification
  "resnet18"
  "resnet50_v2"
  "mobilenet_v2"
  "squeezenet1.0"
  "vgg16"
# object detection
  "resnet34_ssd1200"
  "yolov5s"
  "yolov3_tiny"
  "ssd-12"
  "shufflenet_v12"
)

# run onnx operation test
mkdir -p regression_out
pushd regression_out
test_onnx.py
popd

# run models
for net in ${model_list_all[@]}
do
  echo "======= test $net ====="
  $DIR/run_model.sh $net
  echo "test $net success"
done

echo "test all models success"
