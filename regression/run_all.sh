#!/bin/bash
set -ex

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

model_list_all=(
# classification
  "densenet121-12"
  "efficientnet"
  "mobilenet_v2"
  "resnet18_v1"
  "resnet18"
  "resnet50_v1"
  "resnet50_v2"
  "resnet50_tf"
  "shufflenet_v12"
  "squeezenet1.0"
  "vgg16"
# object detection
  "resnet34_ssd1200"
  "ssd-12"
  "yolov5s"
  "yolov3_tiny"

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
