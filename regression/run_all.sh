#!/bin/bash
set -ex

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

model_list_all=(
# classification, _cf = caffe, _tf = tflite, default is onnx
  "efficientnet"
  "densenet121-12"
  "mobilenet_v2"
  "mobilenet_v2_cf"
  "resnet18_v1"
  "resnet18_v2"
  "resnet18_cf"
  "resnet50_v1"
  "resnet50_v2"
  "resnet50_tf"
  "shufflenet_v2"
  "squeezenet1.0"
  "vgg16"
# object detection
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
