#!/bin/bash
set -ex

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# run onnx operation test
mkdir -p regression_out
pushd regression_out
test_onnx.py
popd

# run models
model_list_basic=(
# classification
  "mobilenet_v2"
  "resnet50_v2"
  "resnet50_tf"
  "squeezenet1.0"
# object detection
  "yolov5s"
)

for net in ${model_list_basic[@]}
do
  echo "======= test $net ====="
  $DIR/run_model.sh $net
  echo "test $net success"
done

echo "test basic models success"
