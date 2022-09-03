#!/bin/bash
# set -ex

OUT=$REGRESSION_PATH/regression_out

mkdir -p $OUT
pushd $OUT

# run models
model_list_basic=(
# classification, _cf = caffe, _tf = tflite, default is onnx
  "densenet121-12"
  "efficientnet"
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

run_regression_net()
{
  local net=$1
  echo "======= test $net ====="
  $REGRESSION_PATH/run_model.sh $net > $net.log 2>&1 | true
  if [ "${PIPESTATUS[0]}" -ne "0" ]; then
    echo "$net regression FAILED" >> result.log
    return 1
  else
    echo "$net regression PASSED" >> result.log
    return 0
  fi
}

export -f run_regression_net

run_regression_op()
{
  echo "======= test_onnx.py ====="
  test_onnx.py > test_onnx.log 2>&1 | true
  if [ "${PIPESTATUS[0]}" -ne "0" ]; then
    echo "test_onnx.py FAILED" >> result.log
    return 1
  else
    echo "test_onnx.py PASSED" >> result.log
    return 0
  fi
}

export -f run_regression_op

run_all()
{
  echo "run_regression_op" > cmd.txt
  for net in ${model_list_basic[@]}
  do
    echo "run_regression_net $net" >> cmd.txt
  done
  cat cmd.txt
  parallel -j8 --delay 5  --joblog job_regression.log < cmd.txt
  return $?
}

ERR=0
run_all
if [ "$?" -ne 0 ]; then
  ERR=1
fi

cat result.log

if [ $ERR -eq 0 ]; then
  echo ALL TEST PASSED
else
  echo ALL TEST FAILED
fi

popd

exit $ERR
