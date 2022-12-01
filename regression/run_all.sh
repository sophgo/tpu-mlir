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
  "inception_v3"
  "mnist-12"
  "mobilenet_v2"
  "mobilenet_v2_cf"
  "mobilenet_v2_tf"
  "resnet18_v1"
  "resnet18_v2"
  "resnet18_cf"
  "resnet50_v1"
  "resnet50_v2"
  "resnet50_tf"
  "segnet_cf"
  "se-resnet50"
  "shufflenet_v2"
  "squeezenet1.0"
  "vgg16"
  "inception_v4_tf"
# object detection
  "ssd-12"
  "yolov5s"
  "yolov3_tiny"
  "yolov5s_tf"
)

# run models
model_list_cv18xx_basic=(
  "mobilenet_v2"
  "resnet18_v1"
  "resnet18_v2"
  "resnet50_v1"
  "resnet50_v2"
  "squeezenet1.0"
  "vgg16"
# object detection
  "ssd-12"
  "yolov5s"
)

run_regression_net()
{
  local net=$1
  local chip=$2
  echo "======= test $net $chip ====="
  if [ x$chip == "xcv18xx" ]; then
    $REGRESSION_PATH/run_model_cv18xx.sh $net 0 > $net.log 2>&1 | true
  else
    $REGRESSION_PATH/run_model.sh $net 0 > $net.log 2>&1 | true
  fi
  if [ "${PIPESTATUS[0]}" -ne "0" ]; then
    echo "$net $chip regression FAILED" >> result.log
    cat $net.log >> fail.log
    return 1
  else
    echo "$net $chip regression PASSED" >> result.log
    return 0
  fi
}

export -f run_regression_net

run_onnx_op()
{
  echo "======= test_onnx.py ====="
  test_onnx.py > test_onnx.log 2>&1 | true
  if [ "${PIPESTATUS[0]}" -ne "0" ]; then
    echo "test_onnx.py FAILED" >> result.log
    cat test_onnx.log >> fail.log
    return 1
  else
    echo "test_onnx.py PASSED" >> result.log
    return 0
  fi
}

export -f run_onnx_op

run_basic_test()
{
  echo "======= basic test ====="
  $REGRESSION_PATH/basic_test/run.sh > basic_test.log 2>&1 | true
  if [ "${PIPESTATUS[0]}" -ne "0" ]; then
    echo "basic test FAILED" >> result.log
    cat basic_test.log >> fail.log
    return 1
  else
    echo "basic test PASSED" >> result.log
    return 0
  fi
}

export -f run_basic_test

run_all()
{
  echo "" > fail.log
  echo "" > result.log
  echo "run_onnx_op" > cmd.txt
  echo "run_basic_test" >> cmd.txt
  for net in ${model_list_basic[@]}
  do
    echo "run_regression_net $net" >> cmd.txt
  done
  for net in ${model_list_cv18xx_basic[@]}
  do
    echo "run_regression_net $net cv18xx" >> cmd.txt
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

if [ $ERR -eq 0 ]; then
  echo ALL TEST PASSED
else
  cat fail.log
  echo ALL TEST FAILED
fi

cat result.log

popd

exit $ERR
