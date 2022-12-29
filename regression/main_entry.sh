#!/bin/bash
# set -ex

# full test (f32/f16/bf16/int8): main_entry.sh all
# basic test (f32/int8): main_entry.sh basic

OUT=$REGRESSION_PATH/regression_out

test_type=$1

if [ x${test_type} != xall ] && [ x${test_type} != xbasic ]; then
  echo "Error: $0 [basic/all]"
  exit 1
fi

source $REGRESSION_PATH/chip.cfg

mkdir -p $OUT
pushd $OUT

run_regression_net() {
  local net=$1
  local chip_name=$2
  local test_type=$3
  echo "======= run_models.sh $net ${chip_name} ${test_type}====="
  $REGRESSION_PATH/run_model.sh $net ${chip_name} ${test_type} >$net_${chip_name}.log 2>&1 | true
  if [ "${PIPESTATUS[0]}" -ne "0" ]; then
    echo "$net ${chip_name} regression FAILED" >>result.log
    cat $net_${chip_name}.log >>fail.log
    return 1
  else
    echo "$net ${chip_name} regression PASSED" >>result.log
    return 0
  fi
}

export -f run_regression_net

run_onnx_op() {
  echo "======= test_onnx.py ====="
  test_onnx.py >test_onnx.log 2>&1 | true
  if [ "${PIPESTATUS[0]}" -ne "0" ]; then
    echo "test_onnx.py FAILED" >>result.log
    cat test_onnx.log >>fail.log
    return 1
  else
    echo "test_onnx.py PASSED" >>result.log
    return 0
  fi
}

export -f run_onnx_op

run_script_test() {
  echo "======= script test ====="
  $REGRESSION_PATH/script_test/run.sh >script_test.log 2>&1 | true
  if [ "${PIPESTATUS[0]}" -ne "0" ]; then
    echo "script test FAILED" >>result.log
    cat script_test.log >>fail.log
    return 1
  else
    echo "script test PASSED" >>result.log
    return 0
  fi
}

export -f run_script_test

run_all() {
  echo "" >fail.log
  echo "" >result.log
  echo "run_onnx_op" >cmd.txt
  echo "run_script_test" >>cmd.txt
  cat cmd.txt
  parallel -j8 --delay 5 --joblog job_regression.log <cmd.txt
  if [[ "$?" -ne 0 ]]; then
    return 1
  fi
  for chip in ${chip_support[@]}; do
    echo "" >cmd.txt
    declare -n list="model_list_${chip}_${test_type}"
    for net in ${list[@]}; do
      echo "run_regression_net ${net} ${chip} ${test_type}" >>cmd.txt
    done
    cat cmd.txt
    parallel -j8 --delay 5 --joblog job_regression.log <cmd.txt
    if [[ "$?" -ne 0 ]]; then
      return 1
    fi
  done

  return 0
}

ERR=0
run_all
if [ "$?" -ne 0 ]; then
  ERR=1
fi

if [ $ERR -eq 0 ]; then
  cat result.log
  echo run ${test_type} TEST PASSED
else
  cat fail.log
  cat result.log
  echo run ${test_type} TEST FAILED
fi

popd

exit $ERR
