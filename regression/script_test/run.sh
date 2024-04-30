#!/bin/bash
set -ex

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

IS_BASIC=$1

TEST_DIR=$REGRESSION_PATH/regression_out/script_test
mkdir -p $TEST_DIR
pushd $TEST_DIR

$DIR/test1.sh # test workflow
$DIR/test2.sh # test quantization with qtable
$DIR/test5.sh # test mix precision
$DIR/test9.sh # test addr mode

if [ $IS_BASIC == 'all' ]; then
  $DIR/test3.sh    # test i32 or i16 input
  $DIR/test4.sh    # test preprocess and postprocess
  $DIR/test6.sh    # test sensitive layer
  $DIR/test7.sh    # test core parallel
  $DIR/test8.sh    # test tpu profile
  $DIR/test10.sh   # test model_tool update_weight
  $DIR/test_llm.sh # test llm models
fi

popd
