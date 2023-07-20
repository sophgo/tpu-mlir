#!/bin/bash
set -ex

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

IS_BASIC=$1

TEST_DIR=$REGRESSION_PATH/regression_out/script_test
mkdir -p $TEST_DIR
pushd $TEST_DIR

$DIR/test1.sh
$DIR/test2.sh
$DIR/test5.sh
$DIR/test6.sh

if [ $IS_BASIC == 'all' ]; then
  $DIR/test3.sh
  $DIR/test4.sh
fi

popd
