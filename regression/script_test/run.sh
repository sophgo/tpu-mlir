#!/bin/bash
set -ex

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

TEST_DIR=$REGRESSION_PATH/regression_out/script_test
mkdir -p $TEST_DIR
pushd $TEST_DIR

$DIR/test1.sh
$DIR/test2.sh
$DIR/test3.sh
$DIR/test4.sh

popd
