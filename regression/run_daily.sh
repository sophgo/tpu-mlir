#!/bin/bash
set -ex

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

rm -rf regression_out
mkdir regression_out
pushd regression_out

# run onnx operation test
test_onnx.py

# to be continued

popd
