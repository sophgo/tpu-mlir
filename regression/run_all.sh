#!/bin/bash
# set -ex
pip3 install --upgrade onnxsim tensorflow-cpu
$REGRESSION_PATH/main_entry.sh all
