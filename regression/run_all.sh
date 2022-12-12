#!/bin/bash
# set -ex
pip3 install --upgrade onnxsim
$REGRESSION_PATH/main_entry.sh all
