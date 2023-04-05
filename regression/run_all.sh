#!/bin/bash
set -e
pip list
pip3 install --upgrade tensorflow-cpu
pip3 install onnxsim==0.4.17
$REGRESSION_PATH/main_entry.sh all
