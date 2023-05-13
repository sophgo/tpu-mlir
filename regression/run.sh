#!/bin/bash
set -e
pip list
pip3 install onnx==1.13.0
$REGRESSION_PATH/main_entry.py --test_type basic
