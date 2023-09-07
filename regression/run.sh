#!/bin/bash
set -e
pip list

if [ "$1" = "op" ]; then
    echo "run operation test"
    $REGRESSION_PATH/main_entry.py --test_type basic --test_set op
elif [ "$1" = "script" ]; then
    echo "run script test"
    $REGRESSION_PATH/main_entry.py --test_type basic --test_set script
elif [ "$1" = "model" ]; then
    echo "run model test"
    $REGRESSION_PATH/main_entry.py --test_type basic --test_set model
else
    echo "run operation, script and model test"
    $REGRESSION_PATH/main_entry.py --test_type basic
fi
