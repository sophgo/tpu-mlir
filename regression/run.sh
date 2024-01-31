#!/bin/bash
set -e
pip list

if [ "$1" = "op" ]; then
    echo "::RUN operation test."
    source $PROJECT_ROOT/third_party/customlayer/envsetup.sh
    rebuild_custom_plugin
    rebuild_custom_backend
    rebuild_custom_firmware_cmodel bm1684x
    rebuild_custom_firmware_cmodel bm1688
    $REGRESSION_PATH/main_entry.py --test_type basic --test_set op
elif [ "$1" = "script" ]; then
    echo "::RUN script test."
    $REGRESSION_PATH/main_entry.py --test_type basic --test_set script
elif [ "$1" = "model" ]; then
    echo "::RUN model test."
    $REGRESSION_PATH/main_entry.py --test_type basic --test_set model
    echo "::RUN check tests and unit tests."
    cmake --build ${BUILD_PATH} --target check-tpumlir
elif [ "$1" = "check_utests" ]; then
    echo "::RUN check tests and unit tests."
    cmake --build ${BUILD_PATH} --target check-tpumlir
else
    echo "run operation, script and model test"
    $REGRESSION_PATH/main_entry.py --test_type basic
fi
