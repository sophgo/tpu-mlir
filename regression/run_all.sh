#!/bin/bash
set -e
pip list
# pip3 install onnxsim==0.4.17

# compile for customlayer
source $PROJECT_ROOT/third_party/customlayer/envsetup.sh
rebuild_custom_plugin
rebuild_custom_backend
rebuild_custom_firmware_cmodel bm1684x
rebuild_custom_firmware_cmodel bm1688

$REGRESSION_PATH/main_entry.py --test_type all
