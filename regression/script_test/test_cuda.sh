#!/bin/bash
# test case: test inference by cuda. only in cuda docker
set -ex

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

python3 ${REGRESSION_PATH}/run.sh resnet18_v2 --mode int8_sym --cuda --chip cv183x

python3 ${REGRESSION_PATH}/run.sh mobilenet_v2 --mode int8_sym --cuda --chip cv183x

python3 ${REGRESSION_PATH}/run.sh yolov5s --mode int8_sym --cuda --chip cv183x

python3 ${REGRESSION_PATH}/run.sh resnet18_v2 --mode int8_sym --cuda --chip bm1684x

python3 ${REGRESSION_PATH}/run.sh mobilenet_v2 --mode int8_sym --cuda --chip bm1684x

python3 ${REGRESSION_PATH}/run.sh yolov5s --mode int8_sym --cuda --chip bm1684x
