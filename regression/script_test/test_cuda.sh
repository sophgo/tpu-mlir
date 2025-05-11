#!/bin/bash
# test case: test inference by cuda. only in cuda docker
set -ex

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

mkdir -p test_cuda
pushd test_cuda

#python3 ${REGRESSION_PATH}/run_model.py resnet18_v2 --mode int8_sym --cuda --chip cv183x

python3 ${REGRESSION_PATH}/run_model.py mobilenet_v2_cf --mode int8_sym --cuda --chip cv183x

python3 ${REGRESSION_PATH}/run_model.py yolov5s --mode int8_sym --cuda --chip cv183x

#python3 ${REGRESSION_PATH}/run_model.py resnet18_v2 --mode int8_sym --cuda --chip bm1684x

python3 ${REGRESSION_PATH}/run_model.py mobilenet_v2_cf --mode int8_sym --cuda --chip bm1684x

python3 ${REGRESSION_PATH}/run_model.py yolov5s --mode int8_sym --cuda --chip bm1684x

popd
