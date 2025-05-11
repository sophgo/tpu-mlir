#!/bin/bash
# test case: test tpu profile
set -ex

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

mkdir -p test8
pushd test8

tpu_profile.py --mode time --arch BM1688 ${REGRESSION_PATH}/profile profile_output --debug

popd
