#!/bin/bash
# test case: test tpu profile
set -ex

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"


tpu_profile.py --mode time --arch BM1688 ${REGRESSION_PATH}/profile profile_output --debug

