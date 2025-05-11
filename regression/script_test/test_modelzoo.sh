#!/bin/bash
# test model-zoo in gitlab regression system
set -ex

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
mkdir -p test_modelzoo
pushd test_modelzoo
python3 $DIR/test_modelzoo.py
popd
