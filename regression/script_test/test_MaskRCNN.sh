#!/bin/bash
# test case: test torch input int32 or int16 situation
set -ex

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

#
python3 $DIR/../../python/test/test_MaskRCNN.py --debug
