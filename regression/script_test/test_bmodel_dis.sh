#!/bin/bash
# test case: test bmodel_dis, bm1684, bm1684x, bm1686
set -ex

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
python3 $DIR/test_bmodel_dis.py \
    --tiu_cmd_bin=${REGRESSION_PATH}/cmds/yolov5s_1684.tiu.bin \
    --dma_cmd_bin=${REGRESSION_PATH}/cmds/yolov5s_1684.dma.bin \
    --device=bm1684 \
    --cmd_file=${REGRESSION_PATH}/cmds/bm1684.cmds

python3 $DIR/test_bmodel_dis.py \
    --tiu_cmd_bin=${REGRESSION_PATH}/cmds/yolov5s_1684x.tiu.bin \
    --dma_cmd_bin=${REGRESSION_PATH}/cmds/yolov5s_1684x.dma.bin \
    --device=bm1684x \
    --cmd_file=${REGRESSION_PATH}/cmds/bm1684x.cmds

python3 $DIR/test_bmodel_dis.py \
    --tiu_cmd_bin=${REGRESSION_PATH}/cmds/yolov5s_1686.tiu.bin \
    --dma_cmd_bin=${REGRESSION_PATH}/cmds/yolov5s_1686.dma.bin \
    --device=bm1686 \
    --cmd_file=${REGRESSION_PATH}/cmds/bm1686.cmds