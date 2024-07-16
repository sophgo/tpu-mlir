#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
# @Time    : 2023/8/23 15:16
# @Author  : chongqing.zeng@sophgo.com
# @Project: PerfAI
from tqdm import tqdm

from include.summary import Summary
from include.dma import Gdma, Sdma, Cdma
from include.asic_summary import AsicSummary
from include.tiu import Tiu
from include.layer import TotalLayerInfo


def set_details_style(out_file, core_num, chip_arch):
    for core_id in tqdm(range(core_num)):
        Tiu.set_style(out_file, core_id)
        Gdma.set_style(out_file, core_id, chip_arch=chip_arch)
        Sdma.set_style(out_file, core_id, chip_arch=chip_arch)
        Cdma.set_style(out_file, core_id, chip_arch=chip_arch)

def set_sim_summary_style(out_file, core_num, chip_arch):
    AsicSummary.set_style(out_file, chip_arch=chip_arch)

def set_layer_style(out_file):
    TotalLayerInfo.set_style(out_file)


def set_summary_style(out_file):
    Summary.set_style(out_file)
