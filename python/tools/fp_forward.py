#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
import  pymlir
import argparse
from utils.fp_layer_search import FpLayerSearcher
from utils.log_setting import setup_logger

logger = setup_logger("fp_forward")

if __name__ == '__main__':
    logger.info("TPU-MLIR {}".format(pymlir.__version__))
    # yapf: disable
    parser = argparse.ArgumentParser(description="Generate quantization table")
    parser.add_argument('mlir_file', help='fp32 mlir file')
    parser.add_argument('--fpfwd_inputs', type=str, default='',
                        help='pre-processing layer')
    parser.add_argument('--fpfwd_outputs', type=str, default='',
                        help='post-processing layer')
    parser.add_argument('--fpfwd_blocks', nargs='+', type=str,
                        help='layers in blocks')
    parser.add_argument('--chip', '--processor', required=True, type=str,
                        choices=['bm1684x', 'bm1684','bm1688', 'cv183x', 'cv182x', 'cv181x', 'cv180x', 'bm1690','cv186x'],
                        help='chip platform name')
    parser.add_argument('--fp_type', default='auto', type=str,
                        choices=['auto', 'F16', 'F32', 'BF16'],
                        help='float type of mix precision')
    parser.add_argument('-o', '--quantize_free_table', required=True,
                        help='output searched fp layer table')
    # yapf: enable
    args = parser.parse_args()

    searcher = FpLayerSearcher(args)
    searcher.run()
