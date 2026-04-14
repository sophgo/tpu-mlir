#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import argparse
import transform.TpuLang as tpul


def main():
    parser = argparse.ArgumentParser(description='tpulang debug dump test Script')
    parser.add_argument('-c', '--chip', type=str, required=True, help='bm1684x or bm1688')
    parser.add_argument('-p', '--dump_path', type=str, required=True, help='tpulang dump path')
    args = parser.parse_args()
    tpul.init(args.chip)
    tpul.debug_dump_compile(args.dump_path)
    tpul.deinit()


if __name__ == "__main__":
    main()
