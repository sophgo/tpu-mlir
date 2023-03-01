#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import os
from .mlir_parser import MlirParser

g_auto_remove_files = []


def file_mark(file: str):
    g_auto_remove_files.append(file)


def file_clean():
    for n in g_auto_remove_files:
        if not os.path.exists(n):
            continue
        if n.endswith('.mlir'):
            try:
                parser = MlirParser(n)
                weight_npz = parser.module_weight_file
                if os.path.exists(weight_npz):
                    os.remove(weight_npz)
            except:
                pass
        os.remove(n)
