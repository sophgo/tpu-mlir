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
        if n.endswith(".mlir"):
            try:
                parser = MlirParser(n)
                weight_npz = parser.module_weight_file
                if os.path.exists(weight_npz):
                    os.remove(weight_npz)
            except:
                pass
        else:
            os.remove(n)


def clean_kmp_files():
    # When using multi-processing with the "fork" mode and compiling a library
    # with clang and libomp, some __KMP__ files may leak in /dev/shm. This can
    # take over all the space if /dev/shm is small.
    # eg: oneDNN/src/common/dnnl_thread.hpp:69 triggers the creation of __KMP__
    #     files when using OpenMP.
    uid = os.getuid()
    os.system(f"rm -f /dev/shm/__KMP_REGISTERED_LIB_*_{uid}")
