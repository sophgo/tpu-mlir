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
import re

from .mlir_parser import MlirParser

g_auto_remove_files = []

# libomp names its runtime-registration shm file "__KMP_REGISTERED_LIB_<pid>_<uid>"
# (see the getpid/getuid pair feeding the format string in libomp.so).
_KMP_FILE_NAME = re.compile(r"__KMP_REGISTERED_LIB_(\d+)_(\d+)$")


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
    names = os.listdir("/dev/shm")
    for name in names:
        m = _KMP_FILE_NAME.match(name)
        if m is None or int(m.group(2)) != uid:
            continue
        if os.path.isdir("/proc/" + m.group(1)):  # the process is still alive
            continue
        try:
            os.remove(os.path.join("/dev/shm", name))
        except FileNotFoundError:
            # FileNotFoundError means we lost a race to a concurrent cleaner
            pass
