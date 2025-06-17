#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import fire
from debugger.debug_base import DebugPerformance, DebugMetric, entry, getstatusoutput_v2
import sys
import textwrap


class DebugIt(DebugPerformance, DebugMetric):
    """

    usage:
    debugit.py: found ref_files.json recursively in current directory
    debugit [command] ref_files.json

    commonly used commands:
    debugit.py redeploy ref_files.json : redeploy after change code
    debugit.py cmp_perf ref_files.json : collect performance info
    debugit.py cmp_metric ref_files.json : collect metric info
    """


if __name__ == "__main__":
    dbg = DebugIt()
    if len(sys.argv) == 1:
        code, out = getstatusoutput_v2("find . -name ref_files.json", shell=True)
        print(
            "debugit.py COMMAND path/to/ref_files.json\n use --help to get more command information\n\nfiles can be used in current directory:"
        )
        print(textwrap.indent(out, "    "))
    else:
        fire.Fire(entry)
