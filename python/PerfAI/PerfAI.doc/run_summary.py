# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
from src import perf_summary
import sys
import os
from datetime import datetime


if __name__ == "__main__":
    f_paths, p_paths, s_paths = [], [], []
    args = sys.argv[1:]
    if (len(args) - 1) % 3  != 0:
        print('Input Error, please check your input,\
              you need prepare 3 file for each model and output path totally')
        exit(-1)
    model_num = int((len(args) - 1) / 3)
    dirs = 'ABCSet_Summary/'
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    out_file = args[0] if '/' in args[0] else dirs + 'summary' + str(datetime.now()) + '.xlsx'
    for i in range(model_num):
        f_paths.append(args[3*i + 1])
        p_paths.append(args[3*i + 2])
        s_paths.append(args[3*i + 3])
    perf_summary.run(out_file, f_paths, p_paths, s_paths, set_style=True)
    print('Successfully generate ABCSetSummary excel in:', out_file)
