# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import os
import shutil
import time
import argparse

from utils.js_prep import *
from utils.utils import *
from utils.power import *

def run_web(reginfo_dir, name, file_path,power, version):
    out_path = os.path.join(reginfo_dir, 'PerfWeb')
    os.makedirs(out_path, exist_ok=True)
    templates_dir = os.path.abspath(__file__).replace("run_web.py","templates")
    htmlfiles = [os.path.join(templates_dir, 'echarts.min.js'), os.path.join(templates_dir, 'jquery-3.5.1.min.js'),os.path.join(templates_dir, 'result.html')]
    for f in htmlfiles:
        shutil.copy2(f, out_path)

    # print(f"Generating data for {out_path}/result.html")
    reginfo = reginfo_dir if reginfo_dir[-1] == '/' else reginfo_dir + '/' #end with /
    generate_jsfile(reginfo, name, out_path, file_path)
    if power:
        html_path = os.path.join(templates_dir, 'power_standard.html')
        htmlfiles.append(html_path)
        prepareOutput(reginfo, version, name, html_path,out_path)
    print(f"The web files are generated successfully under {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process the data and prepare the web files')
    parser.add_argument(
        'reginfo_dir',
        type=str,
        default='',
        help='The folder path that contains tiuRegInfo、dmaRegInfo txt files.')
    parser.add_argument(
        '--name','-n',
        type=str,
        required=True,
        help='The pattern name that would be used to create html and js files.')
    parser.add_argument(
        '--file','-f',
        type=str,
        default='',
        help='The path to the TXT file that records dependent op groups.')
    parser.add_argument('--power','-p', type=bool, default=False, help='Input True for generating power charts. Default is False')
    parser.add_argument('--version', '-v', type=str, default='', help='AI compiler commit ID. Please provide this info if you need to present it on power chart.')
    args = parser.parse_args()
    run_web(args.reginfo_dir, args.name, args.file, args.power, args.version)
