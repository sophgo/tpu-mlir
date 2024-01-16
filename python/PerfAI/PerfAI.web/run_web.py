import os
import shutil
import time
import argparse

from utils.js_prep import *
from utils.utils import *

def run_web(reginfo_dir, name='PerfWeb'):
    out_path = os.path.join(reginfo_dir, name)
    os.makedirs(out_path, exist_ok=True)
    files_dir = os.path.abspath(__file__).replace("run_web.py","files")
    htmlfiles = [os.path.join(files_dir, 'echarts.min.js'), os.path.join(files_dir, 'jquery-3.5.1.min.js'), os.path.join(files_dir, 'result.html')]

    for f in htmlfiles:
        shutil.copy2(f, out_path)

    print(f"Generating data for {out_path}/result.html")
    generate_jsfile(reginfo_dir, name, out_path)
    print(f"The jsfile is generated successfully under {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process the data and prepare the web files')
    parser.add_argument(
        'reginfo_dir',
        type=str,
        default='',
        help='The folder path that contains tiuRegInfo、dmaRegInfo txt files.')
    parser.add_argument(
        '--name',
        type=str,
        default='PerfWeb',
        help='The output folder name that contains html and js files.')
    args = parser.parse_args()
    run_web(args.reginfo_dir, args.name)
