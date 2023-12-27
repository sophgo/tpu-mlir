import os
import argparse
import shutil
import time

from utils.js_prep import *
from utils.utils import *

if __name__ == '__main__':
    start = time.time()
    parser = argparse.ArgumentParser(description='Process the data and prepare the web files')
    parser.add_argument(
        'output_dir',
        type=str,
        default='',
        help='The folder path that contains tiuRegInfo、dmaRegInfo txt files.')
    args = parser.parse_args()
    pagename_input = input("Please enter the name you would entitle the webpage. Skip to use default name <Perfweb>: ").strip()
    if not pagename_input:
        pagename = 'PerfWeb'
    else:
        pagename = pagename_input
    out_path = os.path.join(args.output_dir, 'PerfWeb_' + pagename)
    os.makedirs(out_path, exist_ok=True)
    files_dir = os.path.abspath(__file__).replace("perfAI_html.py","files")
    htmlfiles = [os.path.join(files_dir, 'echarts.min.js'), os.path.join(files_dir, 'jquery-3.5.1.min.js'), os.path.join(files_dir, 'result.html')]

    for f in htmlfiles:
        shutil.copy2(f, out_path)

    print("Start generating data!")
    generate_jsfile(args.output_dir, pagename, out_path)

    end = time.time()
    passed = end - start
    print(f"Total spent time: {passed} seconds")
    print(f"The jsfile is generated successfully under {out_path}")