#!/usr/bin/python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

from bmprofile_parser import BMProfileParser
from bmprofile_generator import BMProfileGenerator
from bmprofile_layergroup import BMProfileLLayerGroup
import subprocess
import os


def bmprofile_analyze(input_dir: str, output_dir: str, out_format: str = "html", options={}):
    parser = BMProfileParser()
    parsed_data = parser.parse(input_dir)
    generator = BMProfileGenerator()
    generator.generate(parsed_data, output_dir, out_format, options)


def bmprofile_simulate(input_dir: str, output_dir: str, out_format: str = "html", options={}):
    parser = BMProfileParser()
    parsed_data = parser.parse(input_dir, sim_only=True)
    generator = BMProfileGenerator()
    generator.generate(parsed_data, output_dir, out_format, options)


def bmprofile_parse_command(input_dir: str, output_dir: str, mark_str, arch="bm1684"):
    parser = BMProfileParser()
    parser.parse_static_command(input_dir, output_dir, mark_str, arch)


def bmprofile_check_command(input_dir: str, output_dir: str, mark_str, arch="bm1684"):
    parser = BMProfileParser()
    parser.check_static_command(input_dir, output_dir, mark_str, arch)


def bmprofile_parse_perfAI(input_dir: str, output_dir: str, mark_str, arch="A2", debug=False):
    core_num = 2
    if arch == "BM1690":
        from bmprofile_perfAI_2260 import BMProfileParserPerfAI
        core_num = 8
    else:
        from bmprofile_perfAI import BMProfileParserPerfAI

    bmProfile = BMProfileParserPerfAI()
    bmProfile.parse(input_dir)

    bmProfile.to_txt(output_dir)
    if not debug:
        subprocess.run(
            [f"python  $PROJECT_ROOT/python/PerfAI/PerfAI.web/run_web.py {os.path.abspath(output_dir)} --name PerfAI_web"],  shell=True)
        subprocess.run(
            [f"python  $PROJECT_ROOT/python/PerfAI/PerfAI.doc/run_doc.py {os.path.abspath(output_dir)} {core_num} --style 1"],  shell=True)
