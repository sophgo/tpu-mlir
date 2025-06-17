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
import traceback
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


def bmprofile_parse_perfAI(
    input_dir: str,
    output_dir: str,
    mark_str="",
    arch="A2",
    debug=False,
    doc=True,
    web=True,
):
    import shutil

    core_num = 2
    style = 1
    if arch == "BM1690":
        from bmprofile_perfAI_2260 import BMProfileParserPerfAI as ParserIns

        core_num = 8
        style = 0
    elif arch == "MARS3":
        from bmprofile_perfAI import BMProfileParserPerfAI_MARS3 as ParserIns
        core_num = 1
    else:
        from bmprofile_perfAI import BMProfileParserPerfAI as ParserIns
    try:
        bmProfile = ParserIns()
        bmProfile.parse(input_dir)
        bmProfile.to_txt(output_dir)
        target_dir = output_dir
    except Exception as e:
        target_dir = input_dir
        if debug:
            traceback.print_exc()
        print(f"parse {arch} profile failed, try run PerfAI directly")

    if not debug:
        if web:
            print("Generate web...")
            subprocess.run(
                [
                    f"python  $TPUC_ROOT/python/PerfAI/PerfAI.web/run_web.py {os.path.abspath(target_dir)} \
                --layerinfo_dir {input_dir} --name PerfAI_web"
                ],
                shell=True,
            )
            shutil.move(os.path.join(target_dir, "PerfWeb"), os.path.join(output_dir, "PerfWeb"))
        if doc:
            print("Generate doc...")
            subprocess.run(
                [
                    f"python  $TPUC_ROOT/python/PerfAI/PerfAI.doc/run_doc.py {os.path.abspath(target_dir)} \
                    --layerinfo_dir {input_dir} {core_num} --style {style}"
                ],
                shell=True,
            )

            shutil.move(os.path.join(target_dir, "PerfDoc"), os.path.join(output_dir, "PerfDoc"))
