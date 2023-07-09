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

def bmprofile_analyze(input_dir:str, output_dir:str, out_format:str = "html", options = {}):
    parser  = BMProfileParser()
    parsed_data = parser.parse(input_dir)
    generator = BMProfileGenerator()
    generator.generate(parsed_data, output_dir, out_format, options)

def bmprofile_simulate(input_dir:str, output_dir:str, out_format:str = "html", options = {}):
    parser  = BMProfileParser()
    parsed_data = parser.parse(input_dir, sim_only=True)
    generator = BMProfileGenerator()
    generator.generate(parsed_data, output_dir, out_format, options)

def bmprofile_parse_command(input_dir:str, output_dir:str, mark_str, arch="bm1684"):
    parser  = BMProfileParser()
    parser.parse_static_command(input_dir, output_dir, mark_str, arch)

def bmprofile_check_command(input_dir:str, output_dir:str, mark_str, arch="bm1684"):
    parser  = BMProfileParser()
    parser.check_static_command(input_dir, output_dir, mark_str, arch)
