#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
# @Time    : 2023/7/18 16:29
# @Author  : chongqing.zeng@sophgo.com
# @Project: PerfAI
import os
import shutil
import pandas as pd
import argparse
from openpyxl import load_workbook
from tqdm import tqdm
from src.generator.details import generate_details, generate_divided_details
from src.generator.layer import generate_layer
from src.generator.style import set_details_style, set_summary_style, set_layer_style, set_sim_summary_style
from src.generator.summary import generate_summary
from src.parser.exfile_parser import GlobalProfileParser
from utils.utils import get_total_time

def run_doc(input, cores, output="PerAI_output.xlsx", style=0, speedup=1, split=0, divided=0):
    input_fold = input if input[-1] == '/' else input + '/'
    out_file = output if '/' in output else input_fold + output
    parser = GlobalProfileParser()
    global_info = parser.parse(input_fold)
    # PerfAI.doc do not support showing layer info without global.profile
    network = global_info.net_name if global_info and global_info.net_name else '--'
    flops = global_info.flops if global_info and global_info.flops else '--'
    quant_type = global_info.quant_type if global_info and global_info.quant_type else '--'
    with pd.ExcelWriter(out_file) as writer:
        if divided == 0:
            tiu_instance_map, gdma_instance_map, chip_arch = generate_details(input_fold, out_file, global_info, writer,
                                                                core_num=cores, split_instr_world=speedup)
            chip_arch['network'] = network
            chip_arch['flops'] = flops
            chip_arch['quant_type'] = quant_type
            if global_info is not None:
                layer_info_map = generate_layer(global_info, writer, out_file, tiu_instance_map, gdma_instance_map, chip_arch)
                generate_summary(layer_info_map, writer, chip_arch)
        else:
            tiu_instance_map, gdma_instance_map, chip_arch = generate_divided_details(input_fold, global_info,
                                                                core_num=cores)

    if style == 1 and divided == 0:
        print('Setting style for ' + out_file)
        set_details_style(out_file, cores, chip_arch)
        set_sim_summary_style(out_file, cores, chip_arch)
        if global_info is not None:
            set_summary_style(out_file)
            set_layer_style(out_file)
    if style == 1 and divided == 1:
        print('Setting style for ' + input_fold)
        for root, dirs, files in os.walk(input_fold):
            for file in files:
                if file.endswith('.xlsx'):
                    out_file = os.path.join(root, file)
                    set_details_style(out_file, cores, chip_arch)
                    set_sim_summary_style(out_file, cores, chip_arch)
                    if global_info is not None:
                        set_summary_style(out_file)
                        set_layer_style(out_file)

    if split:
        xls = pd.ExcelFile(out_file)
        sheet_names = xls.sheet_names
        output_fold = 'ExcelHub'
        if not os.path.exists(output_fold):
            os.makedirs(output_fold)
        print('spliting sheets to csv...')
        for sheet_name in tqdm(sheet_names):
            df = pd.read_excel(xls, sheet_name)
            output_file = f'{output_fold}/{sheet_name}.csv'
            columns = df.columns.tolist()
            for i in range(len(columns)):
                if 'Unnamed' in columns[i]:
                    columns[i] = ' '
            df.columns = columns
            df.to_csv(output_file, index=False)
    if divided == 0:
        perfai_doc_dir = os.path.join(input_fold, 'PerfDoc')
        if not os.path.exists(perfai_doc_dir):
            os.makedirs(perfai_doc_dir)
        for file in os.listdir(input_fold):
            if file.endswith('.xlsx') or file.endswith('.csv'):
                src_file = os.path.join(input_fold, file)
                dst_file = os.path.join(perfai_doc_dir, file)
                if os.path.exists(dst_file):
                    os.remove(dst_file)
                shutil.move(src_file, dst_file)
    if divided == 1:
        for file in os.listdir(input_fold):
            if file.endswith('.xlsx') or file.endswith('.csv'):
                file_path = os.path.join(input_fold, file)
                os.remove(file_path)
    # regInfo_dir = os.path.join(input_fold, 'RegInfo')
    # if not os.path.exists(regInfo_dir):
    #     os.makedirs(regInfo_dir)
    # for file in os.listdir(input_fold):
    #     if 'RegInfo' in file:
    #         shutil.move(os.path.join(input_fold, file), regInfo_dir)
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="The main entry of the PerfAI project")
    parser.add_argument(
        "input",
        type=str,
        default="",
        help="The input file fold path, which contains tiuRegInfo、dmaRegInfo、simTotalCycle、globalProfile file.",
    )
    parser.add_argument(
        "cores",
        type=int,
        default=1,
        help="The number of cores, if bm1684x set 1, set 2 for A2, set 8 for sg2260.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="PerfAI_output.xlsx",
        help="The output file path.",
    )
    parser.add_argument(
        "--style",
        type=int,
        default=0,
        help="If set style for output Excel, input 1 if Yes.",
    )
    parser.add_argument(
        "--speedup",
        type=int,
        default=1,
        help="If separate the instr world sheet into csv, which will increase the speed of the program.",
    )
    parser.add_argument(
        "--split",
        type=int,
        default=0,
        help="If separate the sheets to different excels, which will reduce the size of output.",
    )
    parser.add_argument(
        "--divided",
        type=int,
        default=0,
        help="If write the output Excel into several Excel files separately, which will solve the problem of excessive memory",
    )
    args = parser.parse_args()
    run_doc(args.input, args.cores, args.output, args.style, args.speedup, args.split, args.divided)
