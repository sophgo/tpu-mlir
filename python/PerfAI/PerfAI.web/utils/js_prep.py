# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
import pandas as pd
from decimal import Decimal
from numpy import transpose
import os
from functools import reduce
from collections import defaultdict
import math

from src.tiu import TIU
from src.dma import DMA
from src.summary import SummaryProcessor
from src.mlir_json import *
from src.mlir_json import GlobalProfileParser
from utils.utils import *


def generate_jsfile(dirpath, name, out_path, file_path):
    include_layer = False
    parser = GlobalProfileParser()
    result = parser.parse(dirpath)
    if result is not None:
        mlir_info, file_line_dict = result
        tiu_layer_map, dma_layer_map = get_engine_layer(mlir_info)
    else:
        mlir_info = None
        file_line_dict = None
    tiu_layer_map, dma_layer_map = get_engine_layer(mlir_info)
    tiuProcessor = TIU(dirpath)
    tiu_instance = tiuProcessor.process_file(tiu_layer_map)
    gdmaProcessor = DMA(dirpath, "GDMA")
    gdma_instances = gdmaProcessor.process_file(dma_layer_map)
    sdmaProcessor = DMA(dirpath, "SDMA")
    sdma_instances = sdmaProcessor.process_file(dma_layer_map)
    cdmaProcessor = DMA(dirpath, "CDMA")
    cdma_instances = cdmaProcessor.process_file(dma_layer_map)
    processors = [tiuProcessor, gdmaProcessor, sdmaProcessor, cdmaProcessor]
    chipArchArgs = None
    for processor in processors:
        if processor.chipArgs:
            chipArchArgs = processor.chipArgs
            break
    global CHIP_ARCH
    CHIP_ARCH = chipArchArgs['Chip Arch']
    print("CHIP_ARCH:",CHIP_ARCH)
    # core_num = int(chipArchArgs['Core Num'])
    core_num = max(tiuProcessor.actual_corenum, gdmaProcessor.actual_corenum, sdmaProcessor.actual_corenum, cdmaProcessor.actual_corenum)
    ddrBw = pd.to_numeric(chipArchArgs['DDR Max BW(GB/s/Core)'])
    L2Bw = pd.to_numeric(chipArchArgs['L2 Max BW(GB/s)'])
    dependCmds = parse_cmdgroups(file_path)
    time_header = ["category", "begin_time", "end_time", "Duration", "stall_time", "func_type", "height", "cmd", "func_name","uArchRate/BW", "Data Type", "Info","Msg_Id","Sd/Wt_Count"]
    filter_cols = [time_header.index(c) for c in ["category", "func_type"]]
    # time_header = ["category", "begin_time", "end_time", "Duration", "stall_time", "func_type", "height", "cmd", "func_name", 'layer_id','layer_name','subnet_id','subnet_type',"uArchRate/BW", "Data Type", "Info","Msg_Id","Sd/Wt_Count"]
    # filter_cols.extend([time_header.index(c) for c in ['layer_id','layer_name','subnet_id','subnet_type']])

    categories = ["TPU_BD", "TPU_GDMA"]
    if len(sdma_instances[0]) > 0:
        categories.append("TPU_SDMA")
    if len(cdma_instances) > 0:
        categories.append("TPU_CDMA")
    if tiu_layer_map or dma_layer_map:
        include_layer = True
        categories.append("TPU_LAYER")
        categories.append("TPU_GROUP_LAYER")
        time_header = ["category", "begin_time", "end_time", "Duration", "stall_time", "func_type", "height", "cmd", "func_name", 'layer_id','layer_name','subnet_id','subnet_type',"uArchRate/BW", "Data Type", "Info","Msg_Id","Sd/Wt_Count"]
        filter_cols.extend([time_header.index(c) for c in ['layer_id','layer_name','subnet_id','subnet_type']])
    lmem_size = int(chipArchArgs['TPU Lmem Size(MiB)'])
    lane_num = int(chipArchArgs['NPU Num'])
    lane_size  = lmem_size // lane_num
    lmem_partition = generate_partition(lmem_size,lane_num,'BANK')

    cycle_data_dict = {f"time_data{i}": [] for i in range(0, core_num)}
    lmem_op_dict = {f"lmem_op_record{i}": [] for i in range(0, core_num)}

    max_corenum = max(len(tiu_instance), len(gdma_instances), len(sdma_instances), len(cdma_instances))
    for idx in range(max_corenum):
        if idx < len(tiu_instance):
            tiudf = tiu_instance[idx]
            prepare_data(include_layer, tiudf, tiuProcessor.frequency, idx, 0, [ddrBw, L2Bw], lane_num, cycle_data_dict, lmem_op_dict, lane_size)

        if idx < len(gdma_instances):
            gdmadf = gdma_instances[idx]
            prepare_data(include_layer, gdmadf, gdmaProcessor.frequency, idx, 1, [ddrBw, L2Bw], lane_num, cycle_data_dict, lmem_op_dict, lane_size)

        if idx < len(sdma_instances):
            sdmadf = sdma_instances[idx]
            if not sdmadf.empty:
                prepare_data(include_layer, sdmadf, sdmaProcessor.frequency, idx, categories.index("TPU_SDMA"), [ddrBw, L2Bw], lane_num, cycle_data_dict, lmem_op_dict, lane_size)

        if idx < len(cdma_instances):
            cdmadf = cdma_instances[idx]
            prepare_data(include_layer, cdmadf, cdmaProcessor.frequency,idx, categories.index("TPU_CDMA"), [ddrBw, L2Bw], lane_num, cycle_data_dict, lmem_op_dict, lane_size)

    cycle_data_dict = merge_layer_data(cycle_data_dict, categories)
    cycle_data_dict = merge_group_layer_data(cycle_data_dict, categories, file_line_dict)

    summary = SummaryProcessor(tiuProcessor, gdmaProcessor, sdmaProcessor,cdmaProcessor)
    summarydf = summary.make_summary()
    summary_data =[[str(x) if isinstance(x,Decimal) else x for x in lst] for lst in summarydf.values.tolist()]

    # if CHIP_ARCH == 'sg2260':
    #     ddr_ratios, l2m_ratios = calculate_ratios(cycle_data_dict)
    # else:
    #     ddr_ratios, l2m_ratios = [], []
    data_filepath = f"{out_path}/profile_data.js"
    with open(data_filepath, "w") as js:
        js.write(f'let page_caption = "PerfAI: {name}"\n')
        js.write(f'let platform = "Platform: {CHIP_ARCH}"\n')
        js.write(f'let configs = {chipArchArgs}\n')
        js.write('let summary_caption= "Summary Table"\n')
        js.write(f'let summary_header =  {summarydf.columns.tolist()}\n')
        js.write(f'let summary_data = {summary_data}\n')
        js.write(f'let ddr_bandwidth = {ddrBw}\n')
        js.write(f'let l2_bandwidth = {L2Bw}\n')
        # js.write(f'let ddr_ratios = {ddr_ratios}\n')
        # js.write(f'let l2m_ratios = {l2m_ratios}\n')
        js.write(f'let dependCmds = {dependCmds}\n')
        js.write(f'let categories = {categories}\n')
        js.write(f'let filter_cols = {filter_cols}\n')
        js.write(f'let lmem_partition = {lmem_partition}\n')
        js.write(f'let time_header = {time_header}\n')
        for lmem_op in lmem_op_dict.keys():
            js_content = ""
            for i, sublist in enumerate(lmem_op_dict[lmem_op]):
                js_content += f"{sublist},\n"
            js.write(f'window.{lmem_op} = [{js_content}]\n')

        for keyname in cycle_data_dict.keys():
            js_content = ""
            for i, sublist in enumerate(cycle_data_dict[keyname]):
                js_content += f"{sublist},\n"
            js.write(f'window.{keyname} = [{js_content}]\n')

def prepare_data(if_layer, data, frequency, idx, ip_type, bwlist, lane_num, cycle_data_dict, lmem_op_dict, lane_size):
    if data.empty:
        return
    read_directions = ['DDR->LMEM'] + [f'DDR->LMEM{i}' for i in range(8)] + [f'L2M->LMEM{i}' for i in range(8)] + [f'L2M{i}->LMEM{i}' for i in range(8)]
    write_directions = ['LMEM->DDR'] + [f'LMEM{i}->DDR' for i in range(8)] + [f'LMEM{i}->L2M' for i in range(8)] + [f'LMEM{i}->L2M{i}' for i in range(8)]
    # lmem_direction = [f'LMEM{i}->LMEM{i}' for i in range(8)]
    lmem_temp = []
    frequency = int(frequency)
    if 'DDR Bandwidth(GB/s)' in data:
        data['DDR Bandwidth(GB/s)'] = data['DDR Bandwidth(GB/s)'].apply(lambda x: str(x) if isinstance(x, Decimal) else x)
        data['L2M Bandwidth(GB/s)'] = data['L2M Bandwidth(GB/s)'].apply(lambda x: str(x) if isinstance(x, Decimal) else x)
    for i in range(len(data)):
        uarch_rate = pd.to_numeric(data['uArch Rate'][i][:-1]) if 'uArch Rate' in data.columns else None
        cmd = int(data['Cmd Id'][i])
        if 'DDR Bandwidth(GB/s)' in data:
            if 'L2M' in data['Direction'][i]:
                height = round(pd.to_numeric(data['L2M Bandwidth(GB/s)'][i]) / bwlist[1], 2)
            else:
                height = round(pd.to_numeric(data['DDR Bandwidth(GB/s)'][i]) / (bwlist[0] +1e-6), 2)
        else:
            height = round(uarch_rate/100, 2)
        tmp = [
            ip_type,
            get_realtime_from_cycle(data['Start Cycle'][i], frequency),
            get_realtime_from_cycle(data['End Cycle'][i], frequency),
            get_realtime_from_cycle(data['Asic Cycle'][i], frequency),
            int(data['Stall Cycle'][i]) if 'Stall Cycle' in data and data['Stall Cycle'][i] is not None else '',
            data['Function Type'][i] if 'Function Type' in data else '',
            height,
            cmd,
            data['Function Name'][i]
        ]
        if if_layer:
            tmp.extend([
                data['Layer Id'][i],
                data['Layer Name'][i],
                data['Subnet Id'][i],
                data['Subnet Type'][i],
                data['File Line'][i],
        ])
        tmp.extend([
            data['uArch Rate'][i] if 'uArch Rate' in data else f"DDR:{data['DDR Bandwidth(GB/s)'][i]},L2M:{data['L2M Bandwidth(GB/s)'][i]}",
            data['Data Type'][i],
            f"Direction:{data['Direction'][i]}" if 'Direction' in data else f"Bank Conflict Ratio:{data['Bank Conflict Ratio'][i]}",
            data['Msg Id'][i],
            data['Sd\Wt Count'][i],
        ])
        cycle_data_dict[f'time_data{idx}'].append(tmp)

        ## prepare the data for the mem graph
        if 'Direction' in data: #
            direction = data['Direction'][i]
            src = data['src_start_addr'][i]
            dst = data['dst_start_addr'][i]
            datasize = pd.to_numeric(data['DMA data size(B)'][i])
            op_type = 0 if direction in read_directions else 1
            if direction in read_directions:
                size = datasize / lane_num  #c维除以lane_num
                info = f'{type}{cmd},physical start addr:{dst}<br>{direction},{data["dst_shape"][i]}'
                lmem_temp.append([int(data['Start Cycle'][i]), int(data['End Cycle'][i]), op_type, dst, size, info])
            elif direction in write_directions:
                size = datasize / lane_num #c维除以lane num
                info = f'{type}{cmd},physical start addr:{src}<br>{direction},{data["src_shape"][i]}'
                lmem_temp.append([int(data['Start Cycle'][i]), int(data['End Cycle'][i]), op_type, src, size, info])
        # import pdb; pdb.set_trace()
        if 'des_res0_c' in data and data['des_res0_c'][i] and int(data['des_res0_c'][i]) < lane_num:
            worklane = int(data['des_res0_c'][i])
        else:
            worklane = lane_num
        if 'des_res0_size' in data and data['des_res0_size'][i] is not None:
            #the memory address reads the result tensor: op=0
            size = data['des_res0_size'][i] / worklane
            info = f'{type}{cmd},physical start addr:{data["des_res0_addr"][i]}'
            lmem_temp.append([int(data['Start Cycle'][i]), int(data['End Cycle'][i]), 0, data['des_res0_addr'][i], size, info])
        if 'des_opd0_size' in data and data['des_opd0_size'][i] is not None:
            size = (data['des_opd0_size'][i]+data['des_opd1_size'][i] if data['des_opd1_size'][i] is not None else data['des_opd0_size'][i]) / worklane
            info = f'{type}{cmd},physical start addr:{data["des_opd0_addr"][i]}'
            lmem_temp.append([int(data['Start Cycle'][i]), int(data['End Cycle'][i]), 1, data['des_opd0_addr'][i], size, info])
    process_lmem = processAddr(lmem_temp, lane_size) if len(lmem_temp) > 0 else lmem_temp
    lmem_op_dict[f'lmem_op_record{idx}'].extend(process_lmem)
    lmem_op_dict[f'lmem_op_record{idx}'] = deduplicate_ordered_list(lmem_op_dict[f'lmem_op_record{idx}'])
    cycle_data_dict[f'time_data{idx}'] = deduplicate_ordered_list(cycle_data_dict[f'time_data{idx}'])


def merge_layer_data(cycle_data_dict, categories):
    if "TPU_LAYER" not in categories:
        return cycle_data_dict  # 如果categories中不包含TPU_LAYER，则不进行任何操作

    ip_type = categories.index("TPU_LAYER")
    for key in cycle_data_dict.keys():
        data = cycle_data_dict[key]
        layer_data = {}

        # group by layer_id
        for entry in data:
            layer_id = entry[9]
            if layer_id not in layer_data:
                layer_data[layer_id] = []
            layer_data[layer_id].append(entry)
        for layer_id, entries in layer_data.items():
            start_times = [int(e[1]) for e in entries]
            end_times = [int(e[2]) for e in entries]
            earliest_start = min(start_times)
            latest_end = max(end_times)
            duration = latest_end - earliest_start
            layer_name = entries[0][10]
            subnet_id = entries[0][11]
            subnet_type = entries[0][12]

            merged_entry = [
                ip_type, earliest_start, latest_end, duration, '', layer_name, 0.5,
                '', layer_name, layer_id, layer_name, subnet_id, subnet_type,
                '', '', '', '',''
            ] ##info: to be filled
            data.append(merged_entry)

        cycle_data_dict[key] = data
    return cycle_data_dict


def merge_group_layer_data(cycle_data_dict, categories, file_line_dict):
    if "TPU_GROUP_LAYER" not in categories:
        return cycle_data_dict  # If the categories do not include TPU_GROUP-LAYER, no action will be taken

    ip_type = categories.index("TPU_GROUP_LAYER")
    for key in cycle_data_dict.keys():
        data = cycle_data_dict[key]
        group_data = {}
        group_num = 1
        # group by group
        for entry in data:
            file_line = entry[13]
            if type(file_line) != str and file_line is not None and not math.isnan(file_line):
                for file_line_dict_key in file_line_dict:
                    if file_line in file_line_dict[file_line_dict_key]:
                        if file_line_dict_key not in group_data:
                            group_data[file_line_dict_key] = []
                        group_data[file_line_dict_key].append(entry)

        for group_id, entries in group_data.items():
            start_times = [int(e[1]) for e in entries]
            end_times = [int(e[2]) for e in entries]
            earliest_start = min(start_times)
            latest_end = max(end_times)
            duration = latest_end - earliest_start
            layer_name = f'Group{group_num}  %{group_id}'
            subnet_id = entries[0][11]
            subnet_type = entries[0][12]
            group_num += 1

            merged_entry = [
                ip_type, earliest_start, latest_end, duration, '', layer_name, 0.5,
                '', '', '', '', subnet_id, subnet_type,
                '', '', '', '',''
            ] ##info: to be filled
            data.append(merged_entry)

        cycle_data_dict[key] = data

    for key in cycle_data_dict:
        for entry in cycle_data_dict[key]:
            del entry[13]

    return cycle_data_dict


def parse_cmdgroups(file_path):
    mapped_lines = []
    if file_path:
        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    components = line.strip().split()
                    mapped_components = [component.replace('T', 'TIU').replace('G', 'GDMA').replace('S', 'SDMA') for component in components]
                    mapped_lines.append(mapped_components)
                return mapped_lines
        except FileNotFoundError:
            raise FileNotFoundError("The specified file does not exist. Please check the path.")
    else:
        return mapped_lines

def generate_partition(lmem_size, lane_num, type_name):
    partition = []
    start_value = 0
    divisor = 1 if type_name == 'LANE' else 16
    partition_size = lmem_size // lane_num // divisor
    num = lane_num if type_name == 'LANE' else 16
    for i in range(num):
        sublist = [start_value, partition_size, f'{type_name}[{i}]']
        partition.append(sublist)
        start_value += partition_size
    return partition

def multiply_non_zero_numbers(input_str):
    nums = input_str.split(')')[0][1:].split(',')
    result = reduce(lambda x, y: x * y, [int(n) for n in nums if int(n) != 0])
    return result

def deduplicate_ordered_list(lst):
    seen = set()
    deduped = []
    for item in sorted(lst, key=lambda x: x[0]):
        t_item = tuple(item)
        if t_item not in seen:
            seen.add(t_item)
            deduped.append(item)
    return deduped

def processAddr(lst, lane_size):
    dec_addr = [int(sublist[3], 16) for sublist in lst]
    new_lst = []
    min_value = min(dec_addr) # Aling the address for different cores
    for i, sublist in enumerate(lst):
        # Calculate the converted address value
        shifted_addr = dec_addr[i] - min_value
        new_shifted_addr = shifted_addr % lane_size
        lane_occupied = math.ceil(shifted_addr / lane_size)
        sublist[3] = new_shifted_addr
        sublist[-1] += f"<br>Occupied Lane#:{lane_occupied}" # add the num of lane used
        new_lst.append(sublist)
        # new_lst.append(lane_occupied) #modify the javascript
    return new_lst


def calculate_ratios(cycle_data_dict):
    ddr_ratios_dict = defaultdict(float)
    l2m_ratios_dict = defaultdict(float)
    prev_ddr_ratio, prev_l2m_ratio = None, None
    ddr_ratios, l2m_ratios = [], []

    for data in cycle_data_dict.values():
        # 使用生成器表达式进行过滤
        filtered_data = (
            record for record in data
            if record[0] in [1, 2] and "SYS" not in record[5] #GDMA/SDMA
        )

        for record in filtered_data:
            category, begin_time, end_time, _, _, _, _, _, _, uarch_bw, _, info, _, _ = record

            for time in range(begin_time + 1, end_time + 1):
                bw = float(uarch_bw)
                if 'DDR' in info and bw:
                    ddr_ratios_dict[time] += bw
                if 'L2M' in info and bw:
                    l2m_ratios_dict[time] += bw

    for time, bw in sorted(ddr_ratios_dict.items()):
        new_ddr_ratio = bw / 546
        if new_ddr_ratio != prev_ddr_ratio:
            ddr_ratios.append([time, new_ddr_ratio])
            prev_ddr_ratio = new_ddr_ratio

    for time, bw in sorted(l2m_ratios_dict.items()):
        new_l2m_ratio = bw / 1024
        if new_l2m_ratio != prev_l2m_ratio:
            l2m_ratios.append([time, new_l2m_ratio])
            prev_l2m_ratio = new_l2m_ratio

    return ddr_ratios, l2m_ratios
