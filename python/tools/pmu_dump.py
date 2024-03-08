#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import os
import collections
from enum import Enum
import struct
import csv
from argparse import ArgumentParser

from debugger.pmu_support import (
    decode_cmds,
    BaseCmd,
    EngineType,
)

g_input_num : int = 0

def parse_key_values(line:str):
    fields = line.split(",")
    item_map={}
    for f in fields:
        f = f.strip()
        if f.count("=")==0:
            continue
        elif f.count("=")>1:
            f=f.split(" ")
        else:
            f = [f]
        for s in f:
            s=s.strip()
            if s.endswith("GB/s"):
                continue
            elif s.endswith("us"):
                s = s[:-2]
            key, value = s.split("=")
            item_map[key] = value
    return item_map

def put_cmd_to_list(cmds_list : list, buffer : bytearray, type : int):
    assert type in {EngineType.TPU, EngineType.GDMA, EngineType.SDMA, EngineType.HAU}
    print("cmd_buffer_byary len : {}".format(len(buffer)))
    if(len(buffer) != 0):
        cmd = BaseCmd(buffer.copy(), type)
        cmds_list.append(cmd)
        buffer.clear()
'''
PIO mode
00_6908010000_xxxxxxxx
00_6908010004_xxxxxxxx
        ...
00_690801xxxx_xxxxxxxx
00_690801005c_xxxxxxxx
00_6908000000_xxxxxxxx
00_6908000004_xxxxxxxx
        ...

DES mode
00_6908010000_xxxxxxxx
00_6908010004_xxxxxxxx
        ...
00_690801xxxx_xxxxxxxx
00_690801005c_xxxxxxxx
GDMA
0x000cxxxxx0_0xXXXXXXXX
0x000cxxxxx4_0xXXXXXXXX
        ...
TPU
0x000cxxxxxx_0xXXXXXXXX
0x000cxxxxxx_0xXXXXXXXX
        ...
00_6908010000_xxxxxxxx
00_6908010004_xxxxxxxx
        ...
00_690801xxxx_xxxxxxxx
00_690801005c_xxxxxxxx
'''

def is_des_mode(cmd_file : str) -> bool :
    is_des_mode = False
    with open(cmd_file) as f:
        for line in f.readlines():
            line_strip = line.strip()
            if (line_strip == "GDMA" or line_strip == "TPU") :
                is_des_mode = True
                break
            elif ( len(line_strip) == 22 ) :
                is_des_mode = False
            else :
                assert(0)
    return is_des_mode


def parse_pio_mode_cmds_file(cmd_file : str, cmds_dict : dict):
    type = -1
    pre_type = -1
    cmd_buffer = bytearray()
    cmds_list = []
    last_core_no : int = -1
    with open(cmd_file) as f:
        for line in f.readlines():
            line = line.strip()
            data_list = line.split("_")
            assert ( len(data_list) == 3 )
            sub_str_cmd = data_list[1][2:]
            cur_core_no = int(sub_str_cmd[0])
            u32_addr = int (sub_str_cmd[1:], base=16)
            u32_value = int (data_list[2], base=16)
            value = u32_value.to_bytes(4,'little')
            if last_core_no != cur_core_no and last_core_no != -1 :
                cmds_dict[last_core_no] = cmds_list.copy()
                cmds_list.clear()
            addr_type = u32_addr & 0x80F0000
            if( addr_type == 0x8000000 ) :
                type = EngineType.TPU
            elif( addr_type == 0x8010000 ) :
                type = EngineType.GDMA
            elif( addr_type == 0x8020000 ) :
                type = EngineType.SDMA
            elif( addr_type == 0x8030000 ) :
                type = EngineType.HAU
            else:
                assert (0)
            # switch cmd_type
            if( pre_type != type and pre_type != -1) :
                put_cmd_to_list(cmds_list, cmd_buffer, pre_type)
            # no switch cmd_type
            elif( pre_type == type and type == EngineType.TPU and u32_addr == 0x8000000) :
                put_cmd_to_list(cmds_list, cmd_buffer, pre_type)
            elif( pre_type == type and type == EngineType.GDMA and u32_addr == 0x8010000) :
                put_cmd_to_list(cmds_list, cmd_buffer, pre_type)
            elif( pre_type == type and type == EngineType.SDMA and u32_addr == 0x8020000) :
                put_cmd_to_list(cmds_list, cmd_buffer, pre_type)
            elif( pre_type == type and type == EngineType.HAU and u32_addr == 0x8030000) :
                put_cmd_to_list(cmds_list, cmd_buffer, pre_type)

            pre_type = type
            cmd_buffer += value
            last_core_no = cur_core_no
        if( pre_type == type and pre_type != -1) :
            put_cmd_to_list(cmds_list, cmd_buffer, type)
            cmds_dict[cur_core_no] = cmds_list.copy()
            cmds_list.clear()
    print("parse pio file end")

def parse_des_mode_cmds_file(cmd_file : str, cmds_dict : dict):
    global g_input_num
    type = -1
    pre_type = -1
    pio_start = True
    cmd_buffer = bytearray()
    cmds_list = []
    cur_core_no : int = -1
    with open(cmd_file) as f:
        for line in f.readlines():
            line = line.strip()
            data_list = line.split("_")
            if len(data_list) == 2:
                sub_str_cmd = data_list[1][2:]
                u32_value = int (sub_str_cmd, base=16)
                value = u32_value.to_bytes(4,'little')
                cmd_buffer += value
            elif len(data_list) == 3:
                if pio_start == False:
                    put_cmd_to_list(cmds_list, cmd_buffer, type)
                    pio_start = True

                sub_str_cmd = data_list[1][2:]
                cur_core_no = int(sub_str_cmd[0])
                u32_addr = int (sub_str_cmd[1:], base=16)
                if( pre_type != type and pre_type != -1) :
                    put_cmd_to_list(cmds_list, cmd_buffer, pre_type)
                if( pre_type == type and type == EngineType.GDMA and u32_addr == 0x8010000) :
                    put_cmd_to_list(cmds_list, cmd_buffer, pre_type)
                    g_input_num += 1     #  count input param num
                if( pre_type == type and type == EngineType.SDMA and u32_addr == 0x8020000) :
                    put_cmd_to_list(cmds_list, cmd_buffer, pre_type)

                u32_value = int (data_list[2], base=16)
                value = u32_value.to_bytes(4,'little')
                cmd_buffer += value
                type = EngineType.GDMA
            elif len(data_list) == 1:
                put_cmd_to_list(cmds_list, cmd_buffer, type)
                if( data_list[0] == "GDMA"):
                    type = EngineType.GDMA
                elif( data_list[0] == "TPU"):
                    type = EngineType.TPU
                elif( data_list[0] == "SDMA"):
                    type = EngineType.SDMA
                elif( data_list[0] == "HAU"):
                    type = EngineType.HAU
                else:
                    assert(0)
                if pio_start == True :
                    pio_start = False
                    g_input_num += 1
            pre_type = type
    put_cmd_to_list(cmds_list, cmd_buffer, type)
    cmds_dict[cur_core_no] = cmds_list.copy()
    cmds_list.clear()
    print("parse des file end")

def read_cmds_file(cmd_file : str, cmds_dict : dict):
    des_mode = is_des_mode( cmd_file )
    if des_mode == True :
        parse_des_mode_cmds_file( cmd_file, cmds_dict)
    else :
        parse_pio_mode_cmds_file( cmd_file, cmds_dict)

def read_pmu_log(pmu_log_file : str) -> dict:
    profile_info = {}
    core_no : int = 0
    last_core_no : int = -1
    with open(pmu_log_file) as f:
        profile_gdma_info=[]
        profile_sdma_info=[]
        profile_tpu_info=[]
        start_str : str = ""
        for line in f.readlines():
            if line[0:2] in "[0][1][2][3][4][5][6][7]" :
                core_no = int(line[1])
                line = line[3:]
                if last_core_no != core_no and last_core_no != -1 :
                    profile_info[last_core_no] = [profile_gdma_info.copy(), profile_tpu_info.copy(), profile_sdma_info.copy()]
                    profile_gdma_info.clear()
                    profile_tpu_info.clear()
                    profile_sdma_info.clear()

            if line.startswith("---> gdma") :
                item_map = parse_key_values(line)
                cycle = item_map['cycle']
                start_time = item_map['start']
                end_time = item_map['end']
                interval = item_map['interval']
                profile_gdma_info.append([item_map['inst_id'], item_map['thread_id'], cycle, start_time, end_time, interval])
            elif line.startswith("---> tiu") :
                item_map = parse_key_values(line)
                cycle = item_map['cycle']
                start_time = item_map['start']
                end_time = item_map['end']
                interval = item_map['interval']
                bank_conflict = item_map['bank_conflict']
                profile_tpu_info.append([item_map['inst_id'], item_map['thread_id'], cycle, start_time, end_time, interval, bank_conflict])
            elif line.startswith("---> sdma") :
                item_map = parse_key_values(line)
                cycle = item_map['cycle']
                start_time = item_map['start']
                end_time = item_map['end']
                interval = item_map['interval']
                profile_sdma_info.append([item_map['inst_id'], item_map['thread_id'], cycle, start_time, end_time, interval])
            else :
                continue
            last_core_no = core_no
    profile_info[core_no] = [profile_gdma_info.copy(), profile_tpu_info.copy(), profile_sdma_info.copy()]
    return profile_info

def write_csv_file(cmds_dict : dict, pmu_profile_info : dict, save_file : str) :
    save_content = []
    for k in cmds_dict:
        cur_core_cmds_list = cmds_dict[k]
        cmds_count = len(cur_core_cmds_list)
        pmu_gdma_count = len(pmu_profile_info[k][0])
        pmu_tpu_count = len(pmu_profile_info[k][1])
        pmu_sdma_count = len(pmu_profile_info[k][2])
        pmu_gdma_inst_id : int = 0
        pmu_tpu_inst_id : int = 0
        pmu_sdma_inst_id : int = 0
        gdma_inst_id : int = 0
        tpu_inst_id : int = 0
        sdma_inst_id : int = 0
        for i in range (cmds_count) :
            entry = cur_core_cmds_list[i]
            type, cmd_info = entry
            item = []
            if(type == EngineType.GDMA):
                if pmu_gdma_inst_id >= pmu_gdma_count :
                    item = [str(i), gdma_inst_id, "GDMA", "", "", "", "", "", ""]
                    item += cmd_info
                else :
                    gdma_item = pmu_profile_info[k][0][pmu_gdma_inst_id]
                    inst_id = gdma_item[0]
                    thread_id = gdma_item[1]
                    if(str(gdma_inst_id) == inst_id) :
                        item = [str(i), inst_id, "GDMA", thread_id, gdma_item[2], gdma_item[3], gdma_item[4], gdma_item[5], ""]
                        item += cmd_info
                        pmu_gdma_inst_id += 1
                    elif(gdma_inst_id < int(inst_id)) :
                        item = [str(i), gdma_inst_id, "GDMA", "", "", "", "", "", ""]
                        item += cmd_info
                    else :
                        assert(0)
                save_content.append(item)
                gdma_inst_id += 1
            elif(type == EngineType.TPU) :
                if pmu_tpu_inst_id >= pmu_tpu_count :
                    item = [str(i), tpu_inst_id, "TPU", "", "", "", "", "", ""]
                    item += cmd_info
                else :
                    tpu_item = pmu_profile_info[k][1][pmu_tpu_inst_id]
                    inst_id = tpu_item[0]
                    thread_id = tpu_item[1]
                    ''''''
                    if( str(tpu_inst_id) == inst_id) :
                        item = [str(i), inst_id, "TPU", thread_id, tpu_item[2], tpu_item[3], tpu_item[4], tpu_item[5], tpu_item[6]]
                        item += cmd_info
                        pmu_tpu_inst_id += 1
                    elif(tpu_inst_id < int(inst_id)) :
                        item = [str(i), tpu_inst_id, "TPU", "", "", "", "", "", ""]
                        item += cmd_info
                    else :
                        assert(0)
                save_content.append(item)
                tpu_inst_id += 1
            elif(type == EngineType.SDMA):
                if pmu_sdma_inst_id >= pmu_sdma_count :
                    item = [str(i), sdma_inst_id, "SDMA", "", "", "", "", "", ""]
                    item += cmd_info
                else :
                    sdma_item = pmu_profile_info[k][2][pmu_sdma_inst_id]
                    inst_id = sdma_item[0]
                    thread_id = sdma_item[1]
                    if(str(sdma_inst_id) == inst_id) :
                        item = [str(i), inst_id, "SDMA", thread_id, sdma_item[2], sdma_item[3], sdma_item[4], sdma_item[5], ""]
                        item += cmd_info
                        pmu_sdma_inst_id += 1
                    elif(sdma_inst_id < int(inst_id)) :
                        item = [str(i), sdma_inst_id, "SDMA", "", "", "", "", "", ""]
                        item += cmd_info
                    else :
                        assert(0)
                save_content.append(item)
                sdma_inst_id += 1
            else :
                assert(0)

    with open(save_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['No', 'inst_id', 'cmd_type', 'thread_id', 'cycle', 'start(us)', 'end(us)', 'interval(us)', 'bank_conflict',
        'cmd_name', 'id_dep_info', 'res0_info', 'opd0_info', 'opd1_info', 'opd2_info','opd3_info','opd4_info','opd5_info'])
        count = len(save_content)
        for i in range ( count ) :
            item = save_content[i]
            writer.writerow( [ str( item[j] ) for j in range(len(item)) ])
    print("save file finished")

def argparser():
    usage = "usage: %prog -f cmd_file  -p pmu_file  -o  out_file"
    optparser = ArgumentParser(usage)
    optparser.add_argument('-f',  '--cmd_file',
                         action='store', dest='cmd_file', required=True)
    optparser.add_argument('-p',  '--pmu_file',
                         action='store', dest='pmu_file', required=True)
    optparser.add_argument('-o',  '--out_file',
                         action='store', dest='out_file', required=True)
    optparser.add_argument('-c',  '--chip',
                         action='store', dest='chip', default='BM1690', help='only support bm1690')

    return optparser.parse_args()

if __name__ == "__main__":
    args = argparser()
    decoded_cmds_dict = {}
    pre_decode_cmds_dict = {}
    pmu_profile_dict = {}
    read_cmds_file(args.cmd_file, pre_decode_cmds_dict)
    decoded_cmds_dict = decode_cmds(args.chip, pre_decode_cmds_dict)

    pmu_profile_dict = read_pmu_log(args.pmu_file)
    write_csv_file(decoded_cmds_dict, pmu_profile_dict, args.out_file)
