# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
import os
import sys
import json
import argparse
import warnings
import re
import shutil
import numpy as np
import pandas as pd
from intervaltree import Interval, IntervalTree
from concurrent.futures import ProcessPoolExecutor
from bs4 import BeautifulSoup

from utils.utils import *
from utils.power import *

warnings.filterwarnings('ignore')

def findOverlap(df_a, df_b):
    tree_b = IntervalTree()

    for index, row in df_a.iterrows():
        if row['FuncType'] != 'SYS':
            tree_b[row['StartCycle']:row['EndCycle']+1] = (index, row['BandWidth(GB/s)'])

    overlaps = []

    for index_a, row_a in df_b.iterrows():
        if row_a['FuncType'] != 'SYS':
            overlapping_intervals = tree_b[row_a['StartCycle']:row_a['EndCycle']+1]

            for interval in overlapping_intervals:
                index_b, bw1 = interval.data
                start_b, end_b = interval.begin, interval.end
                overlap_start = max(row_a['StartCycle'], start_b)
                overlap_end = min(row_a['EndCycle'], end_b)

                if overlap_end > overlap_start:
                    bw2 = row_a['BandWidth(GB/s)']
                    sim_power = round((bw1+bw2)/1608*(26-8)+8,4)
                    overlaps.append(['k2k_overlap', overlap_start, overlap_end, overlap_end - overlap_start, '-', bw1+bw2, sim_power]
                    )
    return overlaps

def parse_excel(xlsx):
    try:
        sheets= xlsx.sheet_names
        sheet_data = {}
        for sheet in sheets:
            if sheet == 'summary':
                continue
            df = pd.read_excel(xlsx, sheet_name = sheet)
            data_dict = processSheet(df, sheet)
            sheet_data[sheet] = data_dict
        return sheet_data
    except:
        print("Error tokenizing data!")
        sys.exit(1)

def processSheet(df,type):
    if df is None:
        return None
    header_row_index = df.index[df.isin(["FuncType"]).any(axis=1)].tolist()[0]
    df.columns = df.iloc[header_row_index]
    core_indices = df.index[df['FuncType'].str.contains("CORE", na=False)].tolist()
    core_indices.append(df.index[-1] + 1)
    core_dfs = {}
    for i in range(len(core_indices) - 1):
        core_name = df.iloc[core_indices[i]]['FuncType']
        temp = df.iloc[core_indices[i] + 2 : core_indices[i + 1]-1].reset_index(drop=True)
        temp.insert(0, 'Ip', type)
        if 'tiu' in type:
            temp.insert(0, 'SimPower(w)', 0)
            temp.insert(0, 'uArchRate', 0)
        core_dfs[core_name] = temp.sort_values(by='StartCycle')
    return core_dfs

## process tiu
def match_func_name(str1, str2):
    if not isinstance(str1, str):
        str1 = str(str1) if str1 is not None else ''
    if not isinstance(str1, str):
        str2 = str(str2) if str2 is not None else ''
    str1_clean = re.sub(r"[\d_]", "", str1).lower()
    str2_clean = re.sub(r"[\d_]", "", str2).lower()

    return str1_clean in str2_clean or str2_clean in str1_clean

def compare_shape(pld_row, perf_row):
    values = [perf_row['des_res0_n'], perf_row['des_res0_c'], perf_row['des_res0_h'], perf_row['des_res0_w']]
    nan_values = [pd.isna(x) for x in values]
    if all(nan_values):
        compare_value = "0x0x0x0"
    else:
        compare_value = 'x'.join(str(int(v)) if not pd.isna(v) else '1' for v in values)
    compare_value = "{" + compare_value + "}"
    res_shape_value = pld_row['res0_shape']
    if compare_value != res_shape_value:
        return False
    return True

def read_data_from_perfdoc(sheetname, perfdoc,tiu_power=None):
    if sheetname in perfdoc.sheet_names:
        perfdf = pd.read_excel(perfdoc, sheet_name=sheetname, header=None)
    else:
        #PerfAI single core case
        first_sheet = sheetname[:-1] + '0'
        perfdf = pd.read_excel(perfdoc, sheet_name=first_sheet, header=None)
    header_row_index = perfdf.index[perfdf.isin(["Engine Id"]).any(axis=1)].tolist()[0]
    perfdf.columns = perfdf.iloc[header_row_index]
    perfdf = perfdf.iloc[header_row_index+1:]
    perfdf.insert(0, 'Ip', sheetname[:-2].lower())
    if tiu_power is not None:
        perfdf = perfdf.merge(tiu_power, on=['Function Type', 'Function Name','Data Type'], how='left') #, 'des_opt_opd0_const'
        perfdf['Power'] = perfdf['Power'].fillna(0)
        # perfdf[['in_prec', 'out_prec']] = perfdf['Data Type'].str.split(" -> ", expand=True)
    return perfdf


def match_dma_data_from_perf(row1, perfdf): #row1：pld data
    dma_function = {# pld:perf
    'load': ['tensorMv','tensorLd'],
    'store':['tensorSt', 'add'],
    'copy':['tensorLd','tensorSt','tensorMv'],
    'send_msg':['GDMA_SYS_SEND'],
    'wait_msg':['GDMA_SYS_WAIT'],
    'fill_const':['tensorLd'],
    'end': [],
    }
    for idx2, row2 in perfdf.iterrows():
        # print(row1['FuncName'],row2['Function Name'],dma_function.get(row1['FuncName']))
        if row1['FuncName'] not in dma_function.keys():
            print('pld func_name not in the dict:',row1['FuncName'])
        elif (row2['Function Name'] in dma_function.get(row1['FuncName']) and
            match_func_name(row1['Direction'], row2['Direction'])):
                # and row1['Size(B)'] == row2['DMA data size(B)']):
            #record Perf Cycle(golden)
            PerfStart = row2['Start Cycle']
            PerfEnd = row2['End Cycle']
            PerfCycles = row2['Simulator Cycle']
            PerfCmd = row2['Cmd Id']
            return PerfStart,PerfEnd,PerfCycles,PerfCmd
        # elif(row2['Function Name'] in dma_function.get(row1['FuncName'])):
        #     print('DMA:', row1['FuncName'],row1['Direction'],row2['Direction'],match_func_name(row1['Direction'], row2['Direction']), row1['Size(B)'] == row2['DMA data size(B)'])
    return None,None,None,None

#get uRate, sim cycles, etc from PerfAI.doc and match the pld data(row1)
def match_tiu_data_from_perf(row1, perftiu):
    # pld:perf
    tiu_function = {'cast':'data_convert', 'static_bc':'static_broadcast','avg_pooling':'average_pooling',
            'select_great':'comp_sel_gt','select_equal':'comp_sel_eq','bc_lane_broad':'lane_broadcast',
            'abs':'absolute_value', 'Normal':'conv_normal'}
    for idx2, row2 in perftiu.iterrows():
        if (match_func_name(row1['FuncName'], row2['Function Name']) or row2['Function Name'] == tiu_function.get(row1['FuncName']) and
            # row1['DataType'] == row2['Data Type'] and
            compare_shape(row1, row2)):
            power = float(row2['Power']) if 'Power' in row2 and not pd.isna(row2['Power']) else 0
            pld_sim_power = round(percent_to_float(row2['uArch Rate']) * power * 64,4)
            #record Perf Cycle(golden)
            PerfStart = row2['Start Cycle']
            PerfEnd = row2['End Cycle']
            PerfCycles = row2['Simulator Cycle']
            PerfCmd = row2['Cmd Id']
            return row2['uArch Rate'], pld_sim_power, PerfStart,PerfEnd,PerfCycles,PerfCmd
        # elif(not match_func_name(row1['FuncName'], row2['Function Name']) and row1['DataType'] == row2['Data Type'] and compare_shape(row1, row2)):

    return 0,0,None,None,None,None

def cal_power(corename, pld_cores, key, perf_path, calculate_sim_power_functions, tiu_header, dma_header,perf_tiu_columns,perf_dma_columns):
    results = [{},{}] #pld, perf
    sheetname = f'{key.upper()}_{corename[-1]}'
    perfdoc = pd.ExcelFile(perf_path)
    if key == 'tiu':
        perftiu = read_data_from_perfdoc(sheetname, perfdoc)
        perftiu.insert(0, 'Core', corename)
        perftiu['SimPower(w)'] = perftiu.apply(lambda row: round(float(row['Sim Power(W)']) , 4), axis=1)
        results[1][corename] = perftiu[perf_tiu_columns].values.tolist()
        if pld_cores is not None:
            pld_cores['in_prec'] = pld_cores['in_prec'].apply(lambda x: 'None' if x == '-' else x)
            pld_cores['out_prec'] = pld_cores['out_prec'].apply(lambda x: 'None' if x == '-' else x)
            pld_cores['DataType'] = pld_cores['in_prec'] + " -> " + pld_cores['out_prec']
            pld_cores[['uArchRate', 'SimPower(w)','Perf_StartCycle','Perf_EndCycle','Perf_Cycles','Perf_CmdId']] = pld_cores.apply(lambda row: match_tiu_data_from_perf(row, perftiu), axis=1, result_type='expand')
            pld_cores.insert(0, 'Core', corename)
            results[0][corename] = pld_cores[tiu_header].values.tolist()
    else:
        perfdma = read_data_from_perfdoc(sheetname, perfdoc)
        func_to_call = calculate_sim_power_functions[key]
        perf_new_columns = perfdma.apply(func_to_call, axis=1)
        perfdma.insert(0, 'Core', corename)
        perfdma = perfdma.join(perf_new_columns)
        results[1][corename] = perfdma[perf_dma_columns].values.tolist()
        if pld_cores is not None:
            pld_cores[['Perf_StartCycle','Perf_EndCycle','Perf_Cycles','Perf_CmdId']] = pld_cores.apply(lambda row: match_dma_data_from_perf(row, perfdma), axis=1, result_type='expand')
            new_columns = pld_cores.apply(func_to_call, axis=1) #gdma_pw, sdma_pw,cdma_pw, noc_pw, ddr_pw,l2m_pw
            pld_cores = pld_cores.join(new_columns)
            pld_cores.insert(0, 'Core', corename)
            results[0][corename] = pld_cores[dma_header].values.tolist()
    return results


def get_power_from_data_parallel(sheet_data, perf_path, tiu_header, dma_header,perf_tiu_columns,perf_dma_columns):
    calculate_sim_power_functions = {
        'gdma': gdma_calculate_sim_power,
        'sdma': sdma_calculate_sim_power,
        'cdma': cdma_calculate_sim_power,
    }
    try:
        perfdoc = pd.ExcelFile(perf_path)
    except FileNotFoundError:
        print(f"PerfAI.doc not found.")
    except Exception as e:
        print(f"An error occurred while reading PerfAI.doc: {e}")

    ip_list = [name.split('_0')[0].lower() for name in perfdoc.sheet_names if name.endswith('_0')]
    suffix_numbers = set(int(name.split('_')[-1]) for name in perfdoc.sheet_names if '_' in name and name.split('_')[-1].isdigit())
    pld_dict_to_js = {f'CORE{i}': [] for i in range(max(suffix_numbers) + 1)}
    perf_dict_to_js = {f'CORE{i}': [] for i in range(max(suffix_numbers) + 1)}

    with ProcessPoolExecutor() as executor:
        futures = []
        if sheet_data:
            for key, pld_cores_dict in sheet_data.items():
                for corename, pld_cores in pld_cores_dict.items():
                    future = executor.submit(cal_power, corename, pld_cores, key, perf_path, calculate_sim_power_functions, tiu_header, dma_header, perf_tiu_columns,perf_dma_columns)
                    futures.append(future)
        else: #perf data only
            for key in ip_list:
                for corename in perf_dict_to_js.keys():
                    future = executor.submit(cal_power, corename, None, key, perf_path, calculate_sim_power_functions, tiu_header, dma_header, perf_tiu_columns,perf_dma_columns)
                    futures.append(future)
        for future in futures:
            result = future.result()
            plddict = result[0]
            perfdict = result[1]
            for corename, core_result in plddict.items():
                pld_dict_to_js[corename].extend(core_result)
            for corename, core_result in perfdict.items():
                perf_dict_to_js[corename].extend(core_result)

    if 'gdma' in sheet_data.keys() and 'sdma' in sheet_data.keys():  #check the k2k gdma+sdma case
        gdma_dict = sheet_data['gdma']
        sdma_dict = sheet_data['sdma']
        for key in gdma_dict.keys():
            if key in sdma_dict.keys():
                overlap = findOverlap(gdma_dict[key],sdma_dict[key])
                if overlap:
                    print('Found overlapping case and calculated sim power')
                    pld_dict_to_js[key].extend(overlap)
    return ip_list, pld_dict_to_js, perf_dict_to_js


# power: watt (j/s)
# cycle : 1ns = 10e-09s
# Energy: nJ -> mJ 毫焦耳
def cal_total_energy(dict_to_js):
    energy = []
    missing_item = set()
    perf_missing_cmds = []
    for key, list_of_lists in dict_to_js.items():
        temp = 0
        for lst in list_of_lists:
            if pd.isna(lst[6]): #lst[6]=cycles
                missing_item.add(lst[3]) #function name
            if lst[1] == 'tiu':
                if lst[2] != 'SYS' and 'sys' not in lst[3] and lst[-1] == 0:
                    perf_missing_cmds.append(lst)
                temp += lst[6] * lst[-1] #SimPower
            else:
                if pd.isna(sum(lst[-6:])):
                    print(lst)
                temp += lst[6] * sum(lst[-6:])  #gdma_pw, sdma_pw,cdma_pw, noc_pw, ddr_pw,l2m_pw
        converted = temp /(10 ** 6) #millijoule
        energy.append(round(converted,2))
    energy = [0 if np.isnan(x) else x for x in energy]
    return energy, missing_item, perf_missing_cmds


def percent_to_float(percent_str):
    number_str = percent_str.replace('%', '').strip()
    return float(number_str) / 100

def remove_specific_entries(list_of_lists):
    list_of_lists.sort(key=lambda x: x[4])
    index_to_remove_before = next((i for i, sublist in enumerate(list_of_lists) if sublist[3] == 'wait_msg'), None)
    index_to_remove_after = next((i for i, sublist in enumerate(list_of_lists) if sublist[3] == 'end'), None)

    if index_to_remove_before is not None:
        wait_start_value = list_of_lists[index_to_remove_before][4]
        list_of_lists = [sublist for sublist in list_of_lists if sublist[4] > wait_start_value]
    if index_to_remove_after is not None:
        end_start_value = list_of_lists[index_to_remove_after][4]
        list_of_lists = [sublist for sublist in list_of_lists if sublist[4] < end_start_value]

    return list_of_lists

def sort_and_update_lists(list_of_lists, n):
    updated_lists = []
    for lst in list_of_lists:
        updated_list = lst.copy()
        updated_list[4] += n  # 第五个元素start cycle增加n
        updated_list[5] += n  # 第六个元素end cycle增加n
        updated_lists.append(updated_list)
    updated_lists.sort(key=lambda x: x[5])
    return updated_lists, updated_lists[-1][5] if updated_lists else n  # 返回新的列表和n的更新值


def process_multiple_sets(arg_sets, tiu_header, dma_header, perf_tiu_columns, perf_dma_columns):
    # 初始化最终结果存储
    final_ip_list = []
    final_pld_dict_to_js = {}
    final_perf_dict_to_js = {}
    n_dict1 = {}
    n_dict2 = {}
    pld_concat_timemark = {}
    perf_concat_timemark = {}
    for args in arg_sets:
        # 假设 args 是一个包含 pldfile 和 perf 路径的字典
        pldfile = os.path.abspath(args['pld']) if args['pld'] else ''
        perf_path = os.path.abspath(args['perf'])
        base = os.path.basename(perf_path)# 分离文件名和扩展名
        pattern_name, _ = os.path.splitext(base)
        pld_concat_timemark[pattern_name] = []
        perf_concat_timemark[pattern_name] = []
        if pldfile:
            try:
                pldfile = pd.ExcelFile(pldfile)
                sheetdata = parse_excel(pldfile)
            except FileNotFoundError:
                print("Error: File not found. Please provide a valid xlsx file path!")
                sys.exit(1)
        else:
            sheetdata = {}

        ip_list, pld_dict_to_js, perf_dict_to_js = get_power_from_data_parallel(sheetdata, perf_path, tiu_header, dma_header, perf_tiu_columns, perf_dma_columns)
        # 合并结果
        final_ip_list.extend(x for x in ip_list if x not in final_ip_list)

        for key in pld_dict_to_js:
            pld_dict_to_js[key] = remove_specific_entries(pld_dict_to_js[key]) #删除pld第一个wait之前和最后一个end之后的指令


        for key in {**pld_dict_to_js, **perf_dict_to_js}:
            if key not in n_dict1:
                n_dict1[key] = 0
            if key not in n_dict2:
                n_dict2[key] = 0
        for key, value in pld_dict_to_js.items():
            sorted_value, last_n = sort_and_update_lists(value, n_dict1[key])
            final_pld_dict_to_js.setdefault(key, []).extend(sorted_value)
            n_dict1[key] = last_n+1
            pld_concat_timemark[pattern_name].append(last_n+1)

        for key, value in perf_dict_to_js.items():
            sorted_value, last_n = sort_and_update_lists(value, n_dict2[key])
            final_perf_dict_to_js.setdefault(key, []).extend(sorted_value)
            n_dict2[key] = last_n+1
            perf_concat_timemark[pattern_name].append(last_n+1)

    # 排序
    for key in final_pld_dict_to_js:
        final_pld_dict_to_js[key].sort(key=lambda x: (x[4], x[5]))  # 确认第5个值是start cycle
    for key in final_perf_dict_to_js:
        final_perf_dict_to_js[key].sort(key=lambda x: (x[4], x[5]))

    return final_ip_list, final_pld_dict_to_js, final_perf_dict_to_js,pld_concat_timemark,perf_concat_timemark

def run_power(ptpx, cmds, pld, perf, name, version):
    # Ensure there is at least one perf argument
    if len(perf) == 0:
        print("Error: At least one data source mush be provided. Please --perf to provide PerfAI.doc path")
        exit(1)
    if pld:
        if len(pld) != len(perf):
            print("Each pld file has to match with one perf.doc. Please check your input file paths")
            exit(1)

    pld_list = pld if pld else [None] * len(perf)
    arg_sets = [{
        'pld': pld_pair,
        'perf': perf_pair,
    } for pld_pair, perf_pair in zip(pld_list, perf)]

    tiu_header = ['Core', 'Ip', 'FuncType','FuncName', 'StartCycle', 'EndCyclee', 'Cycle','CmdId', 'Perf_StartCycle', 'Perf_EndCycle','Perf_Cycles','Perf_CmdId', 'DataType', 'uArchRate','SimPower(w)']
    dma_header = ['Core', 'Ip', 'FuncType','FuncName', 'StartCycle', 'EndCycle', 'Cycle', 'CmdId','Perf_StartCycle', 'Perf_EndCycle', 'Perf_Cycles','Perf_CmdId', 'Direction', 'BandWidth(GB/s)', 'gdma_pw(w)','sdma_pw(w)','cdma_pw(w)', 'noc_pw(w)', 'ddr_pw(w)', 'l2m_pw(w)']
    perf_tiu_columns = ['Core', 'Ip', 'Function Type', 'Function Name', 'Start Cycle', 'End Cycle', 'Simulator Cycle','Cmd Id', 'Data Type', 'uArch Rate','SimPower(w)']
    perf_dma_columns = ['Core', 'Ip', 'Function Type', 'Function Name', 'Start Cycle', 'End Cycle', 'Simulator Cycle','Cmd Id', 'Direction', 'DDR Bandwidth(GB/s)','L2M Bandwidth(GB/s)','gdma_pw(w)','sdma_pw(w)','cdma_pw(w)', 'noc_pw(w)', 'ddr_pw(w)', 'l2m_pw(w)']

    if ptpx and cmds:
        tiu_power = get_tiu_power(ptpx,cmds)

    out_path = os.path.join(os.getcwd(), 'PerfWeb_Power')
    os.makedirs(out_path, exist_ok=True)
    templates_dir = os.path.abspath(__file__).replace("run_power.py","templates")
    htmlfiles = [os.path.join(templates_dir, 'echarts.min.js'), os.path.join(templates_dir, 'jquery-3.5.1.min.js')]
    for f in htmlfiles:
        shutil.copy2(f, out_path)
    html_path = os.path.join(templates_dir, 'power_standard.html')

    final_ip_list, final_pld_dict_to_js, final_perf_dict_to_js,pld_concat_timemark,perf_concat_timemark = process_multiple_sets(arg_sets, tiu_header, dma_header, perf_tiu_columns,perf_dma_columns)
    prepareJson(version,name,final_ip_list, final_pld_dict_to_js,final_perf_dict_to_js,tiu_header,dma_header,perf_tiu_columns,perf_dma_columns,pld_concat_timemark,perf_concat_timemark,out_path)
    prepareHtml(html_path,name,out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ptpx', type=str, help='ptpx file path that records the latest power data')
    parser.add_argument('--cmds', type=str, help='cmds file path that paired with ptpx file')
    parser.add_argument('--pld', action='append',type=str, help='pld_data file path')
    parser.add_argument('--perf', action='append',type=str, required=True, help='perfdoc(excel) path of same pld model')
    parser.add_argument('--reg',type=str,default='',help='The folder path that contains tiuRegInfo、dmaRegInfo txt files.')
    parser.add_argument('--name', '-n', type=str, required=True, help='name of the js file that stores the power data')
    parser.add_argument('--version', '-v', type=str, default='', help='AI compiler commit ID. Please provide this info if you need to present it on web')
    args = parser.parse_args()
    run_power(args.ptpx, args.cmds, args.pld, args.perf, args.name, args.version)
