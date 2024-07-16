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

from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup
from src.tiu import TIU
from src.dma import DMA
from utils.utils import *


perf_tiu_columns = ['Core', 'Ip', 'Function Type', 'Function Name', 'Start Cycle', 'End Cycle', 'Asic Cycle','Cmd Id', 'Data Type', 'uArch Rate','SimPower(w)']
perf_dma_columns = ['Core', 'Ip', 'Function Type', 'Function Name', 'Start Cycle', 'End Cycle', 'Asic Cycle','Cmd Id', 'Direction', 'DDR Bandwidth(GB/s)','L2M Bandwidth(GB/s)','gdma_pw(w)','sdma_pw(w)','cdma_pw(w)', 'noc_pw(w)', 'ddr_pw(w)', 'l2m_pw(w)']


def gdma_calculate_sim_power(row):
    if 'BandWidth(GB/s)' in row:
        bw =  float(row['BandWidth(GB/s)'])
        ddr_bw = bw
        l2m_bw = bw
    else:
        ddr_bw = float(row['DDR Bandwidth(GB/s)'])
        l2m_bw = float(row['L2M Bandwidth(GB/s)'])
        bw = ddr_bw + l2m_bw

    try:
        direction = re.sub(r"(?<=LMEM)\d+", "", str(row['Direction'])) if row['Direction'] is not None else ''
        ip_pw = bw / 128 * (0.64 - 0.118)
        lmem_pw = bw / 128 * (0.2 - 0.11)
        gdma_pw = round(ip_pw + lmem_pw,4)
        if direction == 'DDR->LMEM':
            noc_pw = round(bw / 546 * (16 - 8),4)
            ddr_pw = round((ddr_bw / 546 * (12.6 - 7.2)) + (ddr_bw / 546 * (30.9 - 0.88)),4)  # DDR read
            return pd.Series({'gdma_pw(w)': gdma_pw,'sdma_pw(w)':0,'cdma_pw(w)':0, 'noc_pw(w)': noc_pw, 'ddr_pw(w)': ddr_pw, 'l2m_pw(w)': 0})
        elif direction == 'LMEM->DDR':
            noc_pw = round(bw / 546 * (16 - 8),4)
            ddr_pw = round((ddr_bw / 546 * (12.6 - 7.2)) + (ddr_bw / 546 * (20.3 - 0.88)),4)  # DDR write
            return pd.Series({'gdma_pw(w)': gdma_pw,'sdma_pw(w)':0,'cdma_pw(w)':0, 'noc_pw(w)': noc_pw, 'ddr_pw(w)': ddr_pw, 'l2m_pw(w)': 0})
        elif direction == 'L2M->LMEM':
            noc_pw = round(bw / 1024 * (16 - 8),4)
            l2m_pw = round((l2m_bw / 128 * (0.25 - 0.11)),4)  # L2M read
            return pd.Series({'gdma_pw(w)': gdma_pw, 'sdma_pw(w)':0,'cdma_pw(w)':0,'noc_pw(w)': noc_pw, 'ddr_pw(w)': 0, 'l2m_pw(w)': l2m_pw})
        elif direction == 'LMEM->L2M':
            noc_pw = round(bw / 1024 * (16 - 8),4)
            l2m_pw = round((l2m_bw / 128 * (0.29 - 0.11)),4) # L2M write
            return pd.Series({'gdma_pw(w)': gdma_pw, 'sdma_pw(w)':0,'cdma_pw(w)':0,'noc_pw(w)': noc_pw, 'ddr_pw(w)': 0, 'l2m_pw(w)': l2m_pw})
        else:  # sys
            return pd.Series({'gdma_pw(w)': round(ip_pw,4), 'sdma_pw(w)':0,'cdma_pw(w)':0,'noc_pw(w)': 0, 'ddr_pw(w)': 0, 'l2m_pw(w)': 0})  # ip_pw, k2k, ddr, l2m
    except ValueError:
        print(f'ValueError: bandwidth is not a number, received: {bw}')
        return pd.Series({'gdma_pw(w)': 0, 'sdma_pw(w)':0,'cdma_pw(w)':0,'noc_pw(w)': 0, 'ddr_pw(w)': 0, 'l2m_pw(w)': 0})

def sdma_calculate_sim_power(row):
    if 'BandWidth(GB/s)' in row:
        bw =  float(row['BandWidth(GB/s)'])
        ddr_bw = bw
        l2m_bw = bw
    else:
        ddr_bw = float(row['DDR Bandwidth(GB/s)'])
        l2m_bw = float(row['L2M Bandwidth(GB/s)'])
        bw = ddr_bw + l2m_bw
    try:
        ip_pw = round(bw / 128 * (0.48 - 0.05),4)
        direction = str(row['Direction']) if row['Direction'] is not None else ''
        if direction == 'DDR->DDR':
            noc_pw = round(bw / 546 * (14 - 8),4)
            ddr_pw = round((ddr_bw / 546 * (12.6 - 7.2)) * 2 + (ddr_bw / 546 * (30.9 - 0.88)) + (ddr_bw / 546 * (20.3 - 0.88)),4)  # DDR read+write
            return pd.Series({'gdma_pw(w)': 0, 'sdma_pw(w)':ip_pw, 'cdma_pw(w)':0,'noc_pw(w)': noc_pw, 'ddr_pw': ddr_pw, 'l2m_pw': 0})
        elif direction == 'DDR->L2M':  # DDR read + L2M write
            noc_pw = round(bw / 546 * (16 - 8),4)
            ddr_pw = round((ddr_bw / 546 * (12.6 - 7.2)) + (ddr_bw / 546 * (30.9 - 0.88)),4)
            l2m_pw = round((l2m_bw / 128 * (0.29 - 0.11)),4)
            return pd.Series({'gdma_pw(w)': 0, 'sdma_pw(w)':ip_pw,'cdma_pw(w)':0, 'noc_pw(w)': noc_pw, 'ddr_pw(w)': ddr_pw, 'l2m_pw(w)': l2m_pw})
        elif direction == 'L2M->DDR':  # L2M read + DDR write
            noc_pw = round(bw / 1024 * (16 - 8),4)
            ddr_pw = round((ddr_bw / 546 * (12.6 - 7.2)) + (ddr_bw / 546 * (20.3 - 0.88)),4)
            l2m_pw = round((l2m_bw / 128 * (0.25 - 0.11)),4)
            return pd.Series({'gdma_pw(w)': 0, 'sdma_pw(w)':ip_pw, 'cdma_pw(w)':0, 'noc_pw(w)': noc_pw, 'ddr_pw(w)': ddr_pw, 'l2m_pw(w)': l2m_pw})
        elif direction == 'L2M->L2M':
            l2m_pw = round((l2m_bw / 128 * (0.25 - 0.11)) + (l2m_bw / 128 * (0.29 - 0.11)),4)  # L2M read+write
            return pd.Series({'gdma_pw(w)': 0, 'sdma_pw(w)':ip_pw, 'cdma_pw(w)':0, 'noc_pw(w)': 0, 'ddr_pw(w)': 0, 'l2m_pw(w)': l2m_pw})
        else:  # sys
            return pd.Series({'gdma_pw(w)': 0, 'sdma_pw(w)':ip_pw,'cdma_pw(w)':0, 'noc_pw(w)': 0, 'ddr_pw(w)': 0, 'l2m_pw(w)': 0})
    except ValueError:
        print(f'ValueError: bandwidth is not a number, received: {bw}')
        return pd.Series({'gdma_pw(w)': 0, 'sdma_pw(w)':0,'cdma_pw(w)':0, 'noc_pw(w)': 0, 'ddr_pw(w)': 0, 'l2m_pw(w)': 0})

def cdma_calculate_sim_power(row):
    if 'BandWidth(GB/s)' in row:
        bw =  float(row['BandWidth(GB/s)'])
    else:
        ddr_bw = float(row['DDR Bandwidth(GB/s)'])
        l2m_bw = float(row['L2M Bandwidth(GB/s)'])
        bw = ddr_bw + l2m_bw
    try:
        ip_pw = round(bw / 64 * (0.17 - 0.05),4)
        return pd.Series({'gdma_pw': 0, 'sdma_pw':0,'cdma_pw':ip_bw, 'noc_pw': 0, 'ddr_pw': 0, 'l2m_pw': 0})
    except ValueError:
        print(f'ValueError: bandwidth is not a number, received: {bw}')
        return pd.Series({'gdma_pw': 0, 'sdma_pw':0,'cdma_pw':0, 'noc_pw': 0, 'ddr_pw': 0, 'l2m_pw': 0})


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


def process_tiu(tiu_instances, core_idx, corename):
    tiu_list = []
    if core_idx < len(tiu_instances):
        tiudf = tiu_instances[core_idx]
        tiudf.insert(0, 'Core', corename)
        tiudf.insert(1, 'Ip', 'tiu')
        tiudf['SimPower(w)'] = tiudf.apply(lambda row: round(float(row['Sim Power(W)']) , 4), axis=1)
        tiu_list = tiudf[perf_tiu_columns].values.tolist()
    return tiu_list


def process_gdma(gdma_instances, core_idx, corename):
    gdma_list = []
    if core_idx < len(gdma_instances):
        gdmadf = gdma_instances[core_idx]
        perf_new_columns = gdmadf.apply(gdma_calculate_sim_power, axis=1)
        gdmadf.insert(0, 'Core', corename)
        gdmadf.insert(1, 'Ip', 'gdma')
        gdmadf = gdmadf.join(perf_new_columns)
        gdma_list = gdmadf[perf_dma_columns].values.tolist()
    return gdma_list

def process_sdma(sdma_instances, core_idx, corename):
    sdma_list = []
    if core_idx < len(sdma_instances):
        sdmadf = sdma_instances[core_idx]
        perf_new_columns = sdmadf.apply(sdma_calculate_sim_power, axis=1)
        sdmadf.insert(0, 'Core', corename)
        sdmadf.insert(1, 'Ip', 'sdma')
        sdmadf = sdmadf.join(perf_new_columns)
        sdma_list = sdmadf[perf_dma_columns].values.tolist()
    return sdma_list

def process_cdma(cdma_instances, core_idx, corename):
    cdma_list = []
    if core_idx < len(cdma_instances):
        cdmadf = cdma_instances[core_idx]
        perf_new_columns = cdmadf.apply(cdma_calculate_sim_power, axis=1)
        cdmadf.insert(0, 'Core', corename)
        cdmadf.insert(1, 'Ip', 'cdma')
        cdmadf = cdmadf.join(perf_new_columns)
        cdma_list = cdmadf[perf_dma_columns].values.tolist()
    return cdma_list


def read_data_and_cal_power(core_idx, tiu_instances, gdma_instances, sdma_instances, cdma_instances):
    with ThreadPoolExecutor(max_workers=4) as executor:
        # 对每种实例并行处理
        corename = f'CORE{core_idx}'
        results = []
        tiu_future = executor.submit(process_tiu, tiu_instances, core_idx, corename)
        tiu_result = tiu_future.result()
        if tiu_result:
            results.extend(tiu_result)

        gdma_future = executor.submit(process_gdma, gdma_instances, core_idx, corename)
        gdma_result = gdma_future.result()
        if gdma_result:
            results.extend(gdma_result)

        sdma_future = executor.submit(process_sdma, sdma_instances, core_idx, corename)
        sdma_result = sdma_future.result()
        if sdma_result:
            results.extend(sdma_result)

        cdma_future = executor.submit(process_cdma, cdma_instances, core_idx, corename)
        cdma_result = cdma_future.result()
        if cdma_result:
            results.extend(cdma_result)
    return corename, results

def generate_power(dirpath):
    # dirpath = os.path.abspath(dirpath)
    ip_list = []
    tiuProcessor = TIU(dirpath)
    tiu_instances = tiuProcessor.process_file()
    if len(tiu_instances) > 0:
        ip_list.append('tiu')
    gdmaProcessor = DMA(dirpath, "GDMA")
    gdma_instances = gdmaProcessor.process_file()
    if len(gdma_instances) > 0:
        ip_list.append('gdma')
    sdmaProcessor = DMA(dirpath, "SDMA")
    sdma_instances = sdmaProcessor.process_file()
    if len(sdma_instances) > 0:
        ip_list.append('sdma')
    cdmaProcessor = DMA(dirpath, "CDMA")
    cdma_instances = cdmaProcessor.process_file()
    if len(cdma_instances) > 0:
        ip_list.append('cdma')
    max_corenum = max(len(tiu_instances), len(gdma_instances), len(sdma_instances), len(cdma_instances))
    perf_power_dict = {}
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(read_data_and_cal_power, idx, tiu_instances, gdma_instances, sdma_instances, cdma_instances) for idx in range(max_corenum)]
        for future in futures:
            corename, core_results = future.result()
            perf_power_dict[corename] = core_results

    for key in perf_power_dict:
        perf_power_dict[key].sort(key=lambda x: (x[4], x[5]))
    return ip_list, perf_power_dict

def prepareJson(compiler_version, output_name, ip_list, pld_dict_to_js, perf_dict_to_js,tiu_header, dma_header, perf_tiu_columns,perf_dma_columns,pld_concat_timemark,perf_concat_timemark,out_path):
    js_content = ""
    js_content += f"let pattern_name = '{output_name}';\n"
    js_content += f"let category = {ip_list};\n"
    js_content += f"let compiler_commit = '{compiler_version}';\n"
    js_content += f"let tiu_header = {tiu_header};\n"
    js_content += f"let dma_header = {dma_header};\n"
    js_content += f"let perf_tiu_columns = {perf_tiu_columns};\n"
    js_content += f"let perf_dma_columns = {perf_dma_columns};\n"

    js_content += f"let base_power_keys = {list(ip_base_power.keys())};\n"
    js_content += f"let base_power_values = {list(ip_base_power.values())};\n"

    js_content += f"let concat_patterns = {list(perf_concat_timemark.keys())};\n"
    js_content += f"let perf_concat_timemark = {list(perf_concat_timemark.values())};\n"
    js_content += f"let pld_concat_timemark = {list(pld_concat_timemark.values())};\n"

    pld_energy, missing_pld, sd_pld = cal_total_energy(pld_dict_to_js)
    js_content += f"let pld_energy = {pld_energy};\n"
    js_content += f"let missing_pld_data = {list(missing_pld)};\n"
    perf_energy, missing_perf, missing_cmds = cal_total_energy(perf_dict_to_js)

    df = pd.DataFrame(missing_cmds, columns=perf_tiu_columns)
    df_unique = df.drop_duplicates(subset=['Ip', 'Function Type', 'Function Name', 'Data Type'])
    # （可选）将 DataFrame 保存为 CSV 文件
    # df_unique.to_csv(f'{output_name}_missing_cmds.csv', index=False)

    js_content += f"let perf_energy = {perf_energy};\n"
    js_content += f"let missing_perf_data = {list(missing_perf)};\n"
    if pld_dict_to_js:
        for key, value in pld_dict_to_js.items():
            js_value = json.dumps(value)
            js_content += f"window.PLD{key} = {js_value};\n"
    else:
         for key, value in perf_dict_to_js.items():
            js_content += f"window.PLD{key} = [];\n"
    for key, value in perf_dict_to_js.items():
        js_value = json.dumps(value)
        js_content += f"window.PERF{key} = {js_value};\n"
    jsname = os.path.join(out_path,f'{output_name}_power.js')
    with open(jsname, 'w') as js_file:
        js_file.write(js_content)


def prepareHtml(htmlfile, name,out_path):
    new_file = os.path.join(out_path,f'{name}_power.html')
    shutil.copy(htmlfile, new_file)
    with open(new_file, 'r', encoding='utf-8') as file:
        html_content = file.read()

    soup = BeautifulSoup(html_content, 'html.parser')
    script_tags = soup.find_all('script')

    # 更新<title>元素
    title_tag = soup.find('title')
    if title_tag:
        title_tag.string = name
    # 替换jsfile
    for tag in script_tags:
        if tag.get('src') == 'output.js':
            tag['src'] = f'{name}_power.js'
            break

    with open(new_file, 'w', encoding='utf-8') as file:
        file.write(str(soup.prettify()))
    print(f'{new_file} 文件已更新。')


def prepareOutput(dirpath,version, name, html_path,out_path):
    ip_list, perf_power_dict = generate_power(dirpath)
    prepareJson(version,name, ip_list, {},perf_power_dict,[],[],perf_tiu_columns,perf_dma_columns,{},{},out_path)
    prepareHtml(html_path,name,out_path)
