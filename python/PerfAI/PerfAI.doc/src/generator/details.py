#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/8/7 11:26
# @Author  : chongqing.zeng@sophgo.com
# @Project: PerfAI
import os
import pandas as pd
from tqdm import tqdm

from include.dma import Gdma, Sdma, Cdma
from include.instr_world import InstrWorld
from include.asic_summary import AsicSummary
from include.tiu import Tiu
from utils.utils import get_instr_cols, get_instr_reg_list, get_active_cores
from include.summary import GlobalInfo


def palace_holder(writer, splited, g_info):
    if g_info is not None:
        sheet_name = 'Summary'
        df = pd.DataFrame()
        df.to_excel(writer, index=False, sheet_name=sheet_name, engine='xlsxwriter')
        sheet_name = 'Layer by Layer Info'
        df.to_excel(writer, index=False, sheet_name=sheet_name, engine='xlsxwriter')
    sheet_name = 'Asic Summary'
    df = pd.DataFrame()
    df.to_excel(writer, index=False, sheet_name=sheet_name, engine='xlsxwriter')
    if not splited:
        sheet_name = 'Instr World'
        df.to_excel(writer, index=False, sheet_name=sheet_name, engine='xlsxwriter')


def get_engine_layer(g_info):
    tiu_layer_map = dict()
    gdma_layer_map = dict()
    if isinstance(g_info, GlobalInfo):
        for tiu_node in g_info.subnet_list[0].bd_nodes:
            k = tiu_node.bd_id
            if k in tiu_layer_map.keys():
                print('ERROR! Tiu id is not unique.')
                assert 0
            else:
                tiu_layer_map[k] = [tiu_node.layer.layer_id, tiu_node.layer.layer_type]
        for gdma_node in g_info.subnet_list[0].gdma_nodes:
            k = gdma_node.gdma_id - 1
            if k in gdma_layer_map.keys():
                print('ERROR! Gdma id is not unique.')
                assert 0
            else:
                gdma_layer_map[k] = [gdma_node.layer.layer_id, gdma_node.layer.layer_type]
    return tiu_layer_map, gdma_layer_map


def generate_details(input_fold, out_file, g_info, writer, core_num=8, split_instr_world=False):
    tiu_reg_file = 'tiuRegInfo'
    dma_reg_file = 'tdmaRegInfo'
    cdma_reg_file = 'cdmaRegInfo'
    tiu_reg_file = input_fold + tiu_reg_file
    dma_reg_file = input_fold + dma_reg_file
    cdma_reg_file = input_fold + cdma_reg_file
    act_core_num = 0
    act_core_num = max(act_core_num, get_active_cores(tiu_reg_file, core_num))
    act_core_num = max(act_core_num, get_active_cores(dma_reg_file, core_num))
    act_core_num = max(act_core_num, get_active_cores(cdma_reg_file, core_num))
    if not act_core_num:
        print('Error, No engine reg info file input! Please check your input.')
        exit(-1)
    print('Start generate data for ' + out_file)
    palace_holder(writer, split_instr_world, g_info)
    instr_cols, instr_reg_list, reg_list = [], [], []
    tiu_instances, gdma_instances, sdma_instances, cdma_instances = [], [], [], []
    tiu_instance_map, gdma_instance_map = dict(), dict()
    tiu_layer_map, gdma_layer_map = get_engine_layer(g_info)
    chip_arch_act = None
    for core_id in tqdm(range(act_core_num)):
        # tiu
        cur_tiu_reg_file = tiu_reg_file + '_' + str(core_id) + '.txt'
        tiu_instance = Tiu(core_id, writer)
        chip_arch = tiu_instance.load(cur_tiu_reg_file, tiu_layer_map)
        chip_arch_act = chip_arch_act if chip_arch_act else chip_arch
        tiu_instance.add_kpi_field()
        tiu_instance.write()
        if str(core_id) == '0':
            tiu_instance_map = tiu_instance.pop_data()
        # gdma
        cur_dma_reg_file = dma_reg_file + '_' + str(core_id) + '.txt'
        gdma_instance = Gdma(core_id, writer, 'GDMA')
        tmp_chip_arch = gdma_instance.load(cur_dma_reg_file, gdma_layer_map)
        chip_arch_act = chip_arch_act if chip_arch_act else tmp_chip_arch
        gdma_instance.add_kpi_field()
        gdma_instance.write()
        if str(core_id) == '0':
            gdma_instance_map = gdma_instance.pop_data()
        # sdma
        cur_dma_reg_file = dma_reg_file + '_' + str(core_id) + '.txt'
        sdma_instance = Sdma(core_id, writer, 'SDMA')
        sdma_instance.load(cur_dma_reg_file, gdma_layer_map)
        sdma_instance.add_kpi_field()
        sdma_instance.write()
        # cdma
        cdma_instance = Cdma(core_id, writer, 'CDMA')
        if act_core_num:
            cur_cdma_reg_file = cdma_reg_file + '_' + str(core_id) + '.txt'
            if os.path.exists(cur_cdma_reg_file) and os.path.getsize(cur_cdma_reg_file):
                tmp_chip_arch = cdma_instance.load(cur_cdma_reg_file)
                chip_arch_act = chip_arch_act if chip_arch_act else tmp_chip_arch
                cdma_instance.add_kpi_field()
                cdma_instance.write()
            reg_list += tiu_instance.reg_list + gdma_instance.reg_list + sdma_instance.reg_list + cdma_instance.reg_list
        else:
            reg_list += tiu_instance.reg_list + gdma_instance.reg_list + sdma_instance.reg_list
        instr_cols = get_instr_cols(tiu_instance.columns, gdma_instance.columns)
        tiu_instances.append(tiu_instance)
        gdma_instances.append(gdma_instance)
        sdma_instances.append(sdma_instance)
        cdma_instances.append(cdma_instance)

    if act_core_num:
        # instr world
        instr_reg_list = get_instr_reg_list(reg_list, instr_cols)
        instr_instance = InstrWorld(instr_reg_list, instr_cols, writer, split_instr_world)
        instr_instance.write(out_file)
        # summary
        summary_instance = AsicSummary(writer, tiu_instances, gdma_instances, sdma_instances, cdma_instances, act_core_num)
        summary_instance.load(chip_arch_act)
        summary_instance.write()
    return tiu_instance_map, gdma_instance_map, chip_arch_act