# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
import pandas as pd
import os
from decimal import Decimal
import sys

from utils.utils import data_type_dict, data_size_dict
from utils.utils import *

class TIU:
    def __init__(self, dirpath):
        self.dirpath = dirpath
        self.chipArgs = dict()
        self.linecount = 0
        self.frequency = 0
        self.actual_corenum = 0
        self.regList=[]
        self.tiu_working_ratio_list = []
        self.total_time_dict = {"start":[], "end":[]}
        self.tiu_cycle_list = []
        self.total_alg_cycle_list = []
        self.total_alg_ops_list = []
        self.uArach_rate_list = []
        self.total_uarch_ops_list = []
        self.columns = ['Engine Id', 'Core Id', 'Cmd Id', 'Layer Id', 'Layer Name', 'Subnet Id', 'Subnet Type', 'File Line',
                        'Function Type', 'Function Name',
                        'Alg Cycle', 'Alg Ops','Asic Cycle', 'Start Cycle', 'End Cycle',
                        'uArch Ops', 'uArch Rate', 'Bank Conflict Ratio',
                        'Initial Cycle Ratio', 'Data Type', 'Sim Power(W)','des_cmd_id_dep',
                        'des_res0_n', 'des_res0_c', 'des_res0_h', 'des_res0_w',
                        'des_res0_n_str', 'des_res0_c_str', 'des_res0_h_str', 'des_res0_w_str',
                        'des_opd0_n', 'des_opd0_c', 'des_opd0_h', 'des_opd0_w',
                        'des_opd0_n_str', 'des_opd0_c_str', 'des_opd0_h_str', 'des_opd0_w_str',
                        'des_opd1_c', 'des_opd1_h', 'des_opd1_w', 'des_opd1_n_str', 'des_opd1_c_str', 'des_opd1_h_str',
                        'des_opd1_w_str',
                        'des_opd2_n_str', 'des_res0_addr', 'des_res1_addr', 'des_opd0_addr', 'des_opd1_addr',
                        'des_opd2_addr',
                        'des_opd3_addr', 'des_tsk_typ', 'des_tsk_eu_typ', 'des_cmd_short',
                        'des_opt_res0_prec', 'des_opt_opd0_prec', 'des_opt_opd1_prec', 'des_opt_opd2_prec',
                        'des_short_opd0_str',
                        'des_opt_opd0_const', 'des_opt_opd1_const', 'des_opt_opd2_const', 'des_opt_opd3_const',
                        'des_opt_opd4_const', 'des_opt_opd5_const',
                        'des_opt_res_add', 'des_opt_res0_sign', 'des_opt_opd0_sign', 'des_opt_opd1_sign',
                        'des_opt_opd2_sign',
                        'des_opd0_rt_pad', 'des_opd1_x_ins0', 'des_opd0_up_pad', 'des_opd0_lf_pad', 'des_opt_left_tran',
                        'des_pad_mode', 'des_opd0_y_ins0', 'des_opd1_y_ins0',
                        'des_short_res0_str', 'des_short_opd1_str', 'des_sym_range', 'des_opt_rq', 'des_op_code',
                        'des_opt_kernel_rotate', 'des_res_op_x_str', 'des_res_op_y_str', 'des_opd0_x_ins0',
                        'des_tsk_opd_num', 'des_opd0_dn_pad', 'des_intr_en', 'des_opt_relu', 'des_pwr_step', 'Msg Id', 'Sd\Wt Count']

    def process_file(self, layer_map):
        # file_name = f"{self.dirpath}/tiuRegInfo_0.txt"
        file_name = os.path.join(self.dirpath,'tiuRegInfo_0.txt')
        if os.path.exists(file_name) and os.path.getsize(file_name) > 0:
            with open(file_name, "r") as f:
                lines = f.readlines()
                for line in lines:
                    self.linecount += 1
                    if "\t" in line:
                        fields = line.split(': ')
                        attr = fields[0][1:]
                        val = fields[1][:-1]
                        self.chipArgs[attr] = val
                    if '__TIU_REG_INFO__' in line:
                        break
            self.frequency = int(self.chipArgs['TIU Frequency(MHz)'])
            coreNum = int(self.chipArgs['Core Num'])
            for coreId in range(int(coreNum)):
                curTiuRegFile = f"{self.dirpath}/tiuRegInfo" + '_' + str(coreId) + '.txt'
                if os.path.exists(curTiuRegFile) and os.path.getsize(curTiuRegFile) != 0:
                    self.actual_corenum += 1
            # self.actual_corelist = [str[i] for i in range(0, actualCoreNum)]
            tiuDf_list = [] #list of tiu dataframes
            for coreId in range(self.actual_corenum):
                tiuDf_list.append(self.process_data(coreId, layer_map))
            return tiuDf_list
        else:
            self.tiu_cycle_list.append(0)
            return []

    def process_data(self, coreId, layer_map):
        TiuCycle = 0
        algTotalCycle = 0
        algTotalOps = 0
        uArchTotalOps = 0
        lane_num = int(self.chipArgs['NPU Num'])
        new_regList=[]
        totalInstRegList = []
        startTime = sys.maxsize
        endTime = 0

        curTiuRegFile = f"{self.dirpath}/tiuRegInfo" + '_' + str(coreId) + '.txt'
        with open(curTiuRegFile) as f:
            rows = f.readlines()[self.linecount:]
            fieldSet = set()
            for row in rows:
                if "\t" in row:
                    attr = row.split(': ')[0][1:]
                    fieldSet.add(attr)
            tiuCols = self.columns
            fieldList = list(fieldSet) if len(fieldSet) >= len(tiuCols) else tiuCols
            tiuRegDict = dict.fromkeys(fieldList, '')
            idx = 0
            for row in rows:
                if "__TIU_REG_INFO__" in row:
                    if idx != 0:
                        k = int(tiuRegDict['Cmd Id'])
                        layer_info = ['-', '-','-','-','-']
                        if (k,coreId) in layer_map.keys():
                            layer_info = layer_map[(k,coreId)]
                        tiuRegDict['Layer Id'] = int(layer_info[0]) if layer_info[0] != '-' else '-'
                        tiuRegDict['Layer Name'] = layer_info[1]
                        tiuRegDict['Subnet Id'] = int(layer_info[2]) if layer_info[0] != '-' else '-'
                        tiuRegDict['Subnet Type'] = layer_info[3]
                        tiuRegDict['File Line'] = int(layer_info[4]) if layer_info[0] != '-' else '-'
                        new_regList.append(tiuRegDict)
                        tiuRegDict = dict.fromkeys(fieldList, '')

                elif "\t" not in row:
                    tiuRegDict['Function Type'] = row[:-2]
                else:
                    fields = row.split(': ')
                    attr = fields[0][1:]
                    val = fields[1][:-1]
                    tiuRegDict[attr] = val
                idx += 1
            k = int(tiuRegDict['Cmd Id'])
            layer_info = ['-', '-','-','-','-']
            if (k,coreId) in layer_map.keys():
                layer_info = layer_map[(k,coreId)]
            tiuRegDict['Layer Id'] = int(layer_info[0]) if layer_info[0] != '-' else '-'
            tiuRegDict['Layer Name'] = layer_info[1]
            tiuRegDict['Subnet Id'] = int(layer_info[2]) if layer_info[0] != '-' else '-'
            tiuRegDict['Subnet Type'] = layer_info[3]
            new_regList.append(tiuRegDict)

        for i in range(len(new_regList)):
            regDict = new_regList[i] #tiuRegDict
            if regDict['Asic Cycle'].isnumeric() and not (regDict['des_tsk_typ'] == '15' and regDict['des_tsk_eu_typ'] == '9'):
                TiuCycle += int(regDict['Asic Cycle'])
            if regDict['Alg Cycle'].isnumeric():
                algTotalCycle += int(regDict['Alg Cycle'])
            if regDict['Alg Ops'].isnumeric():
                algTotalOps += int(regDict['Alg Ops'])
            if regDict['uArch Ops'].isnumeric():
                uArchTotalOps += int(regDict['uArch Ops'])
            if regDict['des_opt_opd0_prec'].isnumeric():
                regDict['Data Type'] = data_type_dict[regDict['des_opt_opd0_prec']] + \
                                        ' -> ' + data_type_dict[regDict['des_opt_res0_prec']]
            else:
                regDict['Data Type'] = data_type_dict[regDict['des_opt_res0_prec']] + \
                                        ' -> ' + data_type_dict[regDict['des_opt_res0_prec']]
            regDict['Start Cycle'] = int(regDict['Start Cycle'])
            regDict['End Cycle'] = int(regDict['End Cycle'])
            regDict['Asic Cycle'] = int(regDict['Asic Cycle'])
            # regDict['Start Time'] = get_realtime_from_cycle(regDict['Start Cycle'],self.frequency) #ns
            # regDict['End Time'] = get_realtime_from_cycle(regDict['End Cycle'],self.frequency) #ns

            startTime = min(startTime, get_time_by_cycle(regDict['Start Cycle'], self.frequency))
            endTime = max(endTime, get_time_by_cycle(regDict['End Cycle'], self.frequency))
            totalInstRegList.append(regDict)
        self.total_time_dict["start"].append(startTime)
        self.total_time_dict["end"].append(endTime)
        self.tiu_cycle_list.append(TiuCycle)
        # self.tiu_working_ratio_list.append('0.00%' if simTotalCycle == 0 else
        #                                str(Decimal((TiuCycle / simTotalCycle * 100)).quantize(
        #                                    Decimal("0.00"))) + '%')
        self.total_alg_cycle_list.append(algTotalCycle)
        self.total_alg_ops_list.append(algTotalOps)
        self.total_uarch_ops_list.append(uArchTotalOps)
        self.uArach_rate_list.append('0.00%' if uArchTotalOps == 0 else str(
                        Decimal((algTotalOps / uArchTotalOps * 100)).quantize(Decimal("0.00"))) + '%')
        self.regList.append(totalInstRegList)

        tiuDf = pd.DataFrame(totalInstRegList)
        tiuCols = self.columns
        newTiuCols = []
        for tiuCol in tiuCols:
            if tiuCol in tiuDf.columns.values.tolist():
                newTiuCols.append(tiuCol)
        tiuCols = newTiuCols
        tiuDf = tiuDf[tiuCols]
        for col in tiuCols:
            if 'addr' in col or 'mask' in col:
                tiuDf[col] = intToHex(tiuDf[col].values)
        #process tiu compute tensor size
        groups = [
            ['des_res0_n', 'des_res0_c', 'des_res0_h', 'des_res0_w', 'des_opt_res0_prec', 'des_res0_size'],
            ['des_opd0_n', 'des_opd0_c', 'des_opd0_h', 'des_opd0_w', 'des_opt_opd0_prec', 'des_opd0_size'],
            ['des_opd1_n', 'des_opd1_c', 'des_opd1_h', 'des_opd1_w', 'des_opt_opd1_prec', 'des_opd1_size']
        ]

        for _, _, _, _, _, new_col in groups:
            tiuDf[new_col] = None
        for idx, row in tiuDf.iterrows():
            for n_col, c_col, h_col, w_col, prec_col, new_col in groups:
                if prec_col not in tiuDf.columns or row[prec_col] not in data_size_dict:
                    continue
                byte_size = data_size_dict[row[prec_col]]
                factors = []
                for col in [n_col, c_col, h_col, w_col]:
                    if col not in tiuDf.columns or row[col] in ['','0']:
                        continue
                    val = pd.to_numeric(row[col])
                    # if col == c_col:
                    #     # print("col, c_col, val:", col, c_col, val)
                    #     if val <= 64:
                    #         val /= val
                    #     else:
                    #         val /= lane_num
                    factors.append(val)
                if not factors:
                    continue
                result = byte_size
                for factor in factors:
                    result *= factor
                tiuDf.at[idx, new_col] = result
        return tiuDf
