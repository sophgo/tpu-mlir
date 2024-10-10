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
import os
import sys

from utils.utils import *

class DMA(): #GDMA/SDMA/CDMA
    def __init__(self, dirpath, dmaType):
        self.dirpath = dirpath
        self.dmaType = dmaType
        self.chipArgs = dict()
        self.linecount = 0
        self.actual_corenum = 0
        self.regList = []
        self.total_time_dict = {"start":[], "end":[]}
        self.dma_cycle_list = []
        self.dma_ddr_total_datasize_list = []
        self.dma_l2_total_datasize_list = []
        self.dma_ddr_avg_bw_list = []
        self.dma_l2_avg_bw_list = []
        self.dma_ddr_avg_burst_length_list = []
        self.ddr_total_cycle = 0
        self.l2_total_cycle = 0
        self.total_burst_length = 0
        self.total_xact_cnt = 0
        self.frequency = 0
        self.columns = ['Engine Id', 'Core Id', 'Cmd Id', 'Layer Id', 'Layer Name', 'Subnet Id', 'Subnet Type', 'File Line',
                        'Function Type', 'Function Name', 'DMA data size(B)', 'Start Cycle', 'End Cycle',
                        'Asic Cycle', 'Stall Cycle', 'DDR Bandwidth(GB/s)','L2M Bandwidth(GB/s)', 'Direction', 'AvgBurstLength', 'Data Type', 'Non32ByteRatio',
                        'MaskWriteRatio', 'cmd_id_dep', 'cmd_special_function', 'src_start_addr', 'dst_start_addr',
                        'index_shape', 'src_shape', 'dst_shape',
                        'src_nsize', 'src_csize', 'src_hsize', 'src_wsize',
                        'dst_nsize', 'dst_csize', 'dst_hsize', 'dst_wsize',
                        'src_nstride', 'src_cstride', 'src_wstride', 'src_hstride',
                        'dst_nstride', 'dst_cstride', 'dst_hstride', 'dst_wstride',
                        'nchw_copy', 'stride_enable', 'src_data_format', 'cmd_type',
                        'index_csize', 'index_hsize', 'index_cstride', 'index_hstride',
                        'mask_start_addr_h8', 'mask_start_addr_l32', 'mask_data_format', 'localmem_mask_h32',
                        'localmem_mask_l32',
                        'fill_constant_en', 'constant_value', 'index', 'cmd_short', 'intr_en', 'Msg Id', 'Sd\Wt Count']
    def dma_engine_type(self):
        if self.dmaType == 'CDMA':
            return '4'
        elif self.dmaType == 'GDMA':
            self.dmaType = 'TDMA'
            return '1'
        elif self.dmaType == 'SDMA':
            self.dmaType = 'TDMA'
            return '3'

    def process_file(self, layer_map):
        engineId = self.dma_engine_type()
        # file_name = f"{self.dirpath}/{self.dmaType.lower()}RegInfo_0.txt"
        file_name = os.path.join(self.dirpath,f'{self.dmaType.lower()}RegInfo_0.txt')
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
                    if f'__{self.dmaType}_REG_INFO__' in line:
                        break
            self.frequency = int(self.chipArgs['DMA Frequency(MHz)'])
            coreNum = int(self.chipArgs['Core Num'])
            for coreId in range(int(coreNum)):
                curDmaRegFile = f"{self.dirpath}/{self.dmaType.lower()}RegInfo" + '_' + str(coreId) + '.txt'
                if os.path.exists(curDmaRegFile) and os.path.getsize(curDmaRegFile) != 0:
                    self.actual_corenum += 1
            dmaDf_list = [] #list of tiu dataframes
            for coreId in range(self.actual_corenum):
                dmaDf_list.append(self.process_data(coreId,engineId,layer_map))
            return dmaDf_list
        else:
            self.dma_cycle_list.append(0)
            return []

    def process_data(self, coreId, engineId, layer_map):
        curDmaRegFile = f"{self.dirpath}/{self.dmaType.lower()}RegInfo" + '_' + str(coreId) + '.txt'
        new_reglist = []
        with open(curDmaRegFile) as f:
            rows = f.readlines()[self.linecount:]
            fieldSet = set()
            for row in rows:
                if "\t" in row:
                    attr = row.split(': ')[0][1:]
                    fieldSet.add(attr)
            fieldList = list(fieldSet) if len(fieldSet) >= len(self.columns) else self.columns
            dmaRegDict = dict.fromkeys(fieldList, '')
            idx = 0
            for row in rows:
                if f"__{self.dmaType}_REG_INFO__" in row: #TDMA/CDMA
                    if idx != 0:
                        k = int(dmaRegDict['Cmd Id'])
                        layer_info = ['-', '-','-','-']
                        if (k,coreId) in layer_map.keys():
                            layer_info = layer_map[(k,coreId)]
                        if all(map(lambda x: isinstance(x, float) and math.isnan(x), layer_info)):
                            layer_info = ['-', '-', '-', '-', '-']
                        dmaRegDict['Layer Id'] = int(layer_info[0]) if layer_info[0] != '-' else '-'
                        dmaRegDict['Layer Name'] = layer_info[1]
                        dmaRegDict['Subnet Id'] = int(layer_info[2]) if layer_info[0] != '-' else '-'
                        dmaRegDict['Subnet Type'] = layer_info[3]
                        dmaRegDict['File Line'] = int(layer_info[4]) if layer_info[0] != '-' else '-'
                        new_reglist.append(dmaRegDict)
                    dmaRegDict = dict.fromkeys(fieldList, '')
                else:
                    fields = row.split(': ')
                    attr = fields[0][1:]
                    val = fields[1][:-1]
                    dmaRegDict[attr] = val
                idx += 1
            k = int(dmaRegDict['Cmd Id'])
            layer_info = ['-', '-', '-', '-', '-']
            if (k,coreId) in layer_map.keys():
                layer_info = layer_map[(k,coreId)]
            dmaRegDict['Layer Id'] = int(layer_info[0]) if layer_info[0] != '-' else '-'
            dmaRegDict['Layer Name'] = layer_info[1]
            dmaRegDict['Subnet Id'] = int(layer_info[2]) if layer_info[0] != '-' else '-'
            dmaRegDict['Subnet Type'] = layer_info[3]
            new_reglist.append(dmaRegDict)

        temp = []
        for reg_dict in new_reglist:
            if reg_dict['Engine Id'] == engineId:
                temp.append(reg_dict)
        new_reglist = temp
        startTime = sys.maxsize
        endTime = 0
        DmaCycle = 0
        dmaDdrTotalDataSize = 0
        dmaL2TotalDataSize = 0
        dmaWaitMsgTotalTime = 0
        dmaDdrCycle = 0
        dmaL2Cycle = 0
        dmaDdrBurstLength = 0
        dmaDdrXactCnt = 0
        totalInstRegList = []
        for i in range(len(new_reglist)):
            regDict = new_reglist[i]
            if regDict['cmd_type'] == '6':
                regDict['Data Type'] = 'None'
            if int(regDict['cmd_type']) == 6: # dma_sys do not transfer data
                regDict['Direction'] = '-'
            if regDict['Asic Cycle'].isnumeric():
                DmaCycle += int(regDict['Asic Cycle'])
            if 'DDR' in regDict['Direction'] and regDict['DMA data size(B)'].isnumeric():
                dmaDdrTotalDataSize += int(regDict['DMA data size(B)'])
                dmaDdrCycle += float(regDict['Asic Cycle'])
                dmaDdrBurstLength += int(regDict['gmem_bl_sum'])
                dmaDdrXactCnt += int(regDict['gmem_xact_cnt'])
            elif 'L2' in regDict['Direction'] and regDict['DMA data size(B)'].isnumeric():
                dmaL2TotalDataSize += int(regDict['DMA data size(B)'])
                dmaL2Cycle += float(regDict['Asic Cycle'])
            if regDict['cmd_type'] == '6' and regDict['cmd_special_function'] == '4':
                dmaWaitMsgTotalTime += eval(regDict['Asic Cycle'])
            if int(regDict['gmem_xact_cnt']) > 0:
                regDict['AvgBurstLength'] = Decimal(
                    int(regDict['gmem_bl_sum']) / int(regDict['gmem_xact_cnt'])).quantize(Decimal("0.00"))
                regDict['Non32ByteRatio'] = Decimal(
                    int(regDict['gmem_n32Ba_sa_cnt']) / int(regDict['gmem_xact_cnt'])).quantize(Decimal("0.00"))
                regDict['MaskWriteRatio'] = Decimal(
                    int(regDict['gmem_msk_wr_cnt']) / int(regDict['gmem_xact_cnt'])).quantize(Decimal("0.00"))
            else:
                regDict['AvgBurstLength'] = 0
                regDict['Non32ByteRatio'] = 0
                regDict['MaskWriteRatio'] = 0
            regDict['Start Cycle'] = int(regDict['Start Cycle'])
            regDict['End Cycle'] = int(regDict['End Cycle'])
            regDict['Asic Cycle'] = int(regDict['Asic Cycle'])
            # regDict['Start Cycle'] = get_realtime_from_cycle(int(regDict['Start Cycle']),self.frequency)
            # regDict['End Cycle'] = get_realtime_from_cycle(int(regDict['End Cycle']),self.frequency)
            regDict['DDR Bandwidth(GB/s)'] = round(float(regDict['DDR Bandwidth(GB/s)']), 2)
            regDict['L2M Bandwidth(GB/s)'] = round(float(regDict['L2M Bandwidth(GB/s)']), 2)
            startTime = min(startTime, get_time_by_cycle(regDict['Start Cycle'], self.frequency))
            endTime = max(startTime, get_time_by_cycle(reg_dict['End Cycle'], self.frequency))
            totalInstRegList.append(regDict)

        self.regList.append(totalInstRegList)
        self.total_time_dict["start"].append(startTime)
        self.total_time_dict["end"].append(endTime)
        self.dma_cycle_list.append(DmaCycle - dmaWaitMsgTotalTime)
        self.dma_ddr_total_datasize_list.append(dmaDdrTotalDataSize)
        self.dma_l2_total_datasize_list.append(dmaL2TotalDataSize)
        if dmaDdrCycle > 0:
            dmaDdrTotalBandWidth = str(Decimal((dmaDdrTotalDataSize / dmaDdrCycle * self.frequency / 1000)).quantize(Decimal("0.00")))
        else:
            dmaDdrTotalBandWidth = 0
        if dmaL2Cycle > 0:
            dmaL2TotalBandWidth = str(Decimal((dmaL2TotalDataSize / dmaL2Cycle * self.frequency / 1000)).quantize(Decimal("0.00")))
        else:
            dmaL2TotalBandWidth = 0
        self.ddr_total_cycle += dmaDdrCycle
        self.l2_total_cycle += dmaL2Cycle
        self.dma_ddr_avg_bw_list.append(dmaDdrTotalBandWidth)
        self.dma_l2_avg_bw_list.append(dmaL2TotalBandWidth)
        dmaDdrAvgBurstLength = 0 if dmaDdrXactCnt == 0 else Decimal((dmaDdrBurstLength / dmaDdrXactCnt)).quantize(Decimal("0.00"))
        self.dma_ddr_avg_burst_length_list.append(dmaDdrAvgBurstLength)
        self.total_burst_length += dmaDdrBurstLength
        self.total_xact_cnt += dmaDdrXactCnt

        dmaDf = pd.DataFrame(totalInstRegList)
        new_df = pd.DataFrame()
        if len(dmaDf) > 0:
            for column in self.columns:
                if column in dmaDf.columns:
                    new_df[column] = dmaDf[column]
                else:
                    new_df[column] = None
            pre_clos = dmaDf.columns
            dmaDf = new_df
            for col in self.columns:
                if ('addr' in col or 'mask' in col) and col in pre_clos:
                    dmaDf[col] = intToHex(dmaDf[col].values)
        return dmaDf
