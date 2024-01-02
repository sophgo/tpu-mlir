#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/8/7 11:23
# @Author  : chongqing.zeng@sophgo.com
# @Project: PerfAI

"""
Note: The functionality to generate summary and layer information
    is temporarily not maintained due to the incomplete state of the
    global.profile file. Once the global.profile file is updated or
    completed, this feature can be revisited and implemented accordingly.
    Please refer to the global.profile file for more details on its
    current status and any updates that may be needed for the
    generation of summary and layer information.
"""

from enum import Enum

import pandas as pd
from openpyxl.reader.excel import load_workbook
from openpyxl.utils import get_column_letter

from definition.bm1684x_defs import Arch, DataType
from definition.style import SummaryStyle
from utils.utils import load_arch_lib, get_ratio_str_3f, cycle_to_us, ops_to_tops, cycle_to_fps


class GlobalInfo:
    def __init__(self):
        self.subnet_list = []
        self.arch = Arch.UNKNOWN
        self.mem_info = []
        self.archlib = None
        self.freq = None
        self.net_name = None
        self.flops = 0
        self.no_perf_data = False

    def set_arch(self, arch):
        self.arch = arch
        if self.archlib is None:
            self.archlib = load_arch_lib(arch)
        assert self.archlib is not None


class SubnetInfo:
    def __init__(self):
        self.subnet_id = -1
        self.layer_list = []
        self.command_info = None
        self.gdma_nodes = []
        self.bd_nodes = []
        self.sim_info = None


class TensorInfo:
    def __init__(self):
        self.tensor_id = -1
        self.shape = None
        self.dtype = DataType.UNKNOWN
        self.is_const = False
        self.gaddr = -1
        self.gsize = 0
        self.loffset = -1
        self.nslice = 0
        self.hslice = 0
        self.l2addr = 0
        self.in_layer = None
        self.out_layers = []


class StaticRunNode:
    __run_id = -1

    def __init__(self):
        self.__class__.__run_id += 1
        self.run_id = self.__class__.__run_id
        self.type = None
        self.bd_id = -1
        self.gdma_id = -1
        self.gdma_dir = None
        self.gdma_func = None
        self.bd_func = None
        self.layer = None
        self.command = None
        self.sim_info = None
        self.pmu_info = None


class SubnetType(Enum):
    UNKNOWN = -1
    TPU = 0
    CPU = 1
    MERGE = 2
    SWITCH = 3


class SummaryInfo:
    def __init__(self):
        self.layer_id = -1
        self.layer_type = None
        self.layer_name = ""
        self.is_local = False
        self.in_tensors = []
        self.out_tensors = []
        self.group_id = -1
        self.total_size = 0
        self.feature_size = 0
        self.weight_size = 0
        self.gdma_op = None
        self.gdma_tensor = None
        self.begin_usec = None
        self.end_usec = None
        self.gdma_nodes = []
        self.bd_nodes = []


class Summary:
    def __init__(self, writer):
        self.layer_summary_map = dict()
        self.layer_summary_header = ["Function", "Algorithm Tops", "Algorithm Tops Ratio", "Weight Size(MB)",
                                     "uArch Tops",
                                     "uArch URate", "uArch Tops Ratio", "ASIC Cycles", "ASIC Time(us)",
                                     "Time Ratio", "ASIC FPS"]
        self.layer_summary_row_map = {
            "conv": "Conv",
            "conv2d": "Conv",
            "pool": "Pool",
            "pool2d": "Pool",
            "fc": "Matmul",
            "matmul": "Matmul",
            "batch_matmul": "Matmul",
            "softmax": "Softmax",
            "batchnorm": "BatchNorm",
            "layernorm": "LayerNorm",
            "layer_norm": "LayerNorm",
            "group_norm": "LayerNorm",
            "groupnorm": "LayerNorm",
            "eltwise": "Eltwise",
            "eltwise_binary": "Eltwise",
            "mul": "Eltwise",
            "add": "Eltwise",
            "broadcast_binary": "Eltwise",
        }
        self.layer_summary_rows = [
            "Conv",
            "Pool",
            "Matmul",
            "Softmax",
            "BatchNorm",
            "LayerNorm",
            "Eltwise",
            "Others",
        ]
        self.sheet_name = 'Summary'
        self.data_rows = []
        self.writer = writer

    def load(self, layer_infos):
        total_weight_size = 0
        total_alg_ops = 0
        total_arch_ops = 0
        total_sim_cycles = 0
        for k in self.layer_summary_rows:
            self.layer_summary_map[k] = [k, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for layer in layer_infos:
            layer_name = layer.layer_name
            row_name = self.layer_summary_row_map.get(layer_name.lower(), "Others")
            layer_alg_ops = layer.alg_ops
            layer_arch_ops = layer.uarch_ops
            weight_size = layer.weight_size
            layer_cycles = layer.asic_cycle
            total_weight_size += weight_size
            total_alg_ops += layer_alg_ops
            total_arch_ops += layer_arch_ops
            total_asic_cycles += layer_cycles
            item = [row_name, layer_alg_ops, 0, weight_size, layer_arch_ops, 0, 0, layer_cycles,
                    0, 0, layer_cycles]
            for i in range(1, len(item)):
                self.layer_summary_map[row_name][i] += item[i]
        for row in self.layer_summary_rows:
            self.data_rows.append(self.layer_summary_map.get(row))
        for summary in self.data_rows:
            summary[2] = get_ratio_str_3f(summary[1], total_alg_ops)
            summary[6] = get_ratio_str_3f(summary[4], total_arch_ops)
            summary[5] = get_ratio_str_3f(summary[1], summary[4])
            summary[8] = cycle_to_us(summary[7], 1000)
            summary[9] = get_ratio_str_3f(summary[7], total_asic_cycles)
            summary[1] = ops_to_tops(summary[1])
            summary[4] = ops_to_tops(summary[4])
            summary[10] = '-'
        total_arch_urate = get_ratio_str_3f(total_alg_ops, total_arch_ops)
        self.data_rows.append(
            ["Overall", ops_to_tops(total_alg_ops), "100%", total_weight_size, ops_to_tops(total_arch_ops),
             total_arch_urate, "100%",
             total_asic_cycles, cycle_to_us(total_asic_cycles, 1000), "100%", cycle_to_fps(total_asic_cycles)])

    def write(self, chip_arch):
        network = chip_arch['network']
        platform = chip_arch['Chip Arch']
        if platform.lower() == 'sg2260':
            int8_ops = 256
            fp32_ops = '--'
        elif platform.lower() == 'bm1684x':
            int8_ops = 32
            fp32_ops = 2
        ddr_bw = float(chip_arch['DDR Max BW(GB/s)']) * int(chip_arch['NPU Num'])
        tpu_freq = float(chip_arch['Frequency(MHz)']) / 1000
        dma_freq = float(chip_arch['DDR Frequency']) / 1000
        condition_dict = {
            'Network': [network],
            'platform': [platform],
            'TPU INT8 TOPS': [int8_ops],
            'TPU FP16 TOPS': [int8_ops / 2],
            'TPU FP32 TOPS': [fp32_ops],
            'DDR BW(GB/s)': [ddr_bw],
            'TPU Frequency(GHz)': [tpu_freq],
            'DMA Frequency(GHz)': [dma_freq]
        }
        npu_num = int(chip_arch['NPU Num'])
        Cube_IC_Align = int(chip_arch['Cube IC Align(8bits)'])
        Conv_OHOW_Align = int(chip_arch['Cube OHOW Align'])
        Vector_OHOW_Align = int(chip_arch['Vector OHOW Align(8bits)'])
        Conv_ops = int(npu_num * Cube_IC_Align * Conv_OHOW_Align * tpu_freq * 2 / 1000)
        Vector_ops = int(npu_num * Vector_OHOW_Align * tpu_freq / 1000)
        pooling_ops = Conv_ops / 4
        detail_spec = {
            'NPU NUM (lane)': [npu_num],
            'IC Alignment (8bits)': [Cube_IC_Align],
            'Conv OHOW Align': [Conv_OHOW_Align],
            'Vector OHOW Align(8bits)': [Vector_OHOW_Align],
            'Conv Ops/s': [Conv_ops],
            'Vect Ops/s': [Vector_ops],
            'Pool OPs/s': [pooling_ops]
        }

        pd.DataFrame(condition_dict).to_excel(self.writer, index=False, sheet_name=self.sheet_name, engine='xlsxwriter', startrow=0, startcol=1, float_format='%g')
        pd.DataFrame(detail_spec).to_excel(self.writer, index=False, sheet_name=self.sheet_name, engine='xlsxwriter', startrow=3, startcol=1, float_format='%g')

        if len(self.data_rows) > 0:
            df = pd.DataFrame(self.data_rows)
            df.columns = self.layer_summary_header
            df.to_excel(self.writer, index=False, sheet_name=self.sheet_name, engine='xlsxwriter', startrow=8, startcol=1, float_format='%g')

    @classmethod
    def set_style(cls, file_path):
        """
        Set style for Excel, and highlight some fields we care about.
        :param frozen: freeze some statistics above the sheet
        :param file_path: Excel output path
        :param core_id: the id of current core
        :return: None
        """
        wb = load_workbook(file_path)
        summary_style = SummaryStyle()
        sheet_name = 'Summary'
        df = pd.read_excel(file_path, sheet_name)
        ws = wb[sheet_name]
        ws.cell(1, 1).value = 'Condition'
        ws.cell(4, 1).value = 'Detail Spec'
        ws.cell(8, 1).value = 'Performance'
        ws.cell(9, 1).value = 'Summary'
        for h, w in zip([1, 4, 8, 9], [1, 1, 1, 1]):
            ws.cell(h, w).fill = summary_style.title_pattern
            ws.cell(h, w).font = summary_style.title_font
        summary_start_cols = 2
        summary_end_cols = summary_start_cols + 8
        for w in range(summary_start_cols, summary_end_cols):
            for h in [1, 4]:
                if h == 1:
                    ws.cell(h, w).fill = summary_style.title_header_pattern
                    ws.cell(h, w).font = summary_style.title_header_font
                    ws.cell(h, w).border = SummaryStyle.border
                    ws.cell(h+1, w).fill = summary_style.title_content_pattern
                    ws.cell(h+1, w).font = summary_style.title_header_font
                    ws.cell(h+1, w).alignment = summary_style.center_align
                    ws.cell(h+1, w).border = SummaryStyle.border
                elif h == 4 and w < summary_end_cols - 1:
                    # summary title style
                    ws.cell(h, w).fill = summary_style.title_header_pattern
                    ws.cell(h, w).font = summary_style.title_header_font
                    ws.cell(h, w).border = SummaryStyle.border
                    # summary content style
                    ws.cell(h+1, w).fill = summary_style.title_content_pattern
                    ws.cell(h+1, w).font = summary_style.title_header_font
                    ws.cell(h+1, w).alignment = summary_style.center_align
                    ws.cell(h+1, w).border = SummaryStyle.border

        content_start_cols = 2
        content_end_cols = content_start_cols + 11
        content_header_row = 8
        for w in range(content_start_cols, content_end_cols):
            ws.cell(content_header_row+1, w).fill = summary_style.content_header2_pattern
            ws.cell(content_header_row+1, w).font = summary_style.content_header_font
            for h in range(content_header_row+2, content_header_row+11):
                ws.cell(h, w).font = summary_style.title_header_font
                ws.cell(h, w).alignment = summary_style.center_align
                ws.cell(h, w).border = SummaryStyle.border
            ws.cell(content_header_row+10, w).fill = summary_style.content_pattern
        ws.cell(8, 3).value = 'Algorithm'
        ws.cell(8, 6).value = 'uArch'
        ws.cell(8, 9).value = 'Simulation'
        for w in range(3, 13):
            ws.cell(8, w).fill = summary_style.content_header1_pattern
        for h, w in zip([8, 8, 8], [3, 6, 9]):
            ws.cell(h, w).font = summary_style.title_font
            ws.cell(h, w).border = SummaryStyle.title_border
            ws.cell(h, w).alignment = summary_style.center_align
        ws.merge_cells('C8:E8')
        ws.merge_cells('F8:H8')
        ws.merge_cells('I8:L8')
        for col in df.columns:
            # auto set columns width
            index = list(df.columns).index(col)
            collen = 0
            letter = get_column_letter(index + 1)
            for h in range(1, len(df) + 8):
                if h > 30:
                    break
                if ws.cell(h, index + 1).value is not None:
                    if len(str(ws.cell(h, index + 1).value)) > 35:
                        continue
                    collen = max(collen, len(str(ws.cell(h, index + 1).value)))
            ws.column_dimensions[letter].width = collen * 1.05
        ws.sheet_properties.tabColor = '31869B'
        wb.save(file_path)
