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
import sys

from utils.utils import get_realtime_from_cycle

class SummaryProcessor:
    def __init__(self, tiuProcessor, gdmaProcessor, sdmaProcessor, cdmaProcessor):
        self.tiuProcessor = tiuProcessor
        self.gdmaProcessor = gdmaProcessor
        self.sdmaProcessor = sdmaProcessor
        self.cdmaProcessor = cdmaProcessor
        self.totalTime = 0
        self.total_time_list = [] #us
        self.columns = ['CoreId','Parallelism(%)', 'totalTime(us)']
        self.data = [[],[],[]]

    def _get_totalTime(self):
        start_times = [processor.total_time_dict["start"] for processor in [self.tiuProcessor, self.gdmaProcessor, self.sdmaProcessor, self.cdmaProcessor] if processor.total_time_dict["start"]]
        end_times = [processor.total_time_dict["end"] for processor in [self.tiuProcessor, self.gdmaProcessor, self.sdmaProcessor, self.cdmaProcessor] if processor.total_time_dict["end"]]

        if not start_times or not end_times:
            self.totalTime = 0
            return

        min_start_time = min(min(start_times, key=lambda x: min(x)))
        max_end_time = max(max(end_times, key=lambda x: max(x)))
        self.totalTime = max_end_time - min_start_time

    def _process_tiu(self, tiuProcessor):
        # TIU处理器通用代码
        temp_timelist = [self.totalTime / 1000 for _ in range(tiuProcessor.actual_corenum)] #us
        tiu_cycle_to_time = [get_realtime_from_cycle(value, tiuProcessor.frequency) for value in tiuProcessor.tiu_cycle_list] #ns
        for i in range(tiuProcessor.actual_corenum):
            tiuProcessor.tiu_working_ratio_list.append('0.00%' if self.totalTime == 0 else
                                       str(Decimal((tiu_cycle_to_time[i] / self.totalTime * 100)).quantize(
                                           Decimal("0.00"))) + '%')
        tiuProcessor.tiu_working_ratio_list.append('0.00%' if sum(temp_timelist) == 0 else
            str((Decimal((sum(tiu_cycle_to_time) / 1000 / sum(temp_timelist)) * 100)).quantize(
                Decimal("0.00"))) + '%')
        temp_timelist.append(max(temp_timelist))
        if len(temp_timelist) >= len(self.total_time_list):
            self.total_time_list = temp_timelist
        tiuProcessor.tiu_cycle_list.append(max(tiuProcessor.tiu_cycle_list))
        last_row = len(self.total_time_list) - 1
        tiuProcessor.tiu_cycle_list[last_row] = str(
            (Decimal(tiuProcessor.tiu_cycle_list[last_row] / tiuProcessor.frequency)).quantize(Decimal("0.00"))) + 'us'
        tiuProcessor.uArach_rate_list.append(
            '0.00%' if sum(tiuProcessor.total_uarch_ops_list) == 0 else
            str((Decimal((sum(tiuProcessor.total_alg_ops_list) / sum(tiuProcessor.total_uarch_ops_list)) * 100)).quantize(
                Decimal("0.00"))) + '%')
        self.columns.extend(['TiuWorkingRatio', 'totalTiuCycle', 'uArchURate'])
        self.data.extend([tiuProcessor.tiu_working_ratio_list, tiuProcessor.tiu_cycle_list, tiuProcessor.uArach_rate_list])

    def _process_dma(self, dmaProcessor, add_l2=True):
        # DMA处理器通用代码
        temp_timelist = [self.totalTime / 1000 for _ in range(dmaProcessor.actual_corenum)] #us
        temp_timelist.append(max(temp_timelist))
        if len(temp_timelist) >= len(self.total_time_list):
            self.total_time_list = temp_timelist
        dmaProcessor.dma_cycle_list.append(max(dmaProcessor.dma_cycle_list))
        last_row = len(self.total_time_list) - 1
        dmaProcessor.dma_cycle_list[last_row] = str(
            (Decimal(dmaProcessor.dma_cycle_list[last_row] / dmaProcessor.frequency)).quantize(Decimal("0.00"))) + 'us'
        dmaProcessor.dma_ddr_total_datasize_list.append(sum(dmaProcessor.dma_ddr_total_datasize_list))
        dmaProcessor.dma_l2_total_datasize_list.append(sum(dmaProcessor.dma_l2_total_datasize_list))
        dmaProcessor.dma_ddr_avg_burst_length_list.append('0.00' if dmaProcessor.total_xact_cnt == 0 else
                                        str((Decimal(dmaProcessor.total_burst_length / dmaProcessor.total_xact_cnt)).quantize(
                                            Decimal("0.00"))) )
        if dmaProcessor.ddr_total_cycle == 0:
            dmaProcessor.dma_ddr_avg_bw_list.append(0)
        else:
            dmaProcessor.dma_ddr_avg_bw_list.append(
                str((Decimal((dmaProcessor.dma_ddr_total_datasize_list[-1] / dmaProcessor.ddr_total_cycle * dmaProcessor.frequency / 1000))).quantize(Decimal("0.00"))))
        if dmaProcessor.l2_total_cycle == 0:
            dmaProcessor.dma_l2_avg_bw_list.append(0)
        else:
            dmaProcessor.dma_l2_avg_bw_list.append(
                str((Decimal((dmaProcessor.dma_l2_total_datasize_list[-1] / dmaProcessor.l2_total_cycle * dmaProcessor.frequency / 1000))).quantize(Decimal("0.00"))))
        if add_l2:
            self.data.extend([dmaProcessor.dma_cycle_list, dmaProcessor.dma_ddr_avg_bw_list, dmaProcessor.dma_l2_avg_bw_list, dmaProcessor.dma_ddr_avg_burst_length_list])
        else:
            self.data.extend([dmaProcessor.dma_cycle_list, dmaProcessor.dma_ddr_avg_bw_list, dmaProcessor.dma_ddr_avg_burst_length_list])

    def make_summary(self):
        max_corenum = max(self.tiuProcessor.actual_corenum, self.gdmaProcessor.actual_corenum, self.sdmaProcessor.actual_corenum, self.cdmaProcessor.actual_corenum)
        CoreIdList = [str(i) for i in range(0, max_corenum)]
        CoreIdList.append('Overall')
        self._get_totalTime()
        if self.tiuProcessor.regList:
            self._process_tiu(self.tiuProcessor)
        else:
            self.tiuProcessor.tiu_cycle_list.append('0.00us')

        if self.gdmaProcessor.regList:
            self._process_dma(self.gdmaProcessor, True)
            self.columns.extend(['totalGdmaCycle', 'GdmaDdrAvgBandwidth(GB/s)','GdmaL2AvgBandwidth(GB/s)', 'GdmaAvgDdrBurstLength'])
        else:
            self.gdmaProcessor.dma_cycle_list.append('0.00us')

        if self.sdmaProcessor.regList:
            self._process_dma(self.sdmaProcessor, False)
            self.columns.extend(['totalSdmaCycle', 'SdmaDdrAvgBandwidth(GB/s)', 'SdmaAvgDdrBurstLength'])
        else:
            self.sdmaProcessor.dma_cycle_list.append('0.00us')

        if self.cdmaProcessor.regList:
            self._process_dma(self.cdmaProcessor, True)
            self.columns.extend(['totalCdmaCycle', 'CdmaDdrAvgBandwidth(GB/s)','CdmaL2AvgBandwidth(GB/s)', 'CdmaAvgDdrBurstLength'])

        ParallelismList = []
        # 计算每个核心的并行性
        for i in range(max_corenum):
            if self.total_time_list[i] == 0:
                ParallelismList.append('0.00%')
            else:
                tiu_time = self.tiuProcessor.tiu_cycle_list[i] / self.tiuProcessor.frequency if self.tiuProcessor.frequency != 0 else 0
                gdma_time = self.gdmaProcessor.dma_cycle_list[i] / self.gdmaProcessor.frequency if self.gdmaProcessor.frequency != 0 else 0
                sdma_time = self.sdmaProcessor.dma_cycle_list[i] / self.sdmaProcessor.frequency if self.sdmaProcessor.frequency != 0 else 0
                total_cycles = tiu_time + gdma_time + sdma_time
                if total_cycles == 0:
                    ParallelismList.append('0.00%')
                else:
                    parallelism = (Decimal(total_cycles * 100 / self.total_time_list[i])
                                .quantize(Decimal("0.00")))
                    ParallelismList.append(str(parallelism) + '%')

        # 计算所有核心的并行性
        if max_corenum > 1:
            if sum(self.total_time_list[:-1]) == 0:
                ParallelismList.append('0.00%')
            else:
                tiu_time = max(self.tiuProcessor.tiu_cycle_list[:-1])/self.tiuProcessor.frequency if self.tiuProcessor.frequency != 0 else 0
                gdma_time = max(self.gdmaProcessor.dma_cycle_list[:-1]) / self.gdmaProcessor.frequency if self.gdmaProcessor.frequency != 0 else 0
                sdma_time = max(self.sdmaProcessor.dma_cycle_list[:-1]) / self.sdmaProcessor.frequency if self.sdmaProcessor.frequency != 0 else 0
                total_cycles = tiu_time + gdma_time + sdma_time
                total_parallelism = Decimal((total_cycles / max(self.total_time_list[:-1]) * 100))
                ParallelismList.append(str(total_parallelism.quantize(Decimal("0.00"))) + '%')
        else:
            if sum(self.total_time_list[:-1]) == 0:
                ParallelismList.append('0.00%')
            else:
                tiu_time = max(self.tiuProcessor.tiu_cycle_list[:-1])/self.tiuProcessor.frequency if self.tiuProcessor.frequency != 0 else 0
                gdma_time = max(self.gdmaProcessor.dma_cycle_list[:-1]) / self.gdmaProcessor.frequency if self.gdmaProcessor.frequency != 0 else 0
                total_cycles = tiu_time + gdma_time
                total_parallelism = Decimal((total_cycles / max(self.total_time_list[:-1]) * 100))
                ParallelismList.append(str(total_parallelism.quantize(Decimal("0.00"))) + '%')
        self.data[0] = CoreIdList
        self.data[1] = ParallelismList
        self.data[2] = self.total_time_list
        summaryData = transpose(self.data).tolist()
        summaryDf = pd.DataFrame(summaryData, columns=self.columns, index=None)
        return summaryDf
