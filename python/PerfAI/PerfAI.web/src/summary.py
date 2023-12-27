import pandas as pd
from decimal import Decimal
from numpy import transpose

class SummaryProcessor:
    def __init__(self, tiuProcessor, gdmaProcessor, sdmaProcessor, cdmaProcessor):
        self.tiuProcessor = tiuProcessor
        self.gdmaProcessor = gdmaProcessor
        self.sdmaProcessor = sdmaProcessor
        self.cdmaProcessor = cdmaProcessor
        self.simTotalCycles = []
        self.columns = ['CoreId','Parallelism(%)', 'simTotalCycle']
        self.data = [[],[],[]]

    def _process_tiu(self, tiuProcessor):
        # TIU处理器通用代码
        tiuProcessor.tiu_working_ratio_list.append(
            '0.00%' if sum(tiuProcessor.sim_total_cycle_list) == 0 else
            str((Decimal((sum(tiuProcessor.sim_tiu_cycle_list) / sum(tiuProcessor.sim_total_cycle_list)) * 100)).quantize(
                Decimal("0.00"))) + '%')
        tiuProcessor.sim_total_cycle_list.append(max(tiuProcessor.sim_total_cycle_list))
        if len(tiuProcessor.sim_total_cycle_list) >= len(self.simTotalCycles):
            self.simTotalCycles = tiuProcessor.sim_total_cycle_list
        tiuProcessor.sim_tiu_cycle_list.append(max(tiuProcessor.sim_tiu_cycle_list))
        for rowIdx in [len(self.simTotalCycles) - 1]:
            tiuProcessor.sim_tiu_cycle_list[rowIdx] = str(
                (Decimal(tiuProcessor.sim_tiu_cycle_list[rowIdx] / tiuProcessor.frequency)).quantize(Decimal("0.00"))) + 'us'
        tiuProcessor.uArach_rate_list.append(
            '0.00%' if sum(tiuProcessor.total_uarch_ops_list) == 0 else
            str((Decimal((sum(tiuProcessor.total_alg_ops_list) / sum(tiuProcessor.total_uarch_ops_list)) * 100)).quantize(
                Decimal("0.00"))) + '%')
        self.columns.extend(['TiuWorkingRatio', 'simTiuCycle', 'uArchURate'])
        self.data.extend([tiuProcessor.tiu_working_ratio_list, tiuProcessor.sim_tiu_cycle_list, tiuProcessor.uArach_rate_list])

    def _process_dma(self, dmaProcessor, add_l2=True):
        # DMA处理器通用代码
        dmaProcessor.sim_total_cycle_list.append(max(dmaProcessor.sim_total_cycle_list))
        if len(dmaProcessor.sim_total_cycle_list) >= len(self.simTotalCycles):
           self.simTotalCycles = dmaProcessor.sim_total_cycle_list
        dmaProcessor.sim_dma_cycle_list.append(max(dmaProcessor.sim_dma_cycle_list))
        for rowIdx in [len(self.simTotalCycles) - 1]:
            dmaProcessor.sim_dma_cycle_list[rowIdx] = str(
                (Decimal(dmaProcessor.sim_dma_cycle_list[rowIdx] / dmaProcessor.frequency)).quantize(Decimal("0.00"))) + 'us'
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
            self.data.extend([dmaProcessor.sim_dma_cycle_list, dmaProcessor.dma_ddr_avg_bw_list, dmaProcessor.dma_l2_avg_bw_list, dmaProcessor.dma_ddr_avg_burst_length_list])
        else:
            self.data.extend([dmaProcessor.sim_dma_cycle_list, dmaProcessor.dma_ddr_avg_bw_list, dmaProcessor.dma_ddr_avg_burst_length_list])

    def make_summary(self):
        max_corenum = max(self.tiuProcessor.actual_corenum, self.gdmaProcessor.actual_corenum, self.sdmaProcessor.actual_corenum, self.cdmaProcessor.actual_corenum)
        CoreIdList = [str(i) for i in range(0, max_corenum)]
        CoreIdList.append('Overall')
        # import pdb; pdb.set_trace()
        if self.tiuProcessor.regList:
            self._process_tiu(self.tiuProcessor)
        else:
            self.tiuProcessor.sim_tiu_cycle_list.append('0.00us')

        if self.gdmaProcessor.regList:
            self._process_dma(self.gdmaProcessor, True)
            self.columns.extend(['simGdmaCycle', 'GdmaDdrAvgBandwidth(GB/s)','GdmaL2AvgBandwidth(GB/s)', 'GdmaAvgDdrBurstLength'])
        else:
            self.gdmaProcessor.sim_dma_cycle_list.append('0.00us')

        if self.sdmaProcessor.regList:
            self._process_dma(self.sdmaProcessor, False)
            self.columns.extend(['simSdmaCycle', 'SdmaDdrAvgBandwidth(GB/s)', 'SdmaAvgDdrBurstLength'])
        else:
            self.sdmaProcessor.sim_dma_cycle_list.append('0.00us')

        if self.cdmaProcessor.regList:
            self._process_dma(self.cdmaProcessor, True)
            self.columns.extend(['simCdmaCycle', 'CdmaDdrAvgBandwidth(GB/s)','CdmaL2AvgBandwidth(GB/s)', 'CdmaAvgDdrBurstLength'])

        ParallelismList = []
        # 计算每个核心的并行性
        # import pdb; pdb.set_trace()
        for i in range(max_corenum):
            if self.simTotalCycles[i] == 0:
                ParallelismList.append('0.00%')
            else:
                parallelism = (Decimal(((self.tiuProcessor.sim_tiu_cycle_list[i] + 
                                        self.gdmaProcessor.sim_dma_cycle_list[i] + 
                                        self.sdmaProcessor.sim_dma_cycle_list[i]) * 100 / self.simTotalCycles[i]))
                            .quantize(Decimal("0.00")))
                ParallelismList.append(str(parallelism) + '%')
        # 计算所有核心的并行性
        if max_corenum > 1:
            if sum(self.simTotalCycles[:-1]) == 0:
                ParallelismList.append('0.00%')
            else:
                total_parallelism = Decimal(((max(self.tiuProcessor.sim_tiu_cycle_list[:-1]) + 
                                            max(self.gdmaProcessor.sim_dma_cycle_list[:-1]) + 
                                            max(self.sdmaProcessor.sim_dma_cycle_list[:-1])) / max(self.simTotalCycles[:-1]) * 100))
                ParallelismList.append(str(total_parallelism.quantize(Decimal("0.00"))) + '%')
        else:
            if sum(self.simTotalCycles[:-1]) == 0:
                ParallelismList.append('0.00%')
            else:
                total_parallelism = Decimal(((max(self.tiuProcessor.sim_tiu_cycle_list[:-1]) + 
                                            max(self.gdmaProcessor.sim_dma_cycle_list[:-1])) / max(self.simTotalCycles[:-1]) * 100))
                ParallelismList.append(str(total_parallelism.quantize(Decimal("0.00"))) + '%')
        self.data[0] = CoreIdList
        self.data[1] = ParallelismList
        self.data[2] = self.simTotalCycles
        # import pdb;pdb.set_trace()
        summaryData = transpose(self.data).tolist()
        summaryDf = pd.DataFrame(summaryData, columns=self.columns, index=None)
        return summaryDf
