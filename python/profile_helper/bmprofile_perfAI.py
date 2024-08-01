#!/usr/bin/python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

from profile_helper.bmprofile_parser import BMProfileParser, parse_data_blocks, IterRecord, parse_monitor_bd, parse_monitor_gdma
from profile_helper.bmprofile_common import BlockType, GlobalInfo, Arch
from profile_helper.bmprofile_utils import re_key_value
from profile_helper.bm1688_defs import get_tiu_info, get_dma_info
import os
import logging
from typing import List
from pathlib import Path
import itertools


class BMProfileParserPerfAI(BMProfileParser):
    def __init__(self):
        super().__init__()
        self.gdma_cmd = []
        self.bd_cmd = []
        self.bd_monitor = []
        self.gdma_monitor = []
        self.in_dir = None
        self.out_dir = None

    def parse(self, in_dir):
        self.in_dir = in_dir
        if not os.path.exists(in_dir):
            logging.fatal("'{}' does not exist".format(in_dir))
            exit(-1)
        no_perf_data = True
        global_file_path = os.path.join(in_dir, self.global_filename)
        global_info = self.__parse_global_file(global_file_path)
        iter_count = 0
        while True:
            block_filename = self.iter_prefix+str(iter_count)+".profile"
            iter_count += 1
            block_filename = os.path.join(in_dir, block_filename)
            blocks = parse_data_blocks(block_filename)
            if blocks is None:
                break
            item = IterRecord()
            item.command_info = []
            blocks_factory = {
                BlockType.MONITOR_GDMA.value: (item.monitor_gdma, self.__parse_monitor_gdma),
                BlockType.MONITOR_BD.value: (item.monitor_bd, self.__parse_monitor_tiu),
                BlockType.COMMAND.value: (
                    item.command_info, self.__parse_command_info)
            }
            for block in blocks:
                item_list, item_func = blocks_factory.get(
                    block.type.value, (0, lambda x, y: 0))
                item_func(item_list, block.content)
            # print(item)
            for core_num, cmd_info in enumerate(item.command_info):
                self.__read_command_data(cmd_info, core_num)

    def to_txt(self, out_dir):
        assert self.bd_monitor != [] and self.gdma_monitor != [], ""
        self.__cycle2time()
        self.__align_core_time()
        self.__shift_time()
        self.__time2cycle()
        self.out_dir = out_dir

        # 1. make file
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)
        dma_file = os.path.join(self.out_dir, "tdmaRegInfo_{}.txt")
        tiu_file = os.path.join(self.out_dir, "tiuRegInfo_{}.txt")

        # 2. write file
        for idx, (bd, gdma, bd_cmd, gdma_cmd) in enumerate(zip(self.bd_monitor, self.gdma_monitor, self.bd_cmd, self.gdma_cmd)):
            # wirte gdma
            with open(dma_file.format(idx), 'w') as f:
                f.write("__CHIP_ARCH_ARGS__\n")
                f.write("".join(f"\t{key}: {value}\n" for key,
                        value in self.archlib.DMA_ARCH.items()))
                for j in gdma:
                    reg_info = gdma_cmd[j.inst_id]
                    dma_info: dict = self.__get_gdma_info(j, reg_info)
                    dma_info["Core Id"] = idx
                    f.write("__TDMA_REG_INFO__\n")
                    f.write(
                        "".join(f"\t{key}: {value}\n" for key, value in dma_info.items()))
            # wirte tiu
            with open(tiu_file.format(idx), 'w') as f:
                f.write("__CHIP_ARCH_ARGS__\n")
                f.write("".join(f"\t{key}: {value}\n" for key,
                        value in self.archlib.TIU_ARCH.items()))
                for j in bd:
                    reg_info = bd_cmd[j.inst_id]
                    tiu_info0, tiu_info1 = self.__get_tiu_info(j, reg_info)
                    tiu_info0["Core Id"] = idx
                    f.write("__TIU_REG_INFO__\n")
                    f.write(
                        "".join(f"\t{key}: {value}\n" for key, value in tiu_info0.items()))
                    f.write('{}:\n'.format(tiu_info0["Function Type"]))
                    f.write(
                        "".join(f"\t{key}: {value}\n" for key, value in tiu_info1.items()))
                    # f.write("".join(f"\tdes_{key}: {value}\n" for key, value in dict(reg_info).items()))
                    # f.write("\tMsg Id: \n\tSd\Wt Count: \n")

        # deprecated
        # start_cycle = self.gdma_monitor[0][0].inst_start_time
        # start_cycle = min(bd[0].inst_start_time, start_cycle, gdma[0].inst_start_time)
        # end_cycle = max(bd[-1].inst_end_time, end_cycle, gdma[-1].inst_end_time)
        # end_cycle = self.bd_monitor[0][-1].inst_end_time
        # total_cycle_file = os.path.join(self.out_dir, "simulatorTotalCycle.txt")
        # with open(total_cycle_file, 'w') as f:
        #   f.write("totalCycle: {}\n".format(end_cycle - start_cycle))

    def __cycle2time(self):
        for i in self.gdma_monitor:
            for j in i:
                j.inst_start_time = int(
                    j.inst_start_time / self.archlib.GDMA_FREQ * 1000)
                j.inst_end_time = int(
                    j.inst_end_time / self.archlib.GDMA_FREQ * 1000)
        for i in self.bd_monitor:
            for j in i:
                j.inst_start_time = int(
                    j.inst_start_time / self.archlib.BD_FREQ * 1000)
                j.inst_end_time = int(
                    j.inst_end_time / self.archlib.BD_FREQ * 1000)

    def __time2cycle(self):
        for i in self.gdma_monitor:
            for j in i:
                j.inst_start_time = int(
                    j.inst_start_time * self.archlib.GDMA_FREQ / 1000)
                j.inst_end_time = int(
                    j.inst_end_time * self.archlib.GDMA_FREQ / 1000)
        for i in self.bd_monitor:
            for j in i:
                j.inst_start_time = int(
                    j.inst_start_time * self.archlib.BD_FREQ / 1000)
                j.inst_end_time = int(
                    j.inst_end_time * self.archlib.BD_FREQ / 1000)

    def __align_core_time(self):
        first_wait_cmd_id = []
        first_wait_cmd_cycle = []

        # remove first wait
        for bd_cmd, bd_monitor in zip(self.bd_cmd, self.bd_monitor):
            if bd_cmd[bd_monitor[0].inst_id].op_name == "system.send_msg":
                bd_monitor.pop(0)
        # find first wait in core 0 and core 1
        for i in self.bd_cmd:
            for j in i:
                if j.op_name == "system.wait_msg":
                    first_wait_cmd_id.append(j.cmd_id)
                    break
        for cmd_id, item in zip(first_wait_cmd_id, self.bd_monitor):
            for j in item:
                if j.inst_id == cmd_id - 1:
                    first_wait_cmd_cycle.append(j.inst_start_time)
                    break
        num_one_flag = False
        for bd, gdma, cycle in zip(self.bd_monitor, self.gdma_monitor, first_wait_cmd_cycle):
            if not num_one_flag:
                num_one_flag = True
                continue
            delta_cyle = cycle - first_wait_cmd_cycle[0]
            for j1 in itertools.chain(bd, gdma):
                j1.inst_start_time = int(j1.inst_start_time - delta_cyle)
                j1.inst_end_time = int(j1.inst_end_time - delta_cyle)

    def __shift_time(self):
        start_cycle = self.gdma_monitor[0][0].inst_start_time
        for _, (bd, gdma) in enumerate(zip(self.bd_monitor, self.gdma_monitor)):
            start_cycle = min(bd[0].inst_start_time,
                              start_cycle, gdma[0].inst_start_time)
        for _, (bd, gdma) in enumerate(zip(self.bd_monitor, self.gdma_monitor)):
            for j1 in itertools.chain(bd, gdma):
                j1.inst_start_time = int(j1.inst_start_time - start_cycle)
                j1.inst_end_time = int(j1.inst_end_time - start_cycle)

    def __parse_monitor_tiu(self, monitor_tiu: List, raw_data):
        tmp = parse_monitor_bd(raw_data, self.archlib)
        delta_id = 0
        last_id = 0
        for c in tmp:
            if last_id > 65000 and c.inst_id < 1000:
                    delta_id += 65536
            last_id = c.inst_id
            c.inst_id += delta_id
        self.bd_monitor.append(tmp)
        monitor_tiu.append(tmp)

    def __parse_monitor_gdma(self, monitor_gdma: List, raw_data):
        tmp = parse_monitor_gdma(raw_data, self.archlib)
        zero_position = []
        for idx, dma in enumerate(tmp):
            if dma.inst_id == 0:
                zero_position.append(idx)
        if len(zero_position) == 0:
            left = 0
            right = len(tmp)
        elif len(zero_position) < 2:
            left = 1
            right = len(tmp)
        elif len(zero_position) == 2:
            left = zero_position[0] + 1
            right = zero_position[1]
        else:
            left = zero_position[1]
            right = zero_position[-1]
        self.gdma_monitor.append(tmp[left:right])
        monitor_gdma.append(tmp)

    def __parse_command_info(self, command_info: List, raw_data):
        command_info.append(
            self._BMProfileParser__parse_command_info(raw_data))

    def __parse_global_file(self, filename):
        assert os.path.isfile(filename)
        re_arch = re_key_value("", "arch")
        ginfo = GlobalInfo()
        with open(filename) as f:
            for self.line in f:
                if len(self.line) == 0:
                    continue
                if self.match(re_arch) and self.archlib is None:
                    ginfo.set_arch(self.enum_val("arch", Arch))
                    self.archlib = ginfo.archlib
                    break

    def __read_command_data(self, cmd_info, core_num):
        gdma_num = 0
        bd_num = 0
        for num_info in cmd_info.group:
            gdma_num += num_info[0]
            bd_num += num_info[1]
        gdma_parser = self.archlib.GDMACommandParser()
        bd_parser = self.archlib.BDCommandParser()
        gdma_cmd = self.__base_read_command_data(cmd_info.gdma_base,
                                                 cmd_info.gdma_offset,
                                                 self.archlib.EngineType.GDMA,
                                                 core_num, gdma_parser)
        bd_cmd = self.__base_read_command_data(cmd_info.bd_base,
                                               cmd_info.bd_offset,
                                               self.archlib.EngineType.BD,
                                               core_num, bd_parser)
        self.gdma_cmd.append(gdma_cmd)
        self.bd_cmd.append(bd_cmd)

    def __base_read_command_data(self, base, offset, engine_type, core_num, command_parser):
        basename = "cmd_%x_%d_%d.dat"
        command_filename = os.path.join(
            self.in_dir, basename % (base, core_num, engine_type.value))
        if not os.path.isfile(command_filename):
            return []
        with open(command_filename, "rb") as f:
            f.seek(offset)
            raw_data = f.read()
            command_list = command_parser.parse(raw_data)
        return command_list

    def __get_gdma_info(self, monitor_info, reg_info):
        return get_dma_info(monitor_info, reg_info)

    def __get_tiu_info(self, monitor_info, reg_info):
        return get_tiu_info(monitor_info, reg_info)


if __name__ == "__main__":
    bmProfile = BMProfileParserPerfAI()
    bmProfile.parse("/workspace/tpu-mlir/tmp/bmprofile_data-1_v2")
    bmProfile.to_txt('tmp')
