#!/usr/bin/python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

from profile_helper.bmprofile_parser import BMProfileParser, parse_data_blocks, IterRecord, parse_monitor_bd, parse_monitor_gdma, parse_dyn_data, parse_dyn_extra, parse_fixed_length_items
from profile_helper.bmprofile_common import BlockType, GlobalInfo, Arch
from profile_helper.bmprofile_utils import re_key_value
from profile_helper.bm1690_defs import get_tiu_info, get_dma_info, get_tiu_info_dyn, get_dma_info_dyn
import os
import logging
from typing import List
from pathlib import Path
import itertools
import glob

class DynCpuInfo(object):
    def __init__(self, begin_cycle, end_cycle, type, inst_id) -> None:
        self.begin_cycle = begin_cycle
        self.end_cycle = end_cycle
        self.type = type
        self.inst_id = inst_id

class BMProfileParserPerfAI(BMProfileParser):
    def __init__(self):
        super().__init__()
        self.gdma_cmd = []
        self.bd_cmd = []
        self.sdma_cmd = []
        self.vsdma_cmd = []
        self.bd_monitor = []
        self.gdma_monitor = []
        self.sdma_monitor = []
        self.cdmlib_extra = []
        self.profile_sync_points = []
        self.in_dir = None
        self.out_dir = None
        self.is_dyn = False

    def parse_cmd(self, file_list):
        for infile in file_list:
            blocks = parse_data_blocks(infile)
            if blocks is None:
                continue
            item = IterRecord()
            item.command_info = []
            blocks_factory = {
                BlockType.MONITOR_GDMA.value: (item.monitor_gdma, self.__parse_monitor_gdma),
                # # include sdma, vsdma pmu data
                BlockType.MONITOR_SDMA.value: (item.monitor_sdma, self.__parse_monitor_sdma),
                BlockType.MONITOR_BD.value: (item.monitor_bd, self.__parse_monitor_tiu),
                BlockType.DYN_DATA.value: (item.dyn_data, self.__parse_dyn_data),
                BlockType.COMMAND.value: (item.command_info, self.__parse_command_info)
            }
            for block in blocks:
                item_list, item_func = blocks_factory.get(
                    block.type.value, (0, lambda x, y: 0))
                item_func(item_list, block.content)
            if item.command_info:
                self.__read_command_data(item)
            elif item.dyn_data:
                self.__read_dyn_command_data(item)
            else:
                logging.fatal("can't find cmd data.")
                exit(-1)

    def parse(self, in_dir):
        self.in_dir = in_dir
        if not os.path.exists(in_dir):
            logging.fatal("'{}' does not exist".format(in_dir))
            exit(-1)
        global_file_path = os.path.join(in_dir, self.global_filename)
        self.__parse_global_file(global_file_path)
        dyn_cmd = sorted(glob.glob(in_dir + "/cdm*.profile"))
        staic_cmd = sorted(glob.glob(in_dir + "/iter*.profile"))
        if staic_cmd:
            self.parse_cmd(staic_cmd)
        elif dyn_cmd:
            self.is_dyn = True
            self.parse_cmd(dyn_cmd)
        else:
            logging.fatal("can't find cmd data.".format(in_dir))
            exit(-1)

    def to_txt(self, out_dir):
        assert self.bd_monitor != [] and self.gdma_monitor != [], ""
        # if not self.is_dyn:
        #     self.__cycle2time()
        #     self.__align_core_time()
        #     self.__time2cycle()
        # else:
        self.__align_core_time()
        self.__shift_time()

        self.__fix_to_core_num()
        self.out_dir = out_dir

        # 1. make file
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)
        dma_file = os.path.join(self.out_dir, "tdmaRegInfo_{}.txt")
        tiu_file = os.path.join(self.out_dir, "tiuRegInfo_{}.txt")

        # 2. write file
        for idx, (bd, gdma, sdma, bd_cmd, gdma_cmd, sdma_cmd) \
            in enumerate(zip(self.bd_monitor, self.gdma_monitor, \
                             self.sdma_monitor, self.bd_cmd, self.gdma_cmd, \
                             self.sdma_cmd)):
            # wirte dma
            with open(dma_file.format(idx), 'w') as f:
                f.write("__CHIP_ARCH_ARGS__\n")
                f.write("".join(f"\t{key}: {value}\n" for key,
                        value in self.archlib.DMA_ARCH.items()))
                for j in gdma:
                    try:
                        reg_info = gdma_cmd[j.inst_id].pop(0)
                    except (KeyError, IndexError):
                        continue
                    dma_info: dict = self.__get_gdma_info(j, reg_info)
                    dma_info["Core Id"] = idx
                    f.write("__TDMA_REG_INFO__\n")
                    f.write(
                        "".join(f"\t{key}: {value}\n" for key, value in dma_info.items()))
                for j in sdma:
                    reg_info = sdma_cmd[j.inst_id].pop(0)
                    dma_info: dict = self.__get_gdma_info(j, reg_info, 3)
                    dma_info["Core Id"] = idx
                    f.write("__TDMA_REG_INFO__\n")
                    f.write(
                        "".join(f"\t{key}: {value}\n" for key, value in dma_info.items()))
            # wirte tiu
            with open(tiu_file.format(idx), 'w') as f:
                f.write("__CHIP_ARCH_ARGS__\n")
                f.write("".join(f"\t{key}: {value}\n" for key,
                        value in self.archlib.TIU_ARCH.items()))
                for i, j in enumerate(bd):
                    reg_info = bd_cmd[j.inst_id].pop(0)
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
        # write cpu info
        cpu_file = os.path.join(self.out_dir, "cpuInfo_{}.txt")
        for i, cdm_cpu in enumerate(self.cdmlib_extra):
            with open(cpu_file.format(i), 'w') as f:
                for j in cdm_cpu:
                    info = f"core: {i} type: {self.archlib.DynRecordType(j.type).name:<14} " \
                        f"{self.archlib.EngineType(j.engine).name:<5} time_stamp(us): {j.begin_cycle/1000. :<12} " \
                        f"cmd_type: {j.des_tsk_typ:<2}:{j.des_tsk_eu_typ:>2}   " \
                        f"inst_id: {f'{j.inst_id!s:<6}' if isinstance(j.inst_id, dict) else f'{j.inst_id:<6}'}\n"
                    f.write(info)

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
        assert(len(self.profile_sync_points) == len(self.bd_monitor))
        for i, (bd, gdma, cycle) in enumerate(zip(self.bd_monitor, self.gdma_monitor, self.profile_sync_points)):
            if i == 0:
                continue
            delta_cyle = cycle - self.profile_sync_points[0]
            for j1 in itertools.chain(bd, gdma):
                j1.inst_start_time = int(j1.inst_start_time - delta_cyle)
                j1.inst_end_time = int(j1.inst_end_time - delta_cyle)
        for i, (sdma, cycle) in enumerate(zip(self.sdma_monitor, self.profile_sync_points)):
            if i == 0:
                continue
            delta_cyle = cycle - self.profile_sync_points[0]
            for j1 in sdma:
                j1.inst_start_time = int(j1.inst_start_time - delta_cyle)
                j1.inst_end_time = int(j1.inst_end_time - delta_cyle)
                assert(j1.inst_start_time  > 0 and j1.inst_end_time > 0)
        for i, (cpu, cycle) in enumerate(zip(self.cdmlib_extra, self.profile_sync_points)):
            if i == 0:
                continue
            delta_cyle = cycle - self.profile_sync_points[0]
            for j1 in cpu:
                j1.begin_cycle = int(j1.begin_cycle - delta_cyle)

    def __shift_time(self):
        start_cycle = self.gdma_monitor[0][0].inst_start_time

        for _, (bd, gdma) in enumerate(zip(self.bd_monitor, self.gdma_monitor)):
            start_cycle = min(bd[0].inst_start_time,
                              start_cycle, gdma[0].inst_start_time)
        for sdma in self.sdma_monitor:
            if sdma:
                start_cycle = min(sdma[0].inst_start_time, start_cycle)

        for _, (bd, gdma) in enumerate(zip(self.bd_monitor, self.gdma_monitor)):
            for j1 in itertools.chain(bd, gdma):
                j1.inst_start_time = int(j1.inst_start_time - start_cycle)
                j1.inst_end_time = int(j1.inst_end_time - start_cycle)
                assert(j1.inst_start_time  >= 0 and j1.inst_end_time >= 0)
        for sdma in self.sdma_monitor:
            for j1 in sdma:
                j1.inst_start_time = int(j1.inst_start_time - start_cycle)
                j1.inst_end_time = int(j1.inst_end_time - start_cycle)
        for i, cdm_cpu in enumerate(self.cdmlib_extra):
            g_cmd_time = self.gdma_cmd[i][self.gdma_monitor[i][0].inst_id][0].begin_cycle
            shift = g_cmd_time - start_cycle - self.gdma_monitor[i][0].inst_start_time
            for j1 in cdm_cpu:
                j1.begin_cycle = int(j1.begin_cycle - start_cycle - shift)

    def __parse_dyn_extra(self, dyn_extra_data: List, raw_data):
        tmp = parse_dyn_extra(raw_data)
        dyn_extra_data.update(tmp)

    def __parse_dyn_data(self, dyn_data: List, raw_data):
        tmp = parse_fixed_length_items(raw_data, self.archlib.ProfileFormat)
        if len(tmp) > 0:
            start_time = tmp[0].begin_cycle
            for i in tmp:
                i.begin_cycle -= start_time;
            dyn_data.extend(tmp)

    def __veryfy_cmd_id(self, data):
        delta_id = 0
        last_id = 0
        for c in data:
            if last_id > 65000 and c.inst_id < 1000:
                    delta_id += 65536
            last_id = c.inst_id
            c.inst_id += delta_id

    def __parse_monitor_tiu(self, monitor_tiu: List, raw_data):
        tmp = parse_monitor_bd(raw_data, self.archlib)
        self.__veryfy_cmd_id(tmp)
        monitor_tiu.append(tmp)

    def __parse_monitor_dma_base(self, raw_data):
        tmp = parse_monitor_gdma(raw_data, self.archlib)
        self.__veryfy_cmd_id(tmp)
        return tmp

    def __parse_monitor_gdma(self, monitor_gdma: List, raw_data):
        tmp = self.__parse_monitor_dma_base(raw_data)
        monitor_gdma.append(tmp)

    def __parse_monitor_sdma(self, monitor_sdma: List, raw_data):
        tmp = self.__parse_monitor_dma_base(raw_data)
        monitor_sdma.append(tmp)

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

    def __read_command_data(self, item):
        # static cmd include hole cores cmd per iter.profile
        bd_parser = self.archlib.BDCommandParser()
        gdma_parser = self.archlib.GDMACommandParser()
        for core_num, cmd_info in enumerate(item.command_info):
            bd_cmd = self.__base_read_command_data(cmd_info.bd_base,
                                                cmd_info.bd_offset,
                                                self.archlib.EngineType.BD,
                                                core_num, bd_parser)
            gdma_cmd = self.__base_read_command_data(cmd_info.gdma_base,
                                                    cmd_info.gdma_offset,
                                                    self.archlib.EngineType.GDMA,
                                                    core_num, gdma_parser)
            if core_num <= len(self.bd_cmd):
                self.__find_profile_sync_points(bd_cmd, item.monitor_bd[core_num],
                                                self.archlib.bd_sys_code, self.archlib.profile_sys_num)
                if item.monitor_bd[core_num]:
                    self.bd_cmd.append(self.__cmd_to_dict(bd_cmd))
                    self.bd_monitor.append(item.monitor_bd[core_num])
            else:
                for k, v in self.__cmd_to_dict(bd_cmd).items():
                    if k in self.bd_cmd[core_num]:
                        self.bd_cmd[core_num][k].extend(v)
                    else:
                        self.bd_cmd[core_num][k] = [v]
                self.bd_monitor[core_num].extend(item.monitor_bd[core_num])
            if core_num <= len(self.gdma_cmd):
                self.__find_profile_sync_points(gdma_cmd, item.monitor_gdma[core_num],
                                                self.archlib.dma_sys_code, self.archlib.profile_sys_num)
                if item.monitor_gdma[core_num]:
                    self.gdma_cmd.append(self.__cmd_to_dict(gdma_cmd))
                    self.gdma_monitor.append(item.monitor_gdma[core_num])
            else:
                for k, v in self.__cmd_to_dict(gdma_cmd).items():
                    if k in self.gdma_cmd[core_num]:
                        self.gdma_cmd[core_num][k].extend(v)
                    else:
                        self.gdma_cmd[core_num][k] = [v]
                self.gdma_monitor[core_num].extend(item.monitor_gdma[core_num])

    @staticmethod
    def __cmd_to_dict(core_cmd):
        cmd_map = {}
        for i in core_cmd:
            # cmd id - 1 == pmu_isnt_id
            # eg. cmd id : 1,2,3,4 ----- pmu id 0,1,2,3
            if hasattr(i, 'cmd_id'):
                inst_id = i.cmd_id
            else:
                inst_id = i.inst_id
            inst_id -= 1
            if  inst_id not in cmd_map:
                cmd_map[inst_id] = [i]
            else:
                cmd_map[inst_id].append(i)
        return cmd_map

    def __find_profile_sync_points(self, cmd, monitor, sys_code, cmd_offset=0):
        # sys_code:
        # dma sys tsk_typ 6  eu_typ:[3 send, 4 wait]
        # tiu sys tsk_typ 15 eu_typ:[8 send, 9 wait]
        send_wait = {self.archlib.dma_sys_code: [3, 4],
                     self.archlib.bd_sys_code: [8, 9]}
        sys_num = 0
        time_point = 0
        for _ in range(cmd_offset):
            # current for bmodel case cmdbuf don't contain forfile's sync_all cmd
            m = monitor.pop(0)
            time_point = m.inst_end_time
            sys_num += 1

        extra_zero_id = []
        for i in range(len(monitor)):
            # bmodel core0 pmu recored extra ins_id 0 that not in cmd
            if i > 0 and monitor[i].inst_id == 0 and monitor[i - 1].inst_id == 0:
                extra_zero_id.append(i - 1)
        for i in extra_zero_id[::-1]:
            monitor.pop(i)
        for i, (c, m) in enumerate(zip(cmd, monitor)):
            des_tsk_typ, des_tsk_eu_typ = -1, -1
            if hasattr(c, 'reg'):
                if hasattr(c.reg, 'tsk_typ'):
                    des_tsk_typ = c.reg.tsk_typ
                    des_tsk_eu_typ = c.reg.tsk_eu_typ
                if hasattr(c.reg, 'cmd_type'):
                    des_tsk_typ = c.reg.cmd_type
                    des_tsk_eu_typ = c.reg.cmd_special_function
            else:
                des_tsk_typ = c.des_tsk_typ
                des_tsk_eu_typ = c.des_tsk_eu_typ
            if des_tsk_typ != sys_code or sys_num == self.archlib.profile_sys_num:
                break
            if des_tsk_eu_typ not in send_wait[sys_code]:
                break
            if des_tsk_eu_typ == 9:
                time_point = m.inst_end_time
            sys_num += 1
        if sys_code == self.archlib.bd_sys_code:
            self.profile_sync_points.append(time_point)
        if cmd_offset != self.archlib.profile_sys_num:
            for _ in range(sys_num):
                _m = monitor.pop(0)
                _c = cmd.pop(0)
        if sys_num == 0:
            logging.warn("can't find sync cmd at begin.")

    def __read_dyn_command_data(self, item):
        # just for tpudnn
        # dyn cmd include single core cmd per cdm.profile
        if item.dyn_data:
            dyn_data = item.dyn_data
            assert(len(dyn_data)>0)
            gdma_cmd = []
            sdma_cmd = []
            bd_cmd = []
            for d in dyn_data[1:]:  # skip init record
                if d.type == self.archlib.DynRecordType.NODE_SET.value:
                    if d.engine == self.archlib.EngineType.GDMA.value:
                        gdma_cmd.append(d)
                    elif d.engine == self.archlib.EngineType.SDMA.value:
                        sdma_cmd.append(d)
                    elif d.engine == self.archlib.EngineType.VSDMA.value:
                        sdma_cmd.append(d)
                    elif d.engine == self.archlib.EngineType.BD.value:
                        bd_cmd.append(d)
            self.__find_profile_sync_points(bd_cmd, item.monitor_bd[0], self.archlib.bd_sys_code)
            self.__find_profile_sync_points(sdma_cmd, item.monitor_sdma[0], self.archlib.dma_sys_code)
            self.__find_profile_sync_points(gdma_cmd, item.monitor_gdma[0], self.archlib.dma_sys_code)
            valid_engine = 0
            if item.monitor_sdma:
                self.sdma_monitor.append(item.monitor_sdma[0])
                valid_engine += 1
            if item.monitor_gdma:
                self.gdma_monitor.append(item.monitor_gdma[0])
                valid_engine += 1
            if item.monitor_bd:
                self.bd_monitor.append(item.monitor_bd[0])
                valid_engine += 1
            self.bd_cmd.append(self.__cmd_to_dict(bd_cmd))
            self.gdma_cmd.append(self.__cmd_to_dict(gdma_cmd))
            self.sdma_cmd.append(self.__cmd_to_dict(sdma_cmd))
            # skip profile call sync_all
            self.cdmlib_extra.append(dyn_data[1 + valid_engine * self.archlib.profile_sys_num:])

    def __fix_to_core_num(self):
        core_num = len(self.bd_cmd)
        self.sdma_cmd += [{}] * (core_num - len(self.sdma_cmd))
        self.vsdma_cmd += [{}] * (core_num - len(self.vsdma_cmd))
        self.sdma_monitor += [{}] * (core_num - len(self.sdma_monitor))

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

    def __get_gdma_info(self, monitor_info, reg_info, engine_id=1):
        if self.is_dyn:
            return get_dma_info_dyn(monitor_info, reg_info, engine_id)
        else:
            return get_dma_info(monitor_info, reg_info)

    def __get_tiu_info(self, monitor_info, reg_info):
        if self.is_dyn:
            return get_tiu_info_dyn(monitor_info, reg_info)
        else:
            return get_tiu_info(monitor_info, reg_info)


if __name__ == "__main__":
    bmProfile = BMProfileParserPerfAI()
    # bmProfile.parse("/workspace/tpu-mlir/tmp/bmprofile_data-1_v2")
    # bmProfile.parse("/workspace/workdir/prf/cdm_profile_data-0_core1")
    bmProfile.parse("/workspace/workdir/prf/cdm_profile_data-0_core8")
    # bmProfile.to_txt('tmp')
