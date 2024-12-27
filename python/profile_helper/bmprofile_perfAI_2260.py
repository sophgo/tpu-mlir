#!/usr/bin/python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

from profile_helper.bmprofile_parser import BMProfileParser, parse_data_blocks, IterRecord, parse_monitor_bd, parse_monitor_gdma, parse_monitor_cdma, parse_dyn_data, parse_dyn_extra, parse_fixed_length_items
from profile_helper.bmprofile_common import BlockType, GlobalInfo, Arch
from profile_helper.bmprofile_utils import re_key_value
from profile_helper.bm1690_defs import get_tiu_info, get_dma_info, get_tiu_info_dyn, get_dma_info_dyn
import os, re
import logging
from typing import List
from pathlib import Path
import itertools
import glob
from tqdm import tqdm

def get_cmd_id(c):
    if hasattr(c, 'cmd_id'):
        cmd_id = c.cmd_id
    else:
        cmd_id = c.inst_id
    return cmd_id

class DynCpuInfo(object):
    def __init__(self, begin_cycle, end_cycle, type, inst_id) -> None:
        self.end_cycle = end_cycle
        self.type = type
        self.inst_id = inst_id

class BMProfileParserPerfAI(BMProfileParser):
    def __init__(self):
        super().__init__()
        self.gdma_pairs = []
        self.sdma_pairs = []
        self.cdma_pairs = []
        self.bd_pairs = []
        self.cdmlib_extra = []
        self.profile_sync_points = []
        self.in_dir = None
        self.out_dir = None
        self.is_dyn = False

    def parse_cdma_cmd(self, file_list):
        print("Parsing...")
        self.cdma_pairs = [[] for _ in range(self.archlib.CDMA_NUM)]
        for idx, infile in enumerate(file_list):
            blocks = parse_data_blocks(infile)
            if blocks is None or blocks == []:
                continue
            item = IterRecord()
            item.command_info = []
            blocks_factory = {
                BlockType.MONITOR_CDMA.value: (item.monitor_cdma, self.__parse_monitor_cdma),
            }
            for block in blocks:
                item_list, item_func = blocks_factory.get(
                    block.type.value, (0, lambda x, y: 0))
                item_func(item_list, block.content)
            for m in item.monitor_cdma[0]:
                self.cdma_pairs[idx].append({"monitor": m, "cmd": None})
                # todo pars cmd and pmu pair (cdma has des mode?)


    def parse_cmd(self, file_list):
        self.gdma_parser = self.archlib.GDMACommandParser()
        self.bdc_parser = self.archlib.BDCommandParser()
        print("Parsing...")
        self.cdma_cmd = [[] for _ in range(self.archlib.CDMA_NUM)]
        for infile in tqdm(file_list):
            blocks = parse_data_blocks(infile)
            if blocks is None:
                continue
            item = IterRecord()
            item.command_info = []
            item.dyn_extra = []
            blocks_factory = {
                BlockType.MONITOR_GDMA.value: (item.monitor_gdma, self.__parse_monitor_gdma),
                # # include sdma, vsdma pmu data
                BlockType.MONITOR_SDMA.value: (item.monitor_sdma, self.__parse_monitor_sdma),
                BlockType.MONITOR_BD.value: (item.monitor_bd, self.__parse_monitor_tiu),
                BlockType.DYN_DATA.value: (item.dyn_data, self.__parse_dyn_data),
                BlockType.COMMAND.value: (item.command_info, self.__parse_command_info),
                BlockType.DYN_EXTRA.value: (item.dyn_extra, self.__parse_dyn_extra)
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
                self.__read_pure_pmu_data(item)

    def parse(self, in_dir):
        def sort_key_func(filename):
            numbers = re.findall(r'\d+', filename)
            return [int(num) for num in numbers]
        self.in_dir = in_dir
        if not os.path.exists(in_dir):
            logging.fatal("'{}' does not exist".format(in_dir))
            exit(-1)
        global_file_path = os.path.join(in_dir, self.global_filename)
        self.__parse_global_file(global_file_path)
        dyn_cmd = sorted(glob.glob(in_dir + "/cdmlib*.profile"), key=sort_key_func)
        staic_cmd = sorted(glob.glob(in_dir + "/iter*.profile"), key=sort_key_func)
        cdma_cmd = sorted(glob.glob(in_dir + "/cdma*.profile"), key=sort_key_func)
        if staic_cmd:
            self.parse_cmd(staic_cmd)
        elif dyn_cmd:
            self.is_dyn = True
            self.parse_cmd(dyn_cmd)
        if cdma_cmd:
            self.parse_cdma_cmd(cdma_cmd)
        # else:
        #     logging.fatal("can't find cmd data.".format(in_dir))
        #     exit(-1)

    def to_txt(self, out_dir):
        assert self.bd_pairs != [] and self.gdma_pairs != [], ""
        self.__align_core_time()
        self.__shift_time()
        self.out_dir = out_dir
        # make file
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)
        dma_file = os.path.join(self.out_dir, "tdmaRegInfo_{}.txt")
        tiu_file = os.path.join(self.out_dir, "tiuRegInfo_{}.txt")
        cdma_file = os.path.join(self.out_dir, "cdmaPmuInfo_{}.txt")
        # write engine info
        print("Write engine info...")
        for idx, pair in tqdm(enumerate(self.gdma_pairs)):
            self.__write_engine_info(dma_file, idx, pair, self.archlib.EngineType.GDMA)
        for idx, pair in tqdm(enumerate(self.sdma_pairs)):
            self.__write_engine_info(dma_file, idx, pair, self.archlib.EngineType.SDMA, False)
        for idx, pair in tqdm(enumerate(self.bd_pairs)):
            self.__write_engine_info(tiu_file, idx, pair, self.archlib.EngineType.BD)
        for idx, pair in tqdm(enumerate(self.cdma_pairs)):
            self.__write_engine_info(cdma_file, idx, pair, self.archlib.EngineType.CDMA)
        # write cpu info
        cpu_file = os.path.join(self.out_dir, "cpuInfo_{}.txt")
        for i, cdm_cpu in enumerate(self.cdmlib_extra):
            with open(cpu_file.format(i), 'w') as f:
                for j in cdm_cpu:
                    info = f"core: {i} type: {self.archlib.DynRecordType(j.type).name:<14} " \
                        f"{self.archlib.EngineType(j.engine).name:<5} " \
                        f"cmd_type: {j.des_tsk_typ:<2}:{j.des_tsk_eu_typ:>2}   " \
                        f"inst_id: {f'{j.inst_id!s:<6}' if isinstance(j.inst_id, dict) else f'{j.inst_id:<6}'}\n"
                    f.write(info)

    def __write_engine_info(self, nfile, idx, pairs, engine, new_file=True):
        fmode = 'w'
        if not new_file:
            fmode = 'a'
        if engine in [self.archlib.EngineType.GDMA, self.archlib.EngineType.SDMA]:
            fn = self.__get_gdma_info
            arch = self.archlib.DMA_ARCH
            tag = "__TDMA_REG_INFO__\n"
        elif engine == self.archlib.EngineType.BD:
            fn = self.__get_tiu_info
            arch = self.archlib.TIU_ARCH
            tag = "__TIU_REG_INFO__\n"
        elif engine == self.archlib.EngineType.CDMA:
            if pairs:
                with open(nfile.format(idx), fmode) as f:
                    for n, p in enumerate(pairs):
                        p = p["monitor"]
                        info = f"[{idx:<2}]---> cdma record #{n:<7} inst_id: {p.inst_id:<10} thread_id: {p.thread_id:<4} start_time: {p.inst_start_time:<14} " \
                            f"inst_end_time: {p.inst_end_time:<14} cycle: {p.inst_end_time - p.inst_start_time}\n"
                        f.write(info)
                with open(nfile.format(f'{idx}_cmd'), fmode) as f:
                    for n, p in enumerate(self.cdma_cmd[idx]):
                        cmd_type, cmd_func = self.archlib.getCdmaFunctionName(p.des_tsk_typ, p.des_tsk_eu_typ)
                        info = f"[{idx:<2}]---> cdma record #{n:<7} inst_id: {p.inst_id:<10} cmd_type: {cmd_type:<20} cmd_func: {cmd_func:<8}\n"
                        f.write(info)
            return

        else:
            raise ValueError(f"Not support parse {self.archlib.EngineType(engine).name} now.")
        with open(nfile.format(idx), fmode) as f:
            if new_file:
                f.write("__CHIP_ARCH_ARGS__\n")
                f.write("".join(f"\t{key}: {value}\n" for key,
                        value in arch.items()))
            for p in pairs:
                info, extra = fn(p["monitor"], p["cmd"], idx, engine.value)
                info["Core Id"] = idx
                f.write(tag)
                f.write(
                    "".join(f"\t{key}: {value}\n" for key, value in info.items()))
                if extra is not None:
                    f.write('{}:\n'.format(info["Function Type"]))
                    f.write(
                        "".join(f"\t{key}: {value}\n" for key, value in extra.items()))

    def __align_core_time(self):
        assert(len(self.profile_sync_points) == len(self.bd_pairs))
        for i, (bd_pair, gdma_pair, cycle) in enumerate(zip(self.bd_pairs, self.gdma_pairs, self.profile_sync_points)):
            if i == 0:
                continue
            delta_cyle = cycle - self.profile_sync_points[0]
            for j1 in itertools.chain(bd_pair, gdma_pair):
                j1["monitor"].inst_start_time = int(j1["monitor"].inst_start_time - delta_cyle)
                j1["monitor"].inst_end_time = int(j1["monitor"].inst_end_time - delta_cyle)
        for i, (sdma, cycle) in enumerate(zip(self.sdma_pairs, self.profile_sync_points)):
            if i == 0:
                continue
            delta_cyle = cycle - self.profile_sync_points[0]
            for j1 in sdma:
                j1["monitor"].inst_start_time = int(j1["monitor"].inst_start_time - delta_cyle)
                j1["monitor"].inst_end_time = int(j1["monitor"].inst_end_time - delta_cyle)
        # for i, (cdma, cycle) in enumerate(zip(self.cdma_pairs, self.profile_sync_points)):
        #     if i == 0:
        #         continue
        #     delta_cyle = cycle - self.profile_sync_points[0]
        #     for j1 in cdma:
        #         j1["monitor"].inst_start_time = int(j1["monitor"].inst_start_time - delta_cyle)
        #         j1["monitor"].inst_end_time = int(j1["monitor"].inst_end_time - delta_cyle)
        # for i, (cpu, cycle) in enumerate(zip(self.cdmlib_extra, self.profile_sync_points)):
        #     if i == 0:
        #         continue
        #     delta_cyle = cycle - self.profile_sync_points[0]
        #     for j1 in cpu:
        #         j1.begin_cycle = int(j1.begin_cycle - delta_cyle)

    def __shift_time(self):
        start_cycle = self.gdma_pairs[0][0]["monitor"].inst_start_time

        for _, (bd_pair, gdma_pair) in enumerate(zip(self.bd_pairs, self.gdma_pairs)):
            start_cycle = min(bd_pair[0]["monitor"].inst_start_time,
                              start_cycle, gdma_pair[0]["monitor"].inst_start_time)
        for sdma_pair in self.sdma_pairs:
            if sdma_pair:
                start_cycle = min(sdma_pair[0]["monitor"].inst_start_time, start_cycle)

        for _, (bd_pair, gdma_pair) in enumerate(zip(self.bd_pairs, self.gdma_pairs)):
            for j1 in itertools.chain(bd_pair, gdma_pair):
                j1["monitor"].inst_start_time = int(j1["monitor"].inst_start_time - start_cycle)
                j1["monitor"].inst_end_time = int(j1["monitor"].inst_end_time - start_cycle)
                assert(j1["monitor"].inst_start_time >= 0 and j1["monitor"].inst_end_time >= 0)
        for sdma_pair in self.sdma_pairs:
            for j1 in sdma_pair:
                j1["monitor"].inst_start_time = int(j1["monitor"].inst_start_time - start_cycle)
                j1["monitor"].inst_end_time = int(j1["monitor"].inst_end_time - start_cycle)
                assert(j1["monitor"].inst_start_time >= 0 and j1["monitor"].inst_end_time >= 0)

        # for cdma_pair in self.cdma_pairs:
        #     for j1 in cdma_pair:
        #         print(j1["monitor"].inst_start_time,  start_cycle)
        #         j1["monitor"].inst_start_time = int(j1["monitor"].inst_start_time - start_cycle)
        #         j1["monitor"].inst_end_time = int(j1["monitor"].inst_end_time - start_cycle)
        # # for i, cdm_cpu in enumerate(self.cdmlib_extra):
        #     g_cmd_time = self.gdma_pairs[i][0]["cmd"].begin_cycle
        #     shift = min(self.gdma_pairs[i][0]["monitor"].inst_start_time,
        #                 self.bd_pairs[i][0]["monitor"].inst_start_time)
        #     for j1 in cdm_cpu:
        #         j1.begin_cycle = int(j1.begin_cycle - g_cmd_time + shift)

    def __parse_dyn_data(self, dyn_data: List, raw_data):
        tmp = parse_fixed_length_items(raw_data, self.archlib.ProfileFormat)
        if len(tmp) > 0:
            # start_time = tmp[0].begin_cycle
            # for i in tmp:
            #     i.begin_cycle -= start_time;
            dyn_data.extend(tmp)

    def __veryfy_cmd_id(self, data):
        delta_id = 0
        last_id = 0
        for c in data:
            if last_id > 65000 and c.inst_id < 1000:
                    delta_id += 65536
            last_id = c.inst_id
            c.inst_id += delta_id

    def __veryfy_time(self, data):
        last_time = 0
        delta_time = 0
        uint32_max = 4294967295
        for c in data:
            current_time = c.inst_start_time
            if current_time < last_time:
                delta_time += uint32_max # uint32 max
            c.inst_start_time += delta_time
            c.inst_end_time += delta_time
            if c.inst_end_time < c.inst_start_time:
               # start not overflow but end does
               c.inst_end_time += uint32_max
            last_time = c.inst_end_time

    def __parse_monitor_tiu(self, monitor_tiu: List, raw_data):
        tmp = parse_monitor_bd(raw_data, self.archlib)
        self.__veryfy_cmd_id(tmp)
        self.__veryfy_time(tmp)
        monitor_tiu.append(tmp)

    def __parse_monitor_cdma(self, monitor_cdma: List, raw_data):
        tmp = parse_monitor_cdma(raw_data, self.archlib)
        self.__veryfy_cmd_id(tmp)
        self.__veryfy_time(tmp)
        monitor_cdma.append(tmp)


    def __parse_monitor_dma_base(self, raw_data):
        tmp = parse_monitor_gdma(raw_data, self.archlib)
        self.__veryfy_cmd_id(tmp)
        self.__veryfy_time(tmp)
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

    def __parse_dyn_extra(self, dyn_extra_data: List, raw_data):
        tmp = parse_dyn_extra(raw_data, True)
        dyn_extra_data.extend(tmp)

    def __parse_global_file(self, filename):
        assert os.path.isfile(filename), filename
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
        for core_num, cmd_info in enumerate(item.command_info):
            bd_cmd = self.__base_read_command_data(cmd_info.bd_base,
                                                cmd_info.bd_offset,
                                                self.archlib.EngineType.BD,
                                                core_num, self.bdc_parser)
            gdma_cmd = self.__base_read_command_data(cmd_info.gdma_base,
                                                    cmd_info.gdma_offset,
                                                    self.archlib.EngineType.GDMA,
                                                    core_num, self.gdma_parser)
            sdma_cmd = self.__base_read_command_data(cmd_info.sdma_base,
                                                    cmd_info.sdma_offset,
            # TODO tpuv7-runtime/model-runtime/runtime/src/sgruntime_bmodel.cpp:576
                                                    self.archlib.EngineType.VSDMA,
                                                    core_num, self.gdma_parser)

            bd_pair, _ = self.__find_profile_sync_points(bd_cmd, item.monitor_bd[core_num],
                                            self.archlib.bd_sys_code, self.archlib.profile_sys_num)
            gdma_pair, _ = self.__find_profile_sync_points(gdma_cmd, item.monitor_gdma[core_num],
                                                self.archlib.dma_sys_code, self.archlib.profile_sys_num)
            sdma_pair, _ = self.__find_profile_sync_points(sdma_cmd, item.monitor_sdma[core_num],
                                        self.archlib.dma_sys_code, self.archlib.profile_sys_num)

            if core_num <= len(self.bd_pairs):
                if item.monitor_bd[core_num]:
                    self.bd_pairs.append(bd_pair)
            else:
                self.bd_pairs[core_num].extend(bd_pair)
            if core_num <= len(self.gdma_pairs):
                if item.monitor_gdma[core_num]:
                    self.gdma_pairs.append(gdma_pair)
            else:
                self.gdma_pairs[core_num].extend(gdma_pair)
            if core_num <= len(self.sdma_pairs):
                if item.monitor_sdma[core_num]:
                    self.sdma_pairs.append(sdma_pair)
            else:
                self.sdma_pairs[core_num].extend(sdma_pair)

    @staticmethod
    def __get_cmd_type(cmd):
        des_tsk_typ, des_tsk_eu_typ = -1, -1
        if hasattr(cmd, 'reg'):
            if hasattr(cmd.reg, 'tsk_typ'):
                des_tsk_typ = cmd.reg.tsk_typ
                des_tsk_eu_typ = cmd.reg.tsk_eu_typ
            if hasattr(cmd.reg, 'cmd_type'):
                des_tsk_typ = cmd.reg.cmd_type
                des_tsk_eu_typ = cmd.reg.cmd_special_function
        else:
            des_tsk_typ = cmd.des_tsk_typ
            des_tsk_eu_typ = cmd.des_tsk_eu_typ
        return des_tsk_typ, des_tsk_eu_typ

    def __match_sections(self, monitor, cmd):
        m, n = len(monitor), len(cmd)
        dp = [[float('inf')] * (n + 1) for _ in range(m + 1)]
        path = [[None] * (n + 1) for _ in range(m + 1)]

        dp[0][0] = 0
        for i in range(1, m + 1):
            dp[i][0] = i
            path[i][0] = (i - 1, 0)
        for j in range(1, n + 1):
            dp[0][j] = j
            path[0][j] = (0, j - 1)

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if monitor[i - 1][0] == cmd[j - 1][0]:
                    dp[i][j] = dp[i - 1][j - 1]
                    path[i][j] = (i - 1, j - 1)
                else:
                    options = [
                        (dp[i - 1][j] + 1, (i - 1, j)),
                        (dp[i][j - 1] + 1, (i, j - 1)),
                        (dp[i - 1][j - 1] + 2, (i - 1, j - 1))
                    ]
                    dp[i][j], path[i][j] = min(options, key=lambda x: x[0])

        i, j = m, n
        result = []
        while i > 0 or j > 0:
            prev_i, prev_j = path[i][j]
            if prev_i == i - 1 and prev_j == j - 1:
                result.append((monitor[i - 1][1], cmd[j - 1][1]))
            elif prev_i == i - 1:
                result.append((monitor[i - 1][1], None))
            # for debug purpose none_pmu, cmd
            # else:
            #     result.append((None, cmd[j - 1][1]))
            i, j = prev_i, prev_j

        result.reverse()
        return result

    def __make_mix_pairs(self, cmd, monitor, sys_code):
        _cmd = []
        _monitor = []
        cmd_slice = []
        monitor_slice = []
        first_idx = monitor[0].inst_id + 1 if monitor else 1
        start_idx, last_idx = first_idx, first_idx
        for m in monitor:
            m_id = m.inst_id + 1
            if m_id <= last_idx and monitor_slice:
                _monitor.append((last_idx - start_idx + 1, monitor_slice))
                monitor_slice = []
                start_idx = m_id
            monitor_slice.append(m)
            last_idx = m_id
        if monitor_slice:
            _monitor.append((last_idx - start_idx + 1, monitor_slice))
        first_idx = get_cmd_id(cmd[0]) if cmd else 1
        start_idx, last_idx = first_idx, first_idx
        for c in cmd:
            cmd_id = get_cmd_id(c)
            if cmd_id <= last_idx and cmd_slice:
                _cmd.append((last_idx - start_idx + 1, cmd_slice))
                cmd_slice = []
                start_idx = cmd_id
            cmd_slice.append(c)
            last_idx = cmd_id
        if cmd_slice:
            _cmd.append((last_idx - start_idx + 1, cmd_slice))
        # compatible code for tpu-train: pmu miss sys pair reason unknow
        # force align
        if len(_cmd) == len(_monitor):
            for i in range(len(_cmd))[::-1]:
                if _cmd[i][0] == 2 and _monitor[i][0] == 2:
                    des_tsk_typ, _ = self.__get_cmd_type(_cmd[i][1][0])
                    if des_tsk_typ == sys_code:
                        continue
                if _cmd[i][0] == 2 and _cmd[i][0] != _monitor[i][0]:
                    _cmd.pop(i)
                    break
            if len(_cmd) == 1 and len(_monitor) == 1:
                if _cmd[-1][0] != _monitor[-1][0]:
                    max_len = max(_cmd[-1][0], _monitor[-1][0])
                    _cmd[-1] = (max_len,  _cmd[-1][1])
                    _monitor[-1] = (max_len,  _monitor[-1][1])
        pairs = self.__match_sections(_monitor, _cmd)
        # print("####################### pmu ###########################")
        # # for m in _monitor:
        # #     print(m[0], [i.inst_id + 1 for i in m[1]])
        # print([m[0] for m in _monitor])
        # print("####################### cmd ###########################")
        # # for m in _cmd:
        # #     print(m[0], [i.inst_id for i in m[1]])
        # print([m[0] for m in _cmd])
        return pairs


    def __make_pairs(self, cmd, monitor, sys_code, mix_mode=True):
        pairs = []
        if cmd == []:
            for m in monitor:
                pairs.append({"monitor": m, "cmd": None})
            return pairs
        for p_monitor, p_cmd in self.__make_mix_pairs(cmd, monitor, sys_code):
            if p_monitor is None:
                continue
            if p_cmd is None:
                # TODO get cmd from des comand and dyn command here
                if mix_mode:
                    for m in p_monitor:
                        pairs.append({"monitor": m, "cmd": None})
                continue
            # len(cmd) >= len(pmu) cause pmu will drop some data
            m_start_idx = p_monitor[0].inst_id
            c_start_idx = get_cmd_id(p_cmd[0])
            for m in p_monitor:
                m_idx = m.inst_id - m_start_idx
                for i, c in enumerate(p_cmd):
                    c_idx = get_cmd_id(c) - c_start_idx
                    if m_idx == c_idx:
                        pairs.append({"monitor": m, "cmd": c})
                        p_cmd = p_cmd[i+1:]
                        break
        return pairs

    def __find_profile_sync_points(self, cmd, monitor, sys_code, cmd_offset=0, omit_end_sys=True, pure_pmu=False):
        # sys_code:
        # dma sys tsk_typ 6  eu_typ:[3 send, 4 wait]
        # tiu sys tsk_typ 15 eu_typ:[8 send, 9 wait]
        send_wait = {self.archlib.dma_sys_code: [3, 4],
                     self.archlib.bd_sys_code: [8, 9]}
        sys_num = 0
        time_point = 0
        # for bmodel
        if cmd_offset:
            for _ in range(cmd_offset):
                # current for bmodel case cmdbuf don't contain forfile's sync_all cmd
                m = monitor.pop(0)
                time_point = m.inst_end_time
                sys_num += 1
        # skip last dma cpy for bmodel outputs, TODO parse from dyn_cmd
        mix_mode = True
        if cmd_offset and sys_code == self.archlib.dma_sys_code:
            mix_mode = False
        pairs = self.__make_pairs(cmd, monitor, sys_code, mix_mode)
        if cmd_offset and not pure_pmu:
            # for bmodel remove monitor date without cmd
            n = 0
            for item in pairs:
                if item["cmd"] is None:
                    n += 1
                else:
                    break
            for i in range(n):
                pairs.pop(0)
        for i, j in enumerate(pairs):
            if j["cmd"] is None:
                break
            des_tsk_typ, des_tsk_eu_typ = self.__get_cmd_type(j["cmd"])
            if des_tsk_typ != sys_code or sys_num == self.archlib.profile_sys_num:
                break
            if des_tsk_eu_typ not in send_wait[sys_code]:
                break
            if des_tsk_eu_typ == 9:
                time_point = j["monitor"].inst_end_time
            sys_num += 1
        if sys_code == self.archlib.bd_sys_code:
            self.profile_sync_points.append(time_point)
        if cmd_offset != self.archlib.profile_sys_num:
            for _ in range(sys_num):
                _m = pairs.pop(0)
        if sys_num == 0:
            logging.warn("can't find sync cmd at begin.")
        sys_end_num = 0
        if omit_end_sys:
            extra_sys = []
            part = []
            for i in range(len(pairs))[::-1]:
                if pairs[i]["cmd"] is None:
                    if self.is_dyn:
                        break
                    else:
                        # for bmodel
                        extra_sys.append(i)
                        continue
                des_tsk_typ, _ = self.__get_cmd_type(pairs[i]["cmd"])
                if des_tsk_typ != sys_code:
                    break
                else:
                    part.append(i)
                if pairs[i]["monitor"].inst_id == 0:
                    extra_sys.extend(part)
                    part = []
            sys_end_num = len(extra_sys)
            for i in extra_sys:
                pairs.pop(i)
            # assert(len(pairs) != 0)
        return pairs, sys_num

    def __read_dyn_command_data(self, item):
        # just for tpudnn
        # dyn cmd include single core cmd per cdm.profile
        if item.dyn_data:
            dyn_data = item.dyn_data
            assert(len(dyn_data)>0)
            gdma_cmd = []
            sdma_cmd = []
            bd_cmd = []
            for i, d in enumerate(dyn_data[1:]):  # skip init record
                if d.type == self.archlib.DynRecordType.NODE_SET.value:
                    if item.dyn_extra:
                        d.detailed_cmd = item.dyn_extra[i].content
                    if d.engine == self.archlib.EngineType.GDMA.value:
                        gdma_cmd.append(d)
                    elif d.engine == self.archlib.EngineType.SDMA.value:
                        sdma_cmd.append(d)
                    elif d.engine == self.archlib.EngineType.VSDMA.value:
                        sdma_cmd.append(d)
                    elif d.engine == self.archlib.EngineType.BD.value:
                        bd_cmd.append(d)
                    elif d.engine == self.archlib.EngineType.CDMA.value:
                        self.cdma_cmd[d.extra_info >> 7].append(d)
            bd_pair, bd_sys_num = self.__find_profile_sync_points(bd_cmd, item.monitor_bd[0], self.archlib.bd_sys_code)
            sdma_pair, sdma_sys_num = self.__find_profile_sync_points(sdma_cmd, item.monitor_sdma[0], self.archlib.dma_sys_code)
            gdma_pair, gdma_sys_num = self.__find_profile_sync_points(gdma_cmd, item.monitor_gdma[0], self.archlib.dma_sys_code)
            self.bd_pairs.append(bd_pair)
            self.gdma_pairs.append(gdma_pair)
            self.sdma_pairs.append(sdma_pair)
            # skip profile init and call sync_all
            self.cdmlib_extra.append(dyn_data[1 + bd_sys_num + sdma_sys_num + gdma_sys_num:])
        # if item.dyn_extra:
        #     print(item.dyn_extra )
        #     for k, v in item.dyn_extra.items():
        #         print(k)
        #         engine = self.archlib.EngineType((k >> 8) & 0x7)
        #         print(engine, engine == self.archlib.EngineType.GDMA)
        #         parser = self.archlib.GDMACommandParser()
        #         if engine == self.archlib.EngineType.GDMA:
        #             gdma_parser = parser.parse(v[0].content)
        #             print(gdma_parser)


    def __read_pure_pmu_data(self, item):
            bd_pair, _ = self.__find_profile_sync_points([], item.monitor_bd[0], self.archlib.bd_sys_code, self.archlib.profile_sys_num, pure_pmu=True)
            sdma_pair, _ = self.__find_profile_sync_points([], item.monitor_sdma[0], self.archlib.dma_sys_code, self.archlib.profile_sys_num, pure_pmu=True)
            gdma_pair, _ = self.__find_profile_sync_points([], item.monitor_gdma[0], self.archlib.dma_sys_code, self.archlib.profile_sys_num, pure_pmu=True)
            self.bd_pairs.append(bd_pair)
            self.gdma_pairs.append(gdma_pair)
            self.sdma_pairs.append(sdma_pair)

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
        # Compatible code
        end_name = "dma.sys.chain_end"
        if engine_type == self.archlib.EngineType.BD:
            end_name = "system.end"
        op_name = command_list[-1].op_name
        num = 0
        if op_name == end_name:
            for i in command_list[::-2]:
                if i.op_name == end_name:
                    num += 1
        for _ in range(num):
            command_list.pop()
        return command_list

    def __get_gdma_info(self, monitor_info, reg_info, core_id, engine_id=1):
        if self.is_dyn:
            if hasattr(reg_info, 'detailed_cmd'):
                _reg_info = self.gdma_parser.parse(reg_info.detailed_cmd)[0]
                return get_dma_info(monitor_info, _reg_info, core_id, engine_id)
            return get_dma_info_dyn(monitor_info, reg_info, engine_id)
        else:
            return get_dma_info(monitor_info, reg_info, core_id, engine_id)

    def __get_tiu_info(self, monitor_info, reg_info, core_id=None, engine_id=0):
        if self.is_dyn:
            if hasattr(reg_info, 'detailed_cmd'):
                _reg_info = self.bdc_parser.parse(reg_info.detailed_cmd)[0]
                return get_tiu_info(monitor_info, _reg_info)
            return get_tiu_info_dyn(monitor_info, reg_info)
        else:
            return get_tiu_info(monitor_info, reg_info)


if __name__ == "__main__":
    bmProfile = BMProfileParserPerfAI()
    bmProfile.parse("/workspace/workdir/prf/cdm_profile_data-0_core8")
    # bmProfile.to_txt('tmp')