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
import os, re, math
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
        self.cdma_cord_id = None

    def parse_cdma_cmd(self, file_list, mix_mode):
        print("Parsing...")
        self.cdma_pairs = [[] for _ in range(self.archlib.CDMA_NUM)]
        for infile in file_list:
            idx = eval(re.search(r'cdma\d*_(\d+)\.profile', infile).group(1))
            # if len(self.cdma_cmd[idx]) == 0:
            #     continue
            blocks = parse_data_blocks(infile)
            if blocks is None or blocks == []:
                continue
            monitor_cdma = []
            des_cdma = []
            blocks_factory = {
                BlockType.MONITOR_CDMA.value: (monitor_cdma, self.__parse_monitor_cdma),
                BlockType.BLOCK_DES_CDMA.value:
                (des_cdma, lambda l, raw_data: l.extend(self.cdma_parser.parse(raw_data))),
            }
            for block in blocks:
                item_list, item_func = blocks_factory.get(block.type.value, (0, lambda x, y: 0))
                item_func(item_list, block.content)
            self.cdma_pairs[idx] = self.make_pairs(self.cdma_cmd[idx],
                                                   monitor_cdma[0],
                                                   self.archlib.cdma_sys_code,
                                                   des_cdma,
                                                   is_cdma=True,
                                                   mix_mode=mix_mode)

    def parse_cmd(self, file_list):
        self.bdc_parser = self.archlib.BDCommandParser()
        self.gdma_parser = self.archlib.GDMACommandParser()
        self.cdma_parser = self.archlib.CDMACommandParser()
        print("Parsing...")
        self.cdma_cmd = [[] for _ in range(self.archlib.CDMA_NUM)]
        for infile in tqdm(file_list):
            blocks = parse_data_blocks(infile)
            if blocks is None:
                continue
            item = IterRecord()
            item.command_info = []
            item.dyn_extra = []
            item.des_bdc = []
            item.des_gdma = []
            item.des_sdma = []
            item.des_cdma = []
            blocks_factory = {
                BlockType.MONITOR_GDMA.value: (item.monitor_gdma, self.__parse_monitor_gdma),
                # # include sdma, vsdma pmu data
                BlockType.MONITOR_SDMA.value: (item.monitor_sdma, self.__parse_monitor_sdma),
                BlockType.MONITOR_BD.value: (item.monitor_bd, self.__parse_monitor_tiu),
                BlockType.DYN_DATA.value: (item.dyn_data, self.__parse_dyn_data),
                BlockType.COMMAND.value: (item.command_info, self.__parse_command_info),
                BlockType.DYN_EXTRA.value: (item.dyn_extra, self.__parse_dyn_extra),
                BlockType.BLOCK_DES_BDC.value:
                (item.des_bdc, lambda l, raw_data: l.extend(self.bdc_parser.parse(raw_data))),
                BlockType.BLOCK_DES_GDMA.value:
                (item.des_gdma, lambda l, raw_data: l.extend(self.gdma_parser.parse(raw_data))),
                BlockType.BLOCK_DES_SDMA.value:
                (item.des_sdma, lambda l, raw_data: l.extend(self.gdma_parser.parse(raw_data))),
            }
            for block in blocks:
                item_list, item_func = blocks_factory.get(block.type.value, (0, lambda x, y: 0))
                item_func(item_list, block.content)
            if item.command_info:
                self.__read_command_data(item)
            else:
                self.__read_dyn_command_data(item)

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
        self.cdma_cord_id = max(len(self.bd_pairs), len(self.gdma_pairs), len(self.sdma_pairs)) - 1
        if cdma_cmd:
            self.parse_cdma_cmd(cdma_cmd, self.is_dyn)

        self.omit_sys(self.bd_pairs, self.archlib.bd_sys_code)
        self.omit_sys(self.gdma_pairs, self.archlib.dma_sys_code)
        self.omit_sys(self.sdma_pairs, self.archlib.dma_sys_code)
        self.omit_sys(self.cdma_pairs, self.archlib.cdma_sys_code)
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
        cdma_file = os.path.join(self.out_dir, "cdmaRegInfo_{}.txt")
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
        g_idx = 0
        core_id = idx
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
            fn = self.__get_gdma_info
            arch = self.archlib.DMA_ARCH
            tag = "__CDMA_REG_INFO__\n"
            core_id = self.cdma_cord_id
        else:
            raise ValueError(f"Not support parse {self.archlib.EngineType(engine).name} now.")
        if len(pairs):
            with open(nfile.format(idx), fmode) as f:
                if new_file:
                    f.write("__CHIP_ARCH_ARGS__\n")
                    f.write("".join(f"\t{key}: {value}\n" for key, value in arch.items()))
                for p in pairs:
                    info, extra = fn(p["monitor"], p["cmd"], idx, engine.value)
                    info["Global Idx"] = g_idx
                    g_idx += 1
                    info["Core Id"] = core_id
                    f.write(tag)
                    f.write("".join(f"\t{key}: {value}\n" for key, value in info.items()))
                    if extra is not None:
                        f.write('{}:\n'.format(info["Function Type"]))
                        f.write("".join(f"\t{key}: {value}\n" for key, value in extra.items()))

    def __align_core_time(self):
        assert (len(self.profile_sync_points) == len(self.bd_pairs))
        for i, (bd_pair, gdma_pair,
                cycle) in enumerate(zip(self.bd_pairs, self.gdma_pairs, self.profile_sync_points)):
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
        for cdma in self.cdma_pairs:
            cycle = self.profile_sync_points[self.cdma_cord_id]
            delta_cyle = cycle - self.profile_sync_points[0]
            for j1 in cdma:
                j1["monitor"].inst_start_time = int(j1["monitor"].inst_start_time - delta_cyle)
                j1["monitor"].inst_end_time = int(j1["monitor"].inst_end_time - delta_cyle)

    def __shift_time(self):
        start_cycle = math.inf
        # start_cycle = self.gdma_pairs[0][0]["monitor"].inst_start_time
        for _, (bd_pair, gdma_pair,
                sdma_pair) in enumerate(zip(self.bd_pairs, self.gdma_pairs, self.sdma_pairs)):
            if bd_pair:
                start_cycle = min(bd_pair[0]["monitor"].inst_start_time, start_cycle)
            if gdma_pair:
                start_cycle = min(gdma_pair[0]["monitor"].inst_start_time, start_cycle)
            if sdma_pair:
                start_cycle = min(sdma_pair[0]["monitor"].inst_start_time, start_cycle)
        for _, (bd_pair, gdma_pair) in enumerate(zip(self.bd_pairs, self.gdma_pairs)):
            for j1 in itertools.chain(bd_pair, gdma_pair):
                j1["monitor"].inst_start_time = int(j1["monitor"].inst_start_time - start_cycle)
                j1["monitor"].inst_end_time = int(j1["monitor"].inst_end_time - start_cycle)
                assert (j1["monitor"].inst_start_time >= 0 and j1["monitor"].inst_end_time >= 0)
        for sdma_pair in self.sdma_pairs:
            for j1 in sdma_pair:
                j1["monitor"].inst_start_time = int(j1["monitor"].inst_start_time - start_cycle)
                j1["monitor"].inst_end_time = int(j1["monitor"].inst_end_time - start_cycle)
                assert (j1["monitor"].inst_start_time >= 0 and j1["monitor"].inst_end_time >= 0)
        for cdma_pair in self.cdma_pairs:
            for j1 in cdma_pair:
                j1["monitor"].inst_start_time = int(j1["monitor"].inst_start_time - start_cycle)
                j1["monitor"].inst_end_time = int(j1["monitor"].inst_end_time - start_cycle)

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
            current_time = c.inst_start_time + delta_time
            if current_time < last_time:
                delta_time += uint32_max  # uint32 max
            c.inst_start_time += delta_time
            c.inst_end_time += delta_time
            if c.inst_end_time < c.inst_start_time:
                # start not overflow but end does
                c.inst_end_time += uint32_max
            last_time = c.inst_end_time

    def __veryfy_cdma_time(self, data):
        last_st = 0
        last_et = 0
        delta_time = 0
        uint32_max = 4294967295
        for c in data:
            current_st = c.inst_start_time + delta_time
            current_et = c.inst_end_time + delta_time
            if current_st < last_st and current_et < last_et:
                delta_time += uint32_max  # uint32 max
            c.inst_start_time += delta_time
            c.inst_end_time += delta_time
            if c.inst_end_time < c.inst_start_time:
                # start not overflow but end does
                c.inst_end_time += uint32_max
            last_st = c.inst_start_time
            last_et = c.inst_end_time

    def __parse_monitor_tiu(self, monitor_tiu: List, raw_data):
        tmp = parse_monitor_bd(raw_data, self.archlib)
        self.__veryfy_cmd_id(tmp)
        self.__veryfy_time(tmp)
        monitor_tiu.append(tmp)

    def __parse_monitor_cdma(self, monitor_cdma: List, raw_data):
        tmp = parse_monitor_cdma(raw_data, self.archlib)
        self.__veryfy_cmd_id(tmp)
        self.__veryfy_cdma_time(tmp)
        self.__adjust_cmd_id(tmp)
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
        self.__adjust_cmd_id(tmp)
        monitor_sdma.append(tmp)

    def __parse_command_info(self, command_info: List, raw_data):
        command_info.append(self._BMProfileParser__parse_command_info(raw_data))

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
            bd_cmd = self.__base_read_command_data(cmd_info.bd_base, cmd_info.bd_offset,
                                                   self.archlib.EngineType.BD, core_num,
                                                   self.bdc_parser)
            gdma_cmd = self.__base_read_command_data(cmd_info.gdma_base, cmd_info.gdma_offset,
                                                     self.archlib.EngineType.GDMA, core_num,
                                                     self.gdma_parser)
            sdma_cmd = self.__base_read_command_data(
                cmd_info.sdma_base,
                cmd_info.sdma_offset,
                # TODO tpuv7-runtime/model-runtime/runtime/src/sgruntime_bmodel.cpp:576
                self.archlib.EngineType.VSDMA,
                core_num,
                self.gdma_parser)
            bd_pair = self.make_pairs([],
                                      item.monitor_bd[core_num],
                                      self.archlib.bd_sys_code,
                                      bd_cmd,
                                      mix_mode=False)
            gdma_pair = self.make_pairs([],
                                        item.monitor_gdma[core_num],
                                        self.archlib.dma_sys_code,
                                        gdma_cmd,
                                        mix_mode=False)
            sdma_pair = self.make_pairs([],
                                        item.monitor_sdma[core_num],
                                        self.archlib.dma_sys_code,
                                        sdma_cmd,
                                        mix_mode=False)

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
                    options = [(dp[i - 1][j] + 1, (i - 1, j)), (dp[i][j - 1] + 1, (i, j - 1)),
                               (dp[i - 1][j - 1] + 2, (i - 1, j - 1))]
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

    def __make_mix_pairs(self, cmd, monitor, sys_code, des_cmd):

        def get_sections(data):
            # Notice:
            # pmu: bd gdma sdma start idx == 0, cdma strat idx == 1
            # cmd: trat idx == 1
            sections = []
            slice = []
            first_idx = get_cmd_id(data[0]) if data else 1
            start_idx, last_idx = first_idx, first_idx
            for c in data:
                cmd_id = get_cmd_id(c)
                if cmd_id <= last_idx and slice:
                    sections.append((last_idx - start_idx + 1, slice))
                    slice = []
                    start_idx = cmd_id
                slice.append(c)
                last_idx = cmd_id
            if slice:
                sections.append((last_idx - start_idx + 1, slice))
            return sections

        _monitor = get_sections(monitor)
        _cmd = get_sections(cmd)

        # compatible code for tpu-train: pmu miss sys pair reason unknow
        # force align TODO remove this
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
                    _cmd[-1] = (max_len, _cmd[-1][1])
                    _monitor[-1] = (max_len, _monitor[-1][1])
        pairs = self.__match_sections(_monitor, _cmd)
        # des cmd
        _des_cmd = []
        if des_cmd:
            _des_cmd = get_sections(des_cmd)
            idx, rest_monitor = [], []
            for i, p in enumerate(pairs):
                if p[1] is None:
                    idx.append(i)
                    rest_monitor.append(_monitor[i])
            des_pairs = self.__match_sections(rest_monitor, _des_cmd)
            for i, p in enumerate(des_pairs):
                pairs[idx[i]] = p
        # print("+++++++++++++")
        # print("cmd", [c[0] for c in _cmd], [[get_cmd_id(i) for i in c[1]] for c in _cmd])
        # print("des_cmd", [c[0] for c in _des_cmd], [[get_cmd_id(i) for i in c[1]] for c in _des_cmd])
        # print("pmu", [m[0] for m in _monitor], [[i.inst_id for i in c[1]] for c in _monitor])
        # for m in _monitor[:9]:
        #     for i in m[1]:
        #         print(i.inst_id, i.inst_start_time, i.inst_end_time, i.thread_id if hasattr(i, "thread_id") else '')
        # print("pmu", [str(m[0]) for m in _monitor])
        return pairs

    def __make_pairs(self, cmd, monitor, sys_code, des_cmd, mix_mode):
        pairs = []
        if cmd == [] and des_cmd is None:
            for m in monitor:
                pairs.append({"monitor": m, "cmd": None})
            return pairs
        section = 0
        for p_monitor, p_cmd in self.__make_mix_pairs(cmd, monitor, sys_code, des_cmd):
            # print(section, [i.inst_id for i in p_monitor], p_monitor is not None and p_cmd is not None)
            section += 1
            if p_monitor is None:
                continue
            if p_cmd is None:
                if mix_mode:  # mix mode for tpudnn, otherwise bmodel
                    for m in p_monitor:
                        pairs.append({"monitor": m, "cmd": None})
                continue
            # len(cmd) >= len(pmu) cause pmu will drop some data
            m_start_idx = p_monitor[0].inst_id
            for m in p_monitor:
                m_idx = m.inst_id - m_start_idx
                if m_idx <= len(p_cmd):
                    pairs.append({"monitor": m, "cmd": p_cmd[m_idx]})
        return pairs

    def correct_ids(self, data):
        data.sort(key=lambda x: x["start_time"])
        next_id = 1
        id_mapping = {}

        for record in data:
            original_id = record["id"]
            if original_id not in id_mapping:
                id_mapping[original_id] = next_id
                next_id += 1
            record["id"] = id_mapping[original_id]

        for i in range(len(data)):
            if data[i]["id"] == id_mapping[8] and data[i]["start_time"] > data[i - 1]["start_time"]:
                data[i - 1], data[i] = data[i], data[i - 1]

        return data

    @staticmethod
    def __adjust_cmd_id(monitor):
        pre = None
        for i, m in enumerate(monitor):
            # if pre and m.inst_id < pre.inst_id and pre.thread_id == 1:
            # compatiable code for cdma pmu receive thread_id == 0
            if pre and m.inst_id < pre.inst_id and (pre.thread_id == 1 \
                                                    or (pre.inst_id - m.inst_id == 1 and m.inst_id)):
                monitor[i], monitor[i - 1] = monitor[i - 1], monitor[i]
            pre = m

    def __rm_tx_wait_points(self, monitor):
        # tx_wait (wait, nop)
        for i in range(self.archlib.profile_init_cmd_num):
            _m = monitor.pop(0)
            if i == 0:
                start_time = _m.inst_end_time
        # align internal engine
        if len(self.sdma_pairs):
            sdma_send_time = self.sdma_pairs[self.cdma_cord_id].pop(0)["monitor"].inst_end_time
            delta_cycle = start_time - sdma_send_time
            if delta_cycle:
                for j1 in monitor:
                    j1.inst_start_time = j1.inst_start_time - delta_cycle
                    j1.inst_end_time = j1.inst_end_time - delta_cycle

    def __rm_sync_points(self, monitor, record_time_stamp=False):
        # sync points (send, wait)
        for _ in range(self.archlib.profile_init_cmd_num):
            m = monitor.pop(0)
            time_point = m.inst_end_time
        if record_time_stamp:
            # we use bd to set base time stamp
            self.profile_sync_points.append(time_point)
        else:
            # align internal engine, not sure if is needed
            delta_cycle = time_point - self.profile_sync_points[-1]
            if delta_cycle:
                for j1 in monitor:
                    j1.inst_start_time = j1.inst_start_time - delta_cycle
                    j1.inst_end_time = j1.inst_end_time - delta_cycle

    def omit_sys(self, pairs, sys_code):
        for p in pairs:
            self.__omit_sys(p, sys_code, end=True)
            self.__omit_sys(p, sys_code, end=False)  # omit sys at begin

    def __omit_sys(self, pairs, sys_code, end):
        extra_sys = []
        part = []
        idxs = range(len(pairs))
        if end:
            idxs = idxs[::-1]
        for i in idxs:
            if pairs[i]["cmd"] is None:
                if self.is_dyn:
                    break
                else:  # for bmodel
                    extra_sys.append(i)
                    continue
            tsk_type, _ = self.__get_cmd_type(pairs[i]["cmd"])
            if tsk_type != sys_code:
                break
            else:
                part.append(i)
            if end and pairs[i]["monitor"].inst_id == 0:
                extra_sys.extend(part)
                part = []
        if not end:
            extra_sys.extend(part)
            extra_sys = extra_sys[::-1]
        for i in extra_sys:
            pairs.pop(i)

    def __compatiable_make_pairs(self, cmd, monitor, sys_code, des_cmd, mix_mode):
        pairs = []
        if len(cmd) == len(monitor):
            for c, m in zip(cmd, monitor):
                pairs.append({"monitor": m, "cmd": c})
        else:
            pairs = self.__make_pairs(cmd, monitor, sys_code, des_cmd, mix_mode)
        return pairs

    def make_pairs(self, cmd, monitor, sys_code, des_cmd=None, is_cdma=False, mix_mode=True):
        offset = self.archlib.profile_init_cmd_num
        if is_cdma:
            self.__rm_tx_wait_points(monitor)
            # tmp compatiable code cause pio mode pmu inst_id is not correct
            pairs = self.__compatiable_make_pairs(cmd[offset:], monitor, sys_code, des_cmd,
                                                  mix_mode)  # todo support des
            return pairs
        else:
            self.__rm_sync_points(monitor, sys_code == self.archlib.bd_sys_code)
        pairs = self.__make_pairs(cmd[offset:], monitor, sys_code, des_cmd, mix_mode)
        return pairs

    def __read_dyn_command_data(self, item):
        # just for tpudnn
        # dyn cmd include single core cmd per cdm.profile
        gdma_cmd = []
        sdma_cmd = []
        bd_cmd = []
        if item.dyn_data:
            dyn_data = item.dyn_data
            assert (len(dyn_data) > 0)
            core_id = len(self.bd_pairs)
            for i, d in enumerate(dyn_data[1:]):  # skip init record
                d.core_id = core_id
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
        bd_pair = self.make_pairs(bd_cmd, item.monitor_bd[0], self.archlib.bd_sys_code,
                                  item.des_bdc)
        gdma_pair = self.make_pairs(gdma_cmd, item.monitor_gdma[0], self.archlib.dma_sys_code,
                                    item.des_gdma)
        sdma_pair = self.make_pairs(sdma_cmd, item.monitor_sdma[0], self.archlib.dma_sys_code,
                                    item.des_sdma)
        self.bd_pairs.append(bd_pair)
        self.gdma_pairs.append(gdma_pair)
        self.sdma_pairs.append(sdma_pair)

    def __base_read_command_data(self, base, offset, engine_type, core_num, command_parser):
        basename = "cmd_%x_%d_%d.dat"
        command_filename = os.path.join(self.in_dir, basename % (base, core_num, engine_type.value))
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
            for i in command_list[::-1][1:]:
                if i.op_name == end_name:
                    num += 1
        for _ in range(num):
            command_list.pop()
        return command_list

    def __get_gdma_info(self, monitor_info, reg_info, core_id, engine_id=1):
        if reg_info is None:
            return get_dma_info_dyn(monitor_info, reg_info, engine_id)
        if self.is_dyn and hasattr(reg_info, "extra_info"):
            if hasattr(reg_info, 'detailed_cmd') and engine_id != 4:
                _reg_info = self.gdma_parser.parse(reg_info.detailed_cmd)[0]
                return get_dma_info(monitor_info, _reg_info, core_id, engine_id)
            return get_dma_info_dyn(monitor_info, reg_info, engine_id)
        else:
            return get_dma_info(monitor_info, reg_info, core_id, engine_id)

    def __get_tiu_info(self, monitor_info, reg_info, core_id=None, engine_id=0):
        if reg_info is None:
            return get_tiu_info_dyn(monitor_info, reg_info)
        if self.is_dyn and hasattr(reg_info, "extra_info"):
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
