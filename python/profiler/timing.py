# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import ctypes as ct
from collections import namedtuple
import struct as st
import type_def
import os, itertools


class BlockTimelineRecord:
    def _get_ct_items(self, buffer, ctype=ct.c_void_p):
        tlen = ct.sizeof(ctype)
        for i in range(0, len(buffer), tlen):
            obj = ctype()
            ct.memmove(ct.addressof(obj), buffer[i : i + tlen], tlen)
            yield obj

    def _adapter(self, fun=None, ctype=None):
        from functools import partial

        class adapter:
            __slots__ = ("_fun_", "data")

            def __init__(self, fun):
                self._fun_ = fun
                self.data = []

            def append(self, buffer):
                self.data.extend(self._fun_(buffer))

            def __iter__(self):
                return iter(self.data)

            def __getitem__(self, index):
                return self.data[index]

            def __len__(self):
                return len(self.data)

            def __repr__(self) -> str:
                return str(self.data)

        if ctype:
            if fun:
                return adapter(partial(fun, ctype))
            return adapter(partial(self._get_ct_items, ctype=ctype))
        return adapter(fun)

    def _command_info(self, raw_data):
        header_len = 8 * 4 + 4
        gdma_base, gdma_offset, bd_base, bd_offset, group_num = st.unpack(
            "QQQQI", raw_data[0:header_len]
        )
        group = []
        for i in range(group_num):
            group.append(
                st.unpack("II", raw_data[header_len + i * 8 : header_len + (i + 1) * 8])
            )
        CommandInfo = namedtuple(
            "CommandInfo", "gdma_base gdma_offset bd_base bd_offset group_num group"
        )
        return CommandInfo(gdma_base, gdma_offset, bd_base, bd_offset, group_num, group)

    def _read_data_blocks(self, filename):
        if not os.path.isfile(filename):
            return []
        BlockItem = namedtuple("BlockItem", ["type", "content"])
        with open(filename, "rb") as f:
            while True:
                header_data = f.read(8)
                if len(header_data) != 8:
                    break
                block_type, block_len = st.unpack("II", header_data)
                block_type = type_def.BlockType(block_type)
                block_content = f.read(block_len)
                assert len(block_content) == block_len
                yield BlockItem(block_type, block_content)

    def __init__(self, file):
        self.summary = self._adapter(ctype=type_def.IterSummary)
        self.mcu_data = self._adapter(ctype=type_def.MCURecord)
        self.mcu_extra = self._adapter(ctype=type_def.MCUExtraType)
        self.tiu = self._adapter(ctype=type_def.TIUProfile)
        self.dma = self._adapter(ctype=type_def.DMAProfile)
        self.command_info = self._adapter(fun=self._command_info)
        attr = {
            type_def.BlockType.SUMMARY: self.summary,
            type_def.BlockType.MCU_DATA: self.mcu_data,
            type_def.BlockType.MCU_EXTRA: self.mcu_extra,
            type_def.BlockType.MONITOR_TIU: self.tiu,
            type_def.BlockType.MONITOR_DMA: self.dma,
            type_def.BlockType.COMMAND: self.command_info,
        }
        for record in self._read_data_blocks(file):
            attr[record.type].append(record.content)
        self.summary = self.summary[0]

    def __repr__(self) -> str:
        pmu_info = {
            "TIU": len(self.tiu),
            "DMA": len(self.dma),
            "MCU": len(self.mcu_data),
        }
        pmu_info.update(self.summary)
        return "\n".join(f"{k:>15}: {v}" for k, v in pmu_info.items())


def get_hw_timming(in_dir):
    for _iter in itertools.count(0, 1):
        block_filename = f"iter{_iter}.profile"
        block_filename = os.path.join(in_dir, block_filename)
        if os.path.isfile(block_filename):
            yield BlockTimelineRecord(block_filename)
        else:
            break


# sample -> timing
