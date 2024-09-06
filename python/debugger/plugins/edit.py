# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Dict
import os

from rich.progress import (
    Progress as Progressbar,
    TimeRemainingColumn,
    TimeElapsedColumn,
    TaskProgressColumn,
    TextColumn,
    ProgressColumn,
    BarColumn,
)

import pandas as pd

from ..final_mlir import CMD, FinalMlirIndex, TLValue
from .common import FinalMlirIndexPlugin
from ..target_common.op_support import BaseTpuCmd
from ..target_common import Target, CMDType
from ..tdb_support import (
    TdbCmdBackend,
    TdbPlugin,
    Displays,
    TdbStatus,
    TdbPluginCmd,
    complete_file,
)
from copy import deepcopy
from ..disassembler import BModel
import ctypes
from ..atomic_dialect import MlirModule, atomic_context

bm1688 = {
    # firmware_base/src/atomic/atomic_sys_gen_cmd.c:7
    # high = 0
    # low = 0x3FE0000020000
    "tiu_end": b"\x00\x00\x02\x00\x00\xfe\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00",
    # firmware_base/src/atomic/atomic_dma_gen_cmd.c:1551
    # high = 0
    # low = 0x600000028
    "dma_end": b"(\x00\x00\x00\x06\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
}


class BmodelEditor(TdbPlugin, TdbPluginCmd):
    name = "edit-bmodel"
    func_names = ["edit"]

    def after_load(self, tdb: TdbCmdBackend):
        if tdb.context.device != Target.BM1688:
            tdb.error("Currently only support BM1688")
            return

        self.version = 0
        self.bmodel = BModel(tdb.bmodel_file)
        self.atomic_mlir = MlirModule.from_context(self.bmodel, tdb.context)
        self.cmditer = self.atomic_mlir.create_cmdlist()
        tdb.message("copy bmodel, edit enabled.")

    def create_tiu_end(self, cmd_id=0, core_id=0):
        buf = bytearray(bm1688["tiu_end"])
        sys_end = self.tdb.context.decoder.decode_tiu_cmd(
            buf, cmd_id=cmd_id, core_id=core_id
        )
        sys_end.reg.rsvd1 = 1
        return sys_end

    def create_dma_end(self, cmd_id=0, core_id=0):
        buf = bytearray(bm1688["dma_end"])
        sys_end = self.tdb.context.decoder.decode_dma_cmd(
            buf, cmd_id=cmd_id, core_id=core_id
        )
        sys_end.reg.reserved0 = 1
        return sys_end

    def do_cut(self, args):
        """insert command"""
        if self.tdb.context.device != Target.BM1688:
            self.tdb.error("Currently only support BM1688")
            return

        index: FinalMlirIndexPlugin = self.tdb.get_plugin(FinalMlirIndexPlugin)
        loc = index.get_loc_by_point()
        self.tdb.message(f"cut bmodel after {loc}")

        indexs = self.tdb.index_df.loc[self.tdb.index_df.loc_index == loc.loc_index,"executed_id"].tolist()

        index = max(indexs) + 1

        # find tiu / bdc command
        hit_dic = {}
        while index < len(self.cmditer):
            cmd = self.cmditer[index]
            if (cmd.cmd_type, cmd.core_id) not in hit_dic:
                if cmd.cmd_type == CMDType.tiu:
                    self.tdb.message(cmd)
                    cmd = self.create_tiu_end(cmd_id=cmd.cmd_id)
                    old_buf = bytes(cmd.cmd.buf[: len(cmd.reg.buf)])
                    cmd.cmd.buf[: len(cmd.reg.buf)] = bytes(cmd.reg)
                elif cmd.cmd_type == CMDType.dma:
                    self.tdb.message(cmd)
                    cmd = self.create_dma_end(cmd_id=cmd.cmd_id)
                    old_buf = bytes(cmd.cmd.buf[: len(cmd.reg.buf)])
                    cmd.cmd.buf[: len(cmd.reg.buf)] = bytes(cmd.reg)
                else:
                    index += 1
                    continue
                hit_dic[(cmd.cmd_type, cmd.core_id)] = (
                    bytes(cmd.cmd.buf[: len(cmd.reg.buf)]),
                    old_buf,
                )
            index += 1

        fn = f"v{self.version}.bmodel"
        while os.path.exists(fn):
            self.version += 1

        self.bmodel.serialize(fn)
        self.tdb.message(f"serialized at {os.path.abspath(fn)}")
