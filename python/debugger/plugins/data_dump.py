# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import numpy as np
from numpy.lib import format
from typing import Dict, List
import os
import zipfile
import pandas as pd

from numpy_helper.npz_compare import TensorCompare as _TensorCompare
from ..target_common import MType, ValueRef
from ..tdb_support import (
    TdbCmdBackend,
    TdbPlugin,
    TdbPluginCmd,
)

from .common import FinalMlirIndexPlugin, ValueView
from ..target_1688.context import BM1688Context
from ..target_1690.context import BM1690Context
from ..target_2380.context import SG2380Context

class IncNpzFile2:
    def __init__(self, file: str):
        """
        :param file: the ``npz`` file to write
        :param mode: must be one of {'x', 'w', 'a'}. See
               https://docs.python.org/3/library/zipfile.html for detail
        """
        self.fn = file
        self.zip = zipfile.ZipFile(file, mode="a", compression=zipfile.ZIP_DEFLATED)

    def __contains__(self, key: str) -> None:
        return key in self.zip.namelist()

    def __setitem__(self, key: str, data) -> None:
        if key in self:
            return

        kwargs = {
            "mode": "w",
            "force_zip64": True,
        }
        if self.zip is None or self.zip.fp is None:
            self.zip = zipfile.ZipFile(
                self.fn, mode="a", compression=zipfile.ZIP_DEFLATED
            )

        with self.zip.open(key, **kwargs) as fid:
            val = np.asanyarray(data)
            format.write_array(fid, val, allow_pickle=True)

    def __getitem__(self, key: str):
        self.zip.close()
        return np.load(self.fn, allow_pickle=True)[key]

    def close(self):
        if self.zip is not None:
            self.zip.close()
            self.zip = None

    def return_npz(self):
        self.zip.close()
        return np.load(self.fn, allow_pickle=True)

class DataDump(TdbPlugin, TdbPluginCmd):
    """
    DataDump
    """

    name = "data-dump"

    def __init__(self, tdb: TdbCmdBackend) -> None:
        super().__init__(tdb)
        self.index: FinalMlirIndexPlugin = tdb.get_plugin(FinalMlirIndexPlugin)
        self.out_tensors = None
        self.buf_tensors = dict()
        self.output = "tdb_outputs.npz"
        self.excepts = set()

    @property
    def enabled(self):
        return self.index.enabled

    def after_load(self, tdb: TdbCmdBackend):
        file = self.output
        if os.path.exists(file):
            os.remove(file)
            print(f"remove exist {file}")
        self.out_tensors = IncNpzFile2(file)

    def dump_all(self, tdb: TdbCmdBackend, is_operand):
        point_index = tdb.cmd_point
        values = None

        if is_operand:
            point_index += 1
            values = tdb.index_df.loc[
                tdb.index_df["executed_id"] == point_index, "operands"
            ].tolist()
        else:
            values = tdb.index_df.loc[
                tdb.index_df["executed_id"] == point_index, "results"
            ].tolist()

        if values:
            values = values[0]

        for value_view in values:
            if is_operand != value_view.is_operand:
                continue

            if value_view.value is None:  # top.None
                continue

            self.__dump(point_index - 1, is_operand, value_view)

    def __dump(self, point_index: int, is_operand, value_view: ValueView):
        value = value_view.value
        if value.name in self.excepts:
            return
        if value.name in self.out_tensors:
            return

        # this hack should only enable in multicore
        if not is_operand:
            if value_view.file_line in self.index.tdb.global_layer_line:
                self.index.tdb.global_layer_line[value_view.file_line] -= 1
                if self.index.tdb.global_layer_line[value_view.file_line] != 0:
                    return

        context = self.tdb.context
        memref = value.get_memref(context)
        if not context.memory.using_cmodel:
            if memref.mtype != MType.G and memref.mtype != MType.R:
                return

        cmd = self.tdb.cmditer[point_index]
        raw_data = context.memory.get_data(ValueRef(memref, core_id=cmd.core_id))
        new_data = (raw_data.astype(np.float32) - value.zero_point) * value.scale

        value_dict = value.__dict__
        if value_dict['slice'] != '[...]':
            import re
            pattern = re.compile(r'(?<=[\<|x])\d+')   # 查找数字
            glb_shape = [int(x) for x in pattern.findall(value_dict['reshape'])]
            assert glb_shape[2] == 1, "not support 3d shape"
            glb_shape.pop(2)
            pattern = re.compile(r'[0-9]+')
            slice_start_end = [int(x) for x in pattern.findall(value_dict['slice'])]
            start = slice_start_end[::2]
            end = slice_start_end[1::2]
            start.pop(2)
            end.pop(2)

            if start != [0]*len(start) or end != glb_shape:
                if value.name not in self.buf_tensors:
                    self.buf_tensors[value.name] = np.zeros(glb_shape, dtype=new_data.dtype)
                self.buf_tensors[value.name][start[0]:end[0],
                                            start[1]:end[1],
                                            start[2]:end[2],
                                            start[3]:end[3]] = new_data
                if end != glb_shape: return
                new_data = self.buf_tensors[value.name]
                self.buf_tensors.pop(value.name)
        self.out_tensors[value.name] = new_data

    def after_step(self, tdb: TdbCmdBackend):
        if self.enabled:
            self.dump_all(tdb, False)

    def after_stop(self, tdb: TdbCmdBackend):
        self.out_tensors.close()
        print("save", self.output)
        return super().after_stop(tdb)
