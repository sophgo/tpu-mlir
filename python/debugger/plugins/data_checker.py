# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
import pickle
import collections
import math
import numpy as np
from numpy.lib import format
from typing import Tuple, Dict, List, Union
import os
import json
import zipfile
import pandas as pd

from rich import get_console
from rich.console import Group
from rich import box
from rich.json import JSON
from rich.table import Table
from rich.panel import Panel

from typing import Union, Tuple
from debugger.target_common.op_support import Scalar
from ..final_mlir import Pickled_Value, TLValue
from numpy_helper.npz_compare import TensorCompare as _TensorCompare
from ..target_common import MType, ValueRef
from ..tdb_support import (
    BreakpointStop,
    TdbCmdBackend,
    TdbPlugin,
    TdbPluginCmd,
    codelike_format,
)

# from ..final_mlir import Value as MlirValue
from dataclasses import dataclass
from .common import FinalMlirIndexPlugin, ValueView
from enum import Enum


class IncNpzFile:

    def __init__(self, file: str):
        """
        :param file: the ``npz`` file to write
        :param mode: must be one of {'x', 'w', 'a'}. See
               https://docs.python.org/3/library/zipfile.html for detail
        """
        self.fn = file
        self.zip = zipfile.ZipFile(file, mode="a", compression=zipfile.ZIP_DEFLATED)
        self.keys = set()

    def __setitem__(self, key: str, data) -> None:
        if key in self.keys:
            return

        self.keys.add(key)
        kwargs = {
            "mode": "w",
            "force_zip64": True,
        }
        if self.zip is None or self.zip.fp is None:
            self.zip = zipfile.ZipFile(self.fn, mode="a", compression=zipfile.ZIP_DEFLATED)

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


class DumpMode(Enum):
    NEVER = 0
    FAILED = 1
    ALL = 2
    COMB = 3
    TPULANG = 4
    COMB_ALL = 5


class CmpState(Enum):
    Pass = 0
    Fail = 1
    Ignore = 2
    Middle = 3
    Unknown = 9

    @property
    def flag(self):
        return StateFlags[self]


StateFlags = {
    CmpState.Pass: "[green]√[/]",
    CmpState.Fail: "[red]x[/]",
    CmpState.Middle: "-",
    CmpState.Unknown: "?",
    CmpState.Ignore: "[gray]-[/]",
}


@dataclass()
class ComparedResult:
    value_view: ValueView
    cmp: Tuple
    zero_point: int = 0
    scale: int = 1
    # actual: np.ndarray = None
    # desired: np.ndarray = None
    msg: str = ""

    @property
    def state(self) -> CmpState:
        if self.msg == "ignore":
            return CmpState.Ignore  # ignored by excepts
        if self.cmp is None:
            return CmpState.Unknown
        elif self.cmp[0]:
            return CmpState.Pass
        return CmpState.Fail


class TensorCompare(_TensorCompare):
    """
    Check, Record
    """

    def diff_details(self, d1, d2, _):
        # override TensorCompare
        _d1 = d1.ravel()
        _d2 = d2.ravel()
        diff = np.abs(_d1 - _d2)
        k = min(_d1.size, 10)
        idx = np.argpartition(diff, -k)[-k:]
        idx = [x[0] for x in sorted(zip(idx, diff[idx]), key=lambda x: x[1], reverse=True)]
        return {"x": _d1[idx], "y": _d2[idx]}

    def close(self, actual, desired):
        return self.compare(
            actual,
            desired,
            verbose=2,
            int8_tensor_close=True,
        )[0]

    def assert_allclose(self, actual: np.ndarray, desired: np.ndarray, dump_mode: DumpMode):
        from functools import partial
        from numpy.core import array_repr

        cmp_res = self.compare(actual, desired, verbose=2, int8_tensor_close=True)
        state, _, _, metric, details = cmp_res
        msg = ""
        if not state:
            metric = metric
            header = ("Not equal to tolerance " + f"cos={self.cosine_similarity_tol}" +
                      f", euc={self.euclidean_similarity_tol}")

            remarks = [
                f"cosine similarity: {metric['cosine']:.6f}",
                f"euclidean similarity: {metric['euclid']:.6f}",
            ]
            msg = ["\n" + header, "", "\n".join(remarks)]
            r_func = partial(
                array_repr,
                precision=6,
                max_line_width=get_console().size.width,
            )
            msg.append("top10_diff:")
            msg.extend(f" {n}: {r_func(r.astype(float))[6:-1]}" for n, r in details.items())
            msg.extend((
                "0:10_data:",
                f" act: {r_func(actual.ravel()[:10].astype(float))[6:-1]}",
                f" ref: {r_func(desired.ravel()[:10].astype(float))[6:-1]}",
            ))
            try:
                np.testing.assert_allclose(actual, desired, rtol=1e-3, atol=1e-1, verbose=False)
            except AssertionError as e:
                msg.append(str(e))

            msg = "\n".join(msg)
        elif dump_mode == DumpMode.ALL:
            header = ("Succeed comparasion: " + f"cos={self.cosine_similarity_tol}" +
                      f", euc={self.euclidean_similarity_tol}")
            msg = ["\n" + header, ""]
            r_func = partial(
                array_repr,
                precision=6,
                max_line_width=get_console().size.width,
            )
            msg.extend((
                "0:10_data:",
                f" act: {r_func(actual.ravel()[:10].astype(float))[6:-1]}",
                f" ref: {r_func(desired.ravel()[:10].astype(float))[6:-1]}",
            ))
            msg = "\n".join(msg)

        return cmp_res, msg


class DataCheck(TdbPlugin, TdbPluginCmd):
    """
    DataCheck
    """

    name = "data-check"
    func_names = ["check"]
    soc_values_in = collections.defaultdict(list)
    soc_values_out = []

    def __init__(self, tdb: TdbCmdBackend) -> None:
        super().__init__(tdb)

        self.ref_data = []
        self.desire_op = []
        self.ref_data_from_inference = {}
        self.mlir_ref = False
        for ref_fn in tdb.reference_data_fns:
            if ref_fn.endswith(".mlir"):
                self.ops = self.collect_op_name_dict(ref_fn, tdb)
                self.mlir_ref = True
            else:
                self.ref_data.append(np.load(ref_fn))
                self.mlir_ref = False or self.mlir_ref

        self.failed_summary_info = []
        self.watch = None
        self.tc = TensorCompare(
            cosine_similarity_tol=0.99,
            euclidean_similarity_tol=0.99,
            signal_to_quantization_noise_tol=float("-inf"),
        )

        # loc_index ->
        self.reports: Dict[int, List[ComparedResult]] = {}
        self.index_record: Dict[int, List[ComparedResult]] = {}
        self.break_when_fail = False

        self.failed_results_fn = "failed_bmodel_outputs.npz"
        self.excepts = set()
        self.dump_mode = DumpMode.FAILED
        self.out_fixed = False
        self.is_soc = False
        self.skip_check = False
        self.bmodel_output = {}

    def collect_op_name_dict(self, ref_fn, tdb: TdbCmdBackend):
        if tdb.cache_mode in {"generate", "online"}:
            from utils.mlir_parser import MlirParser
            ops = MlirParser(ref_fn).collect_op_name_dict()
            if tdb.cache_mode == "generate":
                tdb.save_pickle(f"{ref_fn}.tdb_cache.ops.pickle", ops)
        elif tdb.cache_mode == "offline":
            ops = tdb.load_pickle(f"{ref_fn}.tdb_cache.ops.pickle")
        return ops

    def set_tol(self, cosine_similarity_tol=0.99, euclidean_similarity_tol=0.9):
        self.tc = TensorCompare(
            cosine_similarity_tol=cosine_similarity_tol,
            euclidean_similarity_tol=euclidean_similarity_tol,
            signal_to_quantization_noise_tol=float("-inf"),
        )

    @property
    def enabled(self):
        return self.index.enabled

    def after_load(self, tdb: TdbCmdBackend):
        self.index: FinalMlirIndexPlugin = tdb.get_plugin(FinalMlirIndexPlugin)
        self.reports.clear()
        self.index_record.clear()

        self._failed_tensor = None
        self._failed_tensor_meta = {}
        self.tdb.message(f"dump mode = {self.dump_mode}")

    @property
    def failed_tensor(self):
        if self._failed_tensor is None:
            file = self.failed_results_fn
            if os.path.exists(file):
                os.remove(file)
                self.tdb.message(f"remove exist {file}")
            self._failed_tensor = IncNpzFile(file)
        return self._failed_tensor

    def do_dump_names(self, arg=None):
        """
        dump failed comparison data into npz file
        """
        failed_names = set()
        for k, v in self.reports.items():
            for cmp_res in v:
                if cmp_res.state == CmpState.Fail:
                    failed_names.add(cmp_res.value_view.value.name)

        if len(failed_names) > 0:
            failed_list_fn = f"{arg}.txt"
            self.tdb.save_txt(failed_list_fn, list(failed_names), name="bmodel_failed_names")

    def do_dump_mode(self, arg: str):
        mode = DumpMode.__members__.get(arg, None)
        if mode is not None:
            self.dump_mode = mode
            self.tdb.message(f"enable {mode}")
        else:
            self.tdb.message(f"Unknown dump mode {arg}, support {DumpMode._member_names_}")

    def reduce_summary(self):
        summary: Dict[int, List[List[str]]] = {}
        for loc_index, v in self.reports.items():
            lino = self.index.final_mlir.loc[loc_index].file_line

            loc_summary = summary.setdefault(lino, [])
            current = ([], [])
            for vv in v:
                if vv.value_view.is_operand:
                    current[0].append(vv.state.flag)
                else:
                    current[1].append(vv.state.flag)
            loc_summary.append(current)

        summary_lis = []
        for k, v in summary.items():
            loc_summary = [f"{''.join(vv[0])},{''.join(vv[1])}" for vv in v]
            loc_summary = "|".join(loc_summary)
            summary_lis.append(f"({k}:{loc_summary})")
        return " ".join(summary_lis)

    def table_summary(self):
        summary: List[Tuple[int, List[str], List[str]]] = []
        for loc_index, v in self.reports.items():
            lino = self.index.final_mlir.loc[loc_index].file_line

            loc_summary = ([], [])
            for vv in v:
                if vv.value_view.is_operand:
                    loc_summary[0].append(vv.state.flag)
                else:
                    loc_summary[1].append(vv.state.flag)

            loc_summary = f"({lino}{''.join(loc_summary[0])}|{''.join(loc_summary[1])})"
            summary.append(loc_summary)
        return " ".join(summary)

    def failed_summary(self):
        summary: List[Tuple[int, List[str], List[str]]] = []
        skip = 0
        for loc_index, v in self.reports.items():
            lino = self.index.final_mlir.loc[loc_index].file_line

            loc_summary = ([], [])
            if all([vv.state != CmpState.Fail for vv in v]):
                skip += 1
                continue

            for vv in v:
                if vv.value_view.is_operand:
                    loc_summary[0].append(vv.state.flag)
                else:
                    loc_summary[1].append(vv.state.flag)

            loc_summary = f"({lino}{''.join(loc_summary[0])}|{''.join(loc_summary[1])})"
            summary.append(loc_summary)
        self.tdb.message(f"summary failed skip {skip} cmds")
        return " ".join(summary)

    def do_summary(self, arg):
        if arg == "" or arg not in {"reduce", "table", "failed"}:
            arg = "table"

        if arg == "reduce":
            self.tdb.message(self.reduce_summary())
        elif arg == "table":
            self.tdb.message(self.table_summary())
        elif arg == "failed":
            self.tdb.message(self.failed_summary())
        else:
            self.tdb.error(f"not support summary mode {arg}")

    def dump_dataframe(self, path="."):
        # debug options
        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.expand_frame_repr", False)

        data_str = self.index.index_df.to_string()
        with open(os.path.join(path, "dumped_dataframe.txt"), "w+", encoding="utf-8") as f:
            f.write(data_str)

        # reset debug option
        pd.reset_option("display.max_rows")
        pd.reset_option("display.max_columns")
        pd.reset_option("display.expand_frame_repr")

    def do_data(self, arg: str):
        if arg == "":
            self.do_summary("")
            return

        try:
            parg = list(map(int, arg.split(" ")))
            select_index = -1
            if len(parg) == 1:
                lino = parg[0]
            else:
                lino, select_index = parg

            if select_index == -1:
                table = Table(f"{arg}", show_lines=True)
                for col in [
                        "loc_index",
                        "owner",
                        "value name",
                        "value type",
                        "compare",
                        "index",
                ]:
                    table.add_column(col, no_wrap=True)
                for index, v in enumerate(self.index_record[lino]):
                    loc = self.index.final_mlir.loc[v.value_view.loc_index]
                    table.add_row(
                        f"{index}",
                        f"{v.value_view.loc_index}",
                        loc.opcode,
                        f"{v.value_view.value.name}",
                        f"{'operand' if v.value_view.is_operand else 'result'}",
                        f"{v.state.flag}",
                        f"{index}",
                    )
                self.tdb.message(table)
            else:
                vv = self.index_record[lino][select_index]
                loc = self.index.final_mlir.loc[vv.value_view.loc_index]
                table = Table(
                    title="value-Info",
                    show_lines=True,
                    show_header=False,
                    box=box.HORIZONTALS,
                )
                table.add_row("opcode", loc.opcode)
                table.add_row("core_id", str(loc.core_id))
                table.add_row("value", JSON(json.dumps(vv.value_view.value.raw_dict)))
                table.add_row("tiu_dma_id (before codegen)", str(loc.tiu_dma_id_before))
                table.add_row("tiu_dma_id (after codegen)", str(loc.tiu_dma_id_after))
                table.add_row("compare", vv.state.flag)
                table.add_row(
                    "value type",
                    f"{'operand' if vv.value_view.is_operand else 'result'}",
                )
                if vv.state == CmpState.Fail or self.dump_mode == DumpMode.ALL:
                    err_msg = Panel(vv.msg, title="data-error", style="red")
                    asm_codes = Panel(
                        codelike_format(self.tdb.get_op_context(2, 2, vv.value_view.cmd_point), 2),
                        title="asm",
                    )
                    msg = Panel(Group(table, err_msg, asm_codes))
                    self.tdb.message(msg)
                else:
                    self.tdb.message(Panel(table))

        except Exception as e:
            self.tdb.error(e)

    def complete_data(self, text="", line="", begidx=0, endidx=0) -> List[str]:
        return list(filter(lambda x: x.startswith(text), map(str, self.index_record.keys())))

    def complete_summary(self, text="", line="", begidx=0, endidx=0) -> List[str]:
        cand = ["reduce", "table", "failed"]
        return [i for i in cand if i.startswith(text)]

    def do_ignore_failed(self, arg):
        if arg == "":
            self.break_when_fail = not self.break_when_fail
        else:
            try:
                self.break_when_fail = bool(eval(arg))
            except Exception as e:
                self.tdb.error(e)
        self.tdb.message(f"ignore failed = {self.break_when_fail}")

    def complete_ignore_failed(self, text: str, state: int) -> Union[List[str], None]:
        return ["True", "False"]

    def format_data(self, operand: TLValue, arr: np.ndarray):
        reshape = operand.reshape
        if arr is None:
            arr = np.zeros(operand.shape)

        if reshape:
            reshape = eval(reshape[1:-1].replace("x", ","))  # '1,3,1,192,1024'
            arr = arr.reshape(reshape)

        _slice = operand.slice
        data = eval(f"arr{_slice}")  # type: np.ndarray
        # The data in HW has a transposed collapsed shape.
        # To align the Bmodel with TPU.mlir, we need to transpose the reference data.
        if operand.layout in (
                "continuous_group3d",
                "eu_align_group3d",
                "compact_group3d",
                "eu_align_xn_group3d",
                "compact_xn_group3d",
        ):
            n, c, d, h, w = 0, 1, 2, 3, 4
            data = data.transpose((d, n, c, h, w))
        return data

    def get_ref_data(self, operand: TLValue):
        for ref in self.ref_data:
            name = operand.name
            if name not in ref and name.startswith("load_"):
                name = name.replace("load_", "")
            if name not in ref:
                continue

            ref_data = ref[name]
            return self.format_data(operand, ref_data)

        return None

    def collect_infer_data(self, operand: TLValue, actual, desired):
        if self.desire_op and operand.name not in self.desire_op:
            return
        if not self.mlir_ref:
            self.collect_infer_data_from_ref(operand, actual, desired)
        else:
            self.collect_infer_data_from_mlir(operand, actual)

    def collect_infer_data_from_ref(self, operand: TLValue, actual, desired):
        for idx, ref in enumerate(self.ref_data):
            if operand.name not in ref:
                continue
            _slice = operand.slice
            if _slice == "[...]":
                slice_list = operand.memory_type[1:-1].replace("x", ",").split(",")[:-1]
                sliced_shape = tuple(int(i) for i in slice_list)
                slices = [slice(None, None) for _ in slice_list]
            else:
                slice_list = _slice[1:-1].split(",")
                sliced_shape = tuple(
                    [int(slice.split(":")[1]) - int(slice.split(":")[0]) for slice in slice_list])
                slices = [
                    slice(int(s.strip().split(":")[0]), int(s.strip().split(":")[1]))
                    for s in slice_list
                ]
            actual = actual.reshape(desired.shape)

            if operand.layout in (
                    "continuous_group3d",
                    "eu_align_group3d",
                    "compact_group3d",
                    "eu_align_xn_group3d",
                    "compact_xn_group3d",
            ):
                d, n, c, h, w = 0, 1, 2, 3, 4
                actual = actual.transpose((n, c, d, h, w))

            reshape = operand.reshape
            origin_shape = ref[operand.name].shape
            if reshape:
                reshape = eval(reshape[1:-1].replace("x", ","))
            else:
                reshape = sliced_shape

            if operand.name not in self.ref_data_from_inference:
                tmp = np.zeros(reshape)
                tmp[tuple(slices)] = actual
                self.ref_data_from_inference[operand.name] = tmp.reshape(origin_shape)
            else:
                tmp = self.ref_data_from_inference[operand.name]
                tmp = tmp.reshape(reshape)
                tmp[tuple(slices)] = actual
                self.ref_data_from_inference[operand.name] = tmp.reshape(origin_shape)
            return

    def cal_desired_shape(self, sliced_shape: Tuple[int], layout: str):
        if layout in (
                "continuous_group3d",
                "eu_align_group3d",
                "compact_group3d",
                "eu_align_xn_group3d",
                "compact_xn_group3d",
        ):
            n, c, d, h, w = 0, 1, 2, 3, 4
            # data = data.transpose((d, n, c, h, w))
            return (
                sliced_shape[d],
                sliced_shape[n],
                sliced_shape[c],
                sliced_shape[h],
                sliced_shape[w],
            )
        return sliced_shape

    def collect_infer_data_from_mlir(self, operand: TLValue, actual: np.ndarray):
        if operand.name not in self.ops:
            return

        _slice = operand.slice
        if _slice == "[...]":
            slice_list = operand.memory_type[1:-1].replace("x", ",").split(",")[:-1]
            sliced_shape = tuple(int(i) for i in slice_list)
            slices = [slice(None, None) for _ in slice_list]
        else:
            slice_list = _slice[1:-1].split(",")
            sliced_shape = tuple(
                [int(slice.split(":")[1]) - int(slice.split(":")[0]) for slice in slice_list])
            slices = [
                slice(int(s.strip().split(":")[0]), int(s.strip().split(":")[1]))
                for s in slice_list
            ]
        actual = actual.reshape(self.cal_desired_shape(sliced_shape, operand.layout))

        if operand.layout in (
                "continuous_group3d",
                "eu_align_group3d",
                "compact_group3d",
                "eu_align_xn_group3d",
                "compact_xn_group3d",
        ):
            d, n, c, h, w = 0, 1, 2, 3, 4
            actual = actual.transpose((n, c, d, h, w))

        reshape = operand.reshape
        origin_shape = self.ops[operand.name].shape
        if reshape:
            reshape = eval(reshape[1:-1].replace("x", ","))
        else:
            reshape = sliced_shape

        if operand.name not in self.ref_data_from_inference:
            tmp = np.zeros(reshape)
            tmp[tuple(slices)] = actual
            self.ref_data_from_inference[operand.name] = tmp.reshape(origin_shape)
        else:
            tmp = self.ref_data_from_inference[operand.name]
            tmp = tmp.reshape(reshape)
            tmp[tuple(slices)] = actual
            self.ref_data_from_inference[operand.name] = tmp.reshape(origin_shape)

    def do_watch_mem(self, args):
        try:
            args = int(args)
            cmd = self.tdb.cmditer[int(args)]
        except ValueError:
            cmd = self.tdb.get_cmd()
        self.watch = cmd
        self.watch_data = []

    def do_display_cmd(self, args):
        self.tdb.message(self.tdb.cmditer[int(args)])

    def check_data(self, point_index, is_operand, value_view: ValueView) -> ComparedResult:
        value = value_view.value
        if value.name in self.excepts:
            value_res = ComparedResult(value_view, None, msg="ignore")
            return value_res

        # this hack should only enable in multicore
        if not is_operand:
            if value_view.file_line in self.tdb.global_layer_line:
                self.tdb.global_layer_line[value_view.file_line] -= 1
                if self.tdb.global_layer_line[value_view.file_line] != 0:
                    value_res = ComparedResult(value_view, None, msg="ignore")
                    return value_res

        context = self.tdb.context
        # only used for pcie mode
        memref = value.get_memref(context)
        if not self.is_soc and not context.using_cmodel:
            if memref.mtype != MType.G and memref.mtype != MType.R:
                value_res = ComparedResult(value_view, None, msg="ignore")
                return value_res

        cmd = self.tdb.cmditer[point_index]

        # only used for soc mode
        if self.is_soc:
            if is_operand:
                DataCheck.soc_values_in[point_index].append(
                    Pickled_Value(value, value_view.file_line, value_view.cmd_point, cmd.core_id))
            else:
                DataCheck.soc_values_out.append(
                    Pickled_Value(value, value_view.file_line, value_view.cmd_point, cmd.core_id))
            return ComparedResult(value_view, None, msg="ignore")
        # breakpoint()
        raw_data = context.memory.get_data(ValueRef(memref, core_id=cmd.core_id))

        desired = self.get_ref_data(value)
        if self.dump_mode == DumpMode.TPULANG and self.out_fixed == True:
            actual = raw_data.astype(np.float32)
        else:
            actual = (raw_data.astype(np.float32) - value.zero_point) * value.scale

        if self.tdb.args.get("infer_mode") == "standalone":
            if self.dump_mode == DumpMode.TPULANG and self.out_fixed == True:
                actual = raw_data.astype(np.float32)
                raw_desired = desired.astype(raw_data.dtype) if desired is not None else None
            else:
                actual = (raw_data.astype(np.float32) - value.zero_point) * value.scale
                raw_desired = (desired / value.scale + value.zero_point).astype(
                    raw_data.dtype) if desired is not None else None
            if raw_desired is not None:
                if not context.memory.set_data(ValueRef(memref, core_id=cmd.core_id), raw_desired):
                    # self.tdb.debug(f"set data {value.name} failed")
                    pass

        if self.dump_mode in {DumpMode.COMB, DumpMode.COMB_ALL
                              } or self.dump_mode == DumpMode.TPULANG:
            self.collect_infer_data(value, actual, desired)

        if self.skip_check:  # CModel mode or Pcie mode
            value_res = ComparedResult(value_view, None, msg="ignore")
            return value_res

        name = f"{value.name}_asm_{value_view.file_line}_{value_view.loc_index}_{value_view.cmd_point}"
        dump_actual = False
        dump_desired = False

        if self.dump_mode == DumpMode.ALL:
            dump_actual = True

        if desired is not None:
            actual = actual.reshape(desired.shape)
            cmp_res, msg = list(self.tc.assert_allclose(actual, desired, dump_mode=self.dump_mode))
            value_res = ComparedResult(
                value_view,
                cmp=cmp_res,
                zero_point=value.zero_point,
                scale=value.scale,
                msg=msg,
            )
            cmp_failed = not cmp_res[0]
            cmd = self.tdb.get_cmd()

            dump_desired = cmp_failed
            if self.dump_mode == DumpMode.ALL or cmp_failed:
                dump_actual = True
        else:
            value_res = ComparedResult(value_view, None)

        if dump_actual:
            self.failed_tensor[f"{name}_actual"] = actual
        if dump_desired:
            self.failed_tensor[f"{name}_desired"] = desired
        if dump_actual or dump_desired:
            self.failed_summary_info.append({
                "loc": value.name,
                "cmd_point": value_view.cmd_point,
                "cmd_id": cmd.cmd_id,
                "core_id": cmd.core_id,
                "subnet_id": cmd.subnet_id,
                "loc_index": value_view.loc_index,
                "operand": is_operand,
                "value": value.to_dict(),
                "cmp_failed": cmp_failed,
            })

        return value_res

    def compare(self, tdb: TdbCmdBackend, is_operand):
        index_plugin = self.index
        point_index = tdb.cmd_point
        values = None

        if is_operand:
            point_index += 1
            values = tdb.index_df.loc[tdb.index_df["executed_id"] == point_index,
                                      "operands"].tolist()
        else:
            values = tdb.index_df.loc[tdb.index_df["executed_id"] == point_index,
                                      "results"].tolist()

        if values:
            values = values[0]

        if not values:
            return CmpState.Middle
        if isinstance(values, float) and math.isnan(values):
            return CmpState.Middle

        reports = self.reports
        success = True
        for value_view in values:
            if is_operand != value_view.is_operand:
                continue

            if value_view.value is None:  # top.None
                continue

            cmp_res = self.check_data(point_index - 1, is_operand, value_view)
            if cmp_res.state == CmpState.Fail:
                success = False

            reports.setdefault(value_view.loc_index, []).append(cmp_res)
            lino = index_plugin.tdb.index_df.loc[self.tdb.index_df["executed_id"] ==
                                                 value_view.cmd_point, "line-num"].item()
            self.index_record.setdefault(lino, []).append(cmp_res)

        return success

    def before_step(self, tdb: TdbCmdBackend):
        if self.ref_data is None or not self.enabled:
            return

        ret = self.compare(tdb, True)
        if tdb.cache_mode == "generate":
            raise BufferError()

        if not ret and self.break_when_fail:
            raise StopIteration()

    def after_step(self, tdb: TdbCmdBackend):
        if self.watch is not None:

            for ret in self.watch.results:
                self.watch_data.append(
                    self.tdb.context.memory.get_data(ret.to_ref(core_id=self.watch.core_id)).copy())
                if len(self.watch_data) > 1:
                    np.savez(f"watch_data_{len(self.watch_data)}.npz", self.watch_data[-1])
                    if not (self.watch_data[-2] == self.watch_data[-1]).all():
                        self.tdb.message("change")
                        raise BreakpointStop()

        if self.ref_data is None or not self.enabled:
            return

        ret = self.compare(tdb, False)
        if not ret and self.break_when_fail:
            raise BreakpointStop()

    def after_stop(self, tdb: TdbCmdBackend):
        # make sure npz file is valid
        if self.dump_mode in {DumpMode.COMB, DumpMode.COMB_ALL}:
            comb_infer_path = os.path.join(self.tdb.bmodel_dir, f"bmodel_inference.npz")
            self.tdb.save_npz(comb_infer_path, self.ref_data_from_inference, "bmodel_inference")

        if self._failed_tensor:
            self.failed_tensor.close()
            self.tdb.file_recorder.add_file(bmodel_failed_tensor=self.failed_tensor.fn)

            if len(self.failed_summary_info) > 0:
                fn = "bmodel_failed_summary.json"
                self.tdb.save_json(fn, self.failed_summary_info, "bmodel_failed_summary")
        return super().after_stop(tdb)


class CheckLMem(TdbPlugin):
    name = "check-lmem"

    def __init__(self, tdb: TdbCmdBackend) -> None:
        super().__init__(tdb)
        self.RET = []

    def before_step(self, tdb: TdbCmdBackend):
        self.store(tdb)

    def after_step(self, tdb: TdbCmdBackend):
        self.store(tdb)

    def store(self, tdb: TdbCmdBackend):
        import ctypes
        import platform
        from debugger.target_1688.device_rt import DeviceRunner

        runner: DeviceRunner = tdb.runner
        runner.lib.memcpy_l2s(
            runner.runner_p,
            runner.memory.LMEM[0].ctypes.data_as(ctypes.c_void_p),
        )
        self.RET.append([str(tdb.get_cmd()), tdb.cmd_point, runner.memory.LMEM[0].copy()])

        if tdb.cmd_point == 11:
            with open(f"{platform.machine()}.npz", "wb") as w:
                pickle.dump(self.RET, w)
            raise StopIteration()
