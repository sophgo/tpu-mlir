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
from ..target_1688.context import BM1688Context
from ..target_1690.context import BM1690Context
from ..target_2380.context import SG2380Context


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


class DumpMode(Enum):
    NEVER = 0
    FAILED = 1
    ALL = 2
    COMB = 3
    TPULANG = 4


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
        idx = [
            x[0] for x in sorted(zip(idx, diff[idx]), key=lambda x: x[1], reverse=True)
        ]
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
            header = (
                "Not equal to tolerance "
                + f"cos={self.cosine_similarity_tol}"
                + f", euc={self.euclidean_similarity_tol}"
            )

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
        self.ref_data_from_inference = []
        self.mlir_ref = False
        for ref_fn in tdb.reference_data_fns:
            self.ref_data_from_inference.append({})
            if ref_fn.endswith(".mlir"):
                self.ops = self.collect_op_name_dict(ref_fn, tdb)
                self.mlir_ref = True
            else:
                self.ref_data.append(np.load(ref_fn))
                self.mlir_ref = False or self.mlir_ref

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

    def collect_op_name_dict(self, ref_fn, tdb: TdbCmdBackend):
        if tdb.cache_mode in {'generate', 'online'}:
            from utils.mlir_parser import MlirParser
            ops = MlirParser(ref_fn).collect_op_name_dict()
            if tdb.cache_mode == 'generate':
                with open(f"{ref_fn}.tdb_cache.ops.pickle", 'wb') as w:
                    pickle.dump(ops, w)
        elif tdb.cache_mode == 'offline':
            with open(f"{ref_fn}.tdb_cache.ops.pickle", 'rb') as r:
                ops = pickle.load(r)
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
        self.tdb.message(f"dump mode = {self.dump_mode}")

    @property
    def failed_tensor(self):
        if self._failed_tensor is None:
            file = self.failed_results_fn
            if os.path.exists(file):
                os.remove(file)
                print(f"remove exist {file}")
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
            with open(f"{arg}.txt", "w") as w:
                w.write("\n".join(failed_names))
                self.tdb.message(f"failed value name list are saved in {arg}.txt")

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
                        codelike_format(
                            self.tdb.get_op_context(2, 2, vv.value_view.cmd_point), 2
                        ),
                        title="asm",
                    )
                    msg = Panel(Group(table, err_msg, asm_codes))
                    self.tdb.message(msg)
                else:
                    self.tdb.message(Panel(table))

        except Exception as e:
            self.tdb.error(e)

    def complete_data(self, text="", line="", begidx=0, endidx=0) -> List[str]:
        return list(
            filter(lambda x: x.startswith(text), map(str, self.index_record.keys()))
        )

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

    def get_ref_data(self, operand: TLValue):
        for ref in self.ref_data:
            name = operand.name
            if name not in ref and name.startswith("load_"):
                name = name.replace("load_", "")
            if name not in ref:
                continue
            reshape = operand.reshape
            ref_data = ref[name]

            if reshape:
                reshape = eval(reshape[1:-1].replace("x", ","))  # '1,3,1,192,1024'
                ref_data = ref_data.reshape(reshape)

            _slice = operand.slice
            data = eval(f"ref_data{_slice}")  # type: np.ndarray
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
                    [
                        int(slice.split(":")[1]) - int(slice.split(":")[0])
                        for slice in slice_list
                    ]
                )
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

            if operand.name not in self.ref_data_from_inference[idx]:
                tmp = np.zeros(reshape)
                tmp[tuple(slices)] = actual
                self.ref_data_from_inference[idx][operand.name] = tmp.reshape(origin_shape)
            else:
                tmp = self.ref_data_from_inference[idx][operand.name]
                tmp = tmp.reshape(reshape)
                tmp[tuple(slices)] = actual
                self.ref_data_from_inference[idx][operand.name] = tmp.reshape(origin_shape)
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
                [
                    int(slice.split(":")[1]) - int(slice.split(":")[0])
                    for slice in slice_list
                ]
            )
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

        if operand.name not in self.ref_data_from_inference[0]:
            tmp = np.zeros(reshape)
            tmp[tuple(slices)] = actual
            self.ref_data_from_inference[0][operand.name] = tmp.reshape(origin_shape)
        else:
            tmp = self.ref_data_from_inference[0][operand.name]
            tmp = tmp.reshape(reshape)
            tmp[tuple(slices)] = actual
            self.ref_data_from_inference[0][operand.name] = tmp.reshape(origin_shape)

    def check_data(
        self, point_index, is_operand, value_view: ValueView
    ) -> ComparedResult:
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
        memref = value.get_memref(context)
        # only used for pcie mode
        if not self.is_soc and not context.using_cmodel:
            if memref.mtype != MType.G and memref.mtype != MType.R:
                value_res = ComparedResult(value_view, None, msg="ignore")
                return value_res

        cmd = self.tdb.cmditer[point_index]

        # only used for soc mode
        if self.is_soc:
            if is_operand:
                DataCheck.soc_values_in[point_index].append(
                    Pickled_Value(value, memref, value.dtype, value.zero_point, value.scale, value_view.file_line, value_view.cmd_point)
                )
            else:
                # memref_type = 0 if isinstance(memref, Scalar) else 1
                DataCheck.soc_values_out.append(
                    Pickled_Value(value, memref, value.dtype, value.zero_point, value.scale, value_view.file_line, value_view.cmd_point)
                )
            return ComparedResult(value_view, None, msg="ignore")

        if self.tdb.cache_mode == 'generate':
            if is_operand:
                DataCheck.soc_values_in[point_index].append(value)
            else:
                DataCheck.soc_values_out.append(value)
            return ComparedResult(value_view, None, msg="ignore")
        # breakpoint()
        raw_data = context.memory.get_data(ValueRef(memref, core_id=cmd.core_id))

        if self.dump_mode == DumpMode.TPULANG and self.out_fixed == True:
            actual = raw_data.astype(np.float32)
        else:
            actual = (raw_data.astype(np.float32) - value.zero_point) * value.scale

        desired = self.get_ref_data(value)
        if self.dump_mode == DumpMode.COMB or self.dump_mode == DumpMode.TPULANG:
            self.collect_infer_data(value, actual, desired)

        if self.skip_check:  # CModel mode or Pcie mode
            value_res = ComparedResult(value_view, None, msg="ignore")
            return value_res

        if desired is None:
            value_res = ComparedResult(value_view, None)
            name = f"{value.name}_asm_{value_view.loc_index}_{value_view.cmd_point}"
            if self.dump_mode == DumpMode.ALL:
                self.failed_tensor[f"{name}_actual"] = actual
        else:
            actual = actual.reshape(desired.shape)
            cmp_res, msg = list(self.tc.assert_allclose(actual, desired, dump_mode=self.dump_mode))

            value_res = ComparedResult(
                value_view,
                cmp=cmp_res,
                zero_point=value.zero_point,
                scale=value.scale,
                msg=msg,
            )
            cmd_failed = not cmp_res[0]
            name = f"{value.name}_asm_{value_view.loc_index}_{value_view.cmd_point}"
            if self.dump_mode == DumpMode.ALL:
                self.failed_tensor[f"{name}_actual"] = actual
                if cmd_failed:
                    self.failed_tensor[f"{name}_desired"] = desired
            elif self.dump_mode == DumpMode.FAILED and cmd_failed:
                self.failed_tensor[f"{name}_actual"] = actual
                self.failed_tensor[f"{name}_desired"] = desired

        return value_res

    def compare(self, tdb: TdbCmdBackend, is_operand):
        index_plugin = self.index
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
            lino = index_plugin.tdb.index_df.loc[
                self.tdb.index_df["executed_id"] == value_view.cmd_point, "line-num"
            ].item()
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
        if self.ref_data is None or not self.enabled:
            return

        ret = self.compare(tdb, False)
        if not ret and self.break_when_fail:
            raise BreakpointStop()

    def after_stop(self, tdb: TdbCmdBackend):
        # make sure npz file is valid
        if self.dump_mode == DumpMode.COMB:
            for idx, tensor_dict in enumerate(self.ref_data_from_inference):
                np.savez(f"bmodel_inference_{idx}.npz", **tensor_dict)
        else:
            self.failed_tensor.close()
        return super().after_stop(tdb)


class CheckLMem(TdbPlugin):
    name = 'check-lmem'

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
            with open(f"{platform.machine()}.npz",'wb') as w:
                pickle.dump(self.RET, w)
            raise StopIteration()
