# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
from collections import OrderedDict
import collections
from dataclasses import dataclass
from typing import List, Dict, Optional


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
import numpy as np

from ..final_mlir import CMD, FinalMlirIndex, Value

from ..target_common.op_support import BaseTpuCmd
from ..target_common import CMDType
from ..tdb_support import (
    TdbCmdBackend,
    TdbPlugin,
    Displays,
    TdbStatus,
    TdbPluginCmd,
    complete_file,
)
from ..target_1688.context import BM1688Context
from ..target_2260.context import SG2260Context


def max_with_none(*args):
    args = [i for i in args if i is not None]
    if len(args) == 1:
        return args[0]
    if len(args) == 0:
        return 0

    return max(args)


@dataclass
class ValueView:
    value: Value
    index: int
    loc_index: int
    loc_name: str
    cmd_point: int
    file_line: int

    @property
    def is_operand(self):
        return isinstance(self, Operand)


class Result(ValueView):
    pass


class Operand(ValueView):
    pass


class ReloadPlugin(TdbPlugin, TdbPluginCmd):
    name = "reload"

    def do_mlir(self, arg):
        """
        reload mlir <final.mlir> <tensor_location.json>
        """
        res = arg.split(" ")
        if len(res) != 2:
            self.tdb.error("reload mlir <final.mlir> <tensor_location.json>")
            return

        final_mlir, tensor_location = res
        self.tdb.final_mlir_fn = final_mlir
        self.tdb.tensor_loc_file = tensor_location
        self.tdb.do_start()

    def do_input(self, input):
        """
        reload input input_fn
        """
        self.tdb.input_data_fn = input
        self.tdb.do_start()

    complete_mlir = complete_file
    complete_input = complete_file


class FinalMlirIndexPlugin(TdbPlugin):
    """
    append final-mlir indexs by extending tdb.index_df columns.

    executed_id, loc_indexs

    """

    name = "final-mlir"

    def __init__(self, tdb: TdbCmdBackend) -> None:
        super().__init__(tdb)

    def __str__(self) -> str:
        flag = "√" if self.enabled else "x"
        return f"{self.name}({flag})"

    @property
    def enabled(self):
        tdb = self.tdb
        return tdb.final_mlir_fn is not None and tdb.tensor_loc_file is not None

    @property
    def index_df(self):
        return self.tdb.index_df

    @property
    def final_mlir_fn(self):
        return self.tdb.final_mlir_fn

    @property
    def tensor_loc_file(self):
        return self.tdb.tensor_loc_file

    def after_load(self, tdb: TdbCmdBackend):
        if self.enabled:
            self._build_index(tdb)

    def _build_index(self, tdb: TdbCmdBackend):
        # create subnet tiu, dma id offset
        self.final_mlir = FinalMlirIndex(self.final_mlir_fn, self.tensor_loc_file)
        last_af_point = 0
        indexs = tdb.op_df.index

        def find_point(key):
            ret = tdb.op_df["executed_id"][indexs == key]

            if len(ret) == 0:
                raise KeyError(f"cannot find command of key {key}")
            elif len(ret) > 1:
                raise ValueError(
                    f"find multiple command have key {key}, please report this bug."
                )

            return ret[0]

        def assemble_tuple_key(subnet_id, core_id, cmd_id, cmd_type):
            if cmd_type == CMDType.tiu:
                return (subnet_id, cmd_id, None, core_id)
            elif cmd_type == CMDType.dma:
                return (subnet_id, None, cmd_id, core_id)
            else:
                raise RuntimeError("Not Supported CMDType!")

        # debug options
        # pd.set_option("display.max_rows", None)
        # pd.set_option("display.max_columns", None)
        # pd.set_option("display.expand_frame_repr", False)

        # when cmd_point reach point
        # it means the cmd in cmditer[point-1] has been executed
        # data-checker need to compare loc operands before execute bf_point
        # and after execute af_point

        loc_records = []
        for loc_index, loc in enumerate(self.final_mlir.loc.tensor_loc):
            if loc.tiu_dma_id_before == loc.tiu_dma_id_after:
                # no cmd operation, like reshape
                continue

            # the tiu/dma cmd-id state before execution this loc
            # we need to find the pointer

            (
                subnet_id,
                tiu_before,
                dma_before,
                core_id,
            ) = loc.tuple_key_before
            (
                subnet_id,
                tiu_after,
                dma_after,
                core_id,
            ) = loc.tuple_key_after

            tiu_point = dma_point = None
            if tiu_before > 0:
                tiu_point = find_point((subnet_id, tiu_before, None, core_id))
            if dma_before > 0:
                dma_point = find_point((subnet_id, None, dma_before, core_id))
            bf_point = max_with_none(tiu_point, dma_point, last_af_point) + 1
            operands_tmp = collections.defaultdict(list)
            operands_tmp[tdb.cmditer[bf_point - 1].tuple_key].extend(
                Operand(opd, opd_index, loc_index, opd.name, bf_point, loc.file_line)
                for opd_index, opd in enumerate(loc.operands)
                if opd
            )

            tiu_point = dma_point = None
            if tiu_after > 0:
                tiu_point = find_point((subnet_id, tiu_after, None, core_id))
            if dma_after > 0:
                dma_point = find_point((subnet_id, None, dma_after, core_id))
            last_af_point = af_point = max_with_none(tiu_point, dma_point)
            results_tmp = collections.defaultdict(list)
            results_tmp[tdb.cmditer[af_point - 1].tuple_key].extend(
                Result(opd, opd_index, loc_index, opd.name, af_point, loc.file_line)
                for opd_index, opd in enumerate(loc.results)
                if opd
            )

            for i in range(tiu_before + 1, tiu_after + 1):
                record = {
                    "loc_index": loc.loc_index,
                    "line-num": loc.file_line,
                    "subnet_id": subnet_id,
                    "core_id": core_id,
                    "cmd_id": i,
                    "cmd_type": CMDType.tiu,
                    "operands": operands_tmp[
                        assemble_tuple_key(subnet_id, core_id, i, CMDType.tiu)
                    ],
                    "results": results_tmp[
                        assemble_tuple_key(subnet_id, core_id, i, CMDType.tiu)
                    ],
                }
                loc_records.append(record)

            for j in range(dma_before + 1, dma_after + 1):
                record = {
                    "loc_index": loc.loc_index,
                    "line-num": loc.file_line,
                    "subnet_id": subnet_id,
                    "core_id": core_id,
                    "cmd_id": j,
                    "cmd_type": CMDType.dma,
                    "operands": operands_tmp[
                        assemble_tuple_key(subnet_id, core_id, j, CMDType.dma)
                    ],
                    "results": results_tmp[
                        assemble_tuple_key(subnet_id, core_id, j, CMDType.dma)
                    ],
                }
                loc_records.append(record)

        loc_df = pd.DataFrame.from_records(loc_records)
        tdb.index_df = pd.merge(tdb.op_df, loc_df, how="outer")
        tdb.index_df = tdb.index_df.set_index("cmd_index", drop=True)
        # replace all NaN values with zeros
        tdb.index_df["loc_index"] = tdb.index_df["loc_index"].fillna(-1)
        # convert 'loc_index' column from float to integer
        tdb.index_df["loc_index"] = tdb.index_df["loc_index"].astype(int)

    def get_mlir_by_point(self, point=None) -> Optional[str]:
        """NOTE: file-line in tensor_location.json starts from 1"""
        file_line = self.tdb.index_df.loc[
            self.tdb.index_df["executed_id"] == point, "line-num"
        ].item()
        return self.final_mlir.lines[file_line - 1]

    def get_mlir_context_by_point(
        self, point=None, pre=2, next=2
    ) -> Optional[List[str]]:
        file_line = self.tdb.index_df.loc[
            self.tdb.index_df["executed_id"] == point, "line-num"
        ].item()
        return self.final_mlir.lines[max(0, file_line - 1 - pre) : file_line - 1 + next]

    def get_locindex_by_atomic(self, point=None) -> Optional[int]:
        """
        N cmds have N+1 positions,
        use tdb.cmd_point other than cmd2index to get current point
        """
        if point is None:
            point = self.tdb.cmd_point

        loc_index = self.tdb.index_df.loc[
            self.tdb.index_df["executed_id"] == point, "loc_index"
        ].item()

        if np.isnan(loc_index):
            return None
        return int(loc_index)

    def get_loc_by_point(self, point=None) -> Optional[CMD]:
        loc_index = self.get_locindex_by_atomic(point)
        if loc_index is None:
            return None
        return self.final_mlir.loc[loc_index]

    def get_loc_context_by_point(
        self, point=None, pre=2, next=2
    ) -> Optional[List[CMD]]:
        loc_index = self.get_locindex_by_atomic(point)
        if loc_index is None:
            return None
        return self.final_mlir.loc[max(0, loc_index - pre) : loc_index + next]


class DisplayPlugin(TdbPlugin, TdbPluginCmd):
    name = "display"

    def __init__(self, tdb: TdbCmdBackend) -> None:
        super().__init__(tdb)
        self.displays = Displays.get_instance()
        tdb.complete_display = tdb._complete_expression

    def do_delete(self, arg):
        """remove display"""
        pass

    def default(self, arg):
        """
        display arg info after each break

         - address:  4295028736@40
        """
        try:
            eval(arg)
        except Exception as e:
            self.error(f"Can not add display {arg}")
            self.error(e)
            return
        item_id = self.displays.add_display(arg)
        self.message(f"{item_id} {eval(arg)}")
        # self.plugins.after_stop(self)

    def after_stop(that, tdb: TdbCmdBackend):
        self = tdb
        table = []
        for k, dis in self.displays.display.items():
            try:
                table.append([f"%{k}:", str(eval(dis.expr))])
            except Exception:
                continue
        if len(table) > 0:
            df = pd.DataFrame(table)
            tdb.message(df.to_string(index=False, header=False))


class PrintPlugin(TdbPlugin, TdbPluginCmd):
    name = "print"
    func_names = ["p", "print"]

    def do_in(self, arg):
        try:
            cmd = self.tdb.get_cmd()
        except StopIteration:
            self.tdb.message("no cmd next.")
            return
        if arg == "":
            self.tdb.message(cmd.operands)
            return

        try:
            index = int(arg)
            if cmd.cmd_type == CMDType.cpu:
                if cmd.cmd_id == 0:
                    data = self.tdb.memory.get_data(cmd.operands[index])
                else:
                    data = self.tdb.memory.get_cpu_data(cmd.cmd_id)[cmd.operands[index]]
            elif cmd.cmd_type.is_static():
                if cmd.operands[index].is_scalar:
                    data = cmd.operands[index].data
                else:
                    if isinstance(self.tdb.context, SG2260Context) or isinstance(self.tdb.context, BM1688Context):
                        data = self.tdb.memory.get_data(cmd.operands[index], core_id=cmd.core_id)
                    else:
                        data = self.tdb.memory.get_data(cmd.operands[index])
            else:
                self.tdb.error("")
                return
            print(data)
        except (IndexError, SyntaxError, ValueError) as e:
            self.tdb.error(e)

    def do_next(self, arg):
        try:
            cmd = self.tdb.get_cmd()
            self.tdb.message(cmd)
        except StopIteration:
            self.tdb.error("no cmd next.")

    do_op = do_next

    def do_pre(self, arg):
        try:
            op = self.tdb.get_precmd()
            self.tdb.message(op)
        except StopIteration:
            self.tdb.message("no cmd pre.")
            return

    def do_out(self, arg):
        try:
            cmd = self.tdb.get_precmd()
        except StopIteration:
            self.tdb.message("no cmd pre.")
            return

        if arg == "":
            self.tdb.message(cmd.results)
            return

        try:
            index = int(arg)
            if cmd.cmd_type == CMDType.cpu:
                data = self.tdb.memory.get_cpu_data(cmd.cmd_id)[index]
            elif cmd.cmd_type.is_static():
                if cmd.results[index].is_scalar:
                    data = cmd.results[index].data
                else:
                    if isinstance(self.tdb.context, SG2260Context) or isinstance(self.tdb.context, BM1688Context):
                        data = self.tdb.memory.get_data(cmd.operands[index], core_id=cmd.core_id)
                    else:
                        data = self.tdb.memory.get_data(cmd.operands[index])
            else:
                self.tdb.error("")
                return
            print(data)
        except (IndexError, SyntaxError, ValueError) as e:
            self.tdb.error(e)

    def after_start(self, tdb: TdbCmdBackend):
        try:
            tdb.message(tdb.get_cmd())
        except StopIteration:
            pass

    def after_stop(self, tdb: TdbCmdBackend):
        try:
            tdb.message(tdb.get_cmd())
        except StopIteration:
            pass


class ProgressPlugin(TdbPlugin):
    name = "progress"

    def after_load(self, tdb: TdbCmdBackend):
        columns: List["ProgressColumn"] = []
        columns.extend(
            (
                TextColumn("{task.description}"),
                BarColumn(
                    style="bar.back",
                    complete_style="bar.complete",
                    finished_style="bar.finished",
                    pulse_style="bar.pulse",
                ),
                TaskProgressColumn(show_speed=True),
                TimeRemainingColumn(elapsed_when_finished=True),
                TimeElapsedColumn(),
            )
        )

        progress = Progressbar(
            *columns,
            auto_refresh=True,
            console=None,
            transient=True,
            get_time=None,
            refresh_per_second=10,
            disable=False,
        )
        self.progress_id = progress.add_task("progress", total=len(tdb.cmditer))
        self.progress = progress

        self.visited_subnet = set()
        self.progress.stop()

    def after_step(self, tdb: TdbCmdBackend):
        if tdb.status != TdbStatus.RUNNING:
            return

        self.progress.start()
        (subnet_id, tiu_id, dma_id, core_id) = tdb.get_cmd().tuple_key
        if subnet_id not in self.visited_subnet:
            self.progress.print(f"run subnet {subnet_id}")
            self.visited_subnet.add(subnet_id)

        if tiu_id is None:
            tiu_id = "-"
        if dma_id is None:
            dma_id = "-"

        self.progress.update(
            self.progress_id,
            description=f"{tdb.cmd_point} {tiu_id}/{dma_id}",
            completed=tdb.cmd_point,
        )
        self.progress.refresh()

    def after_stop(self, tdb: TdbCmdBackend):
        self.progress.stop()


class AutoStaticCheck(TdbPlugin, TdbPluginCmd):
    name = "static-check"

    def default(self, args: str):
        if args.strip() in {"?", ""}:
            self.tdb.message(self.tdb.checker)
            return
        elif self.tdb.status == TdbStatus.UNINIT:
            self.tdb.error("do check after load context, type s/start to load")
            return
        for arg in args.split(","):
            self.tdb.checker.do_checker(arg)
            self.tdb.message(f"[DONE] {arg}")

    def after_load(self, tdb: TdbCmdBackend):
        for check_name in tdb.extra_check:
            tdb.checker.do_checker(check_name)
            tdb.message(f"[DONE] {check_name}")
