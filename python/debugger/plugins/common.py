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


from rich.progress import (
    Progress as Progressbar,
    TimeRemainingColumn,
    TaskProgressColumn,
    TextColumn,
    ProgressColumn,
    BarColumn,
)

import pandas as pd

from ..final_mlir import CMD, FinalMlirIndex, Value

from ..target_common.op_support import BaseTpuOp
from ..target_common import CMDType
from ..tdb_support import (
    TdbCmdBackend,
    TdbPlugin,
    Displays,
    TdbStatus,
    TdbPluginCmd,
    complete_file,
)


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
    def cmd2index(self):
        return self.tdb.cmd2index

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
        subnet_offsets = {}
        for loc_index, loc in enumerate(self.final_mlir.loc.tensor_loc):
            (
                subnet_id,
                tiu,
                dma,
                core_id,
            ) = (loc.subnet_id, *loc.tiu_dma_id_before, loc.core_id)
            tiu_offset, dma_offset = subnet_offsets.get(
                subnet_id, (float("inf"), float("inf"))
            )
            tiu_offset = min(tiu_offset, tiu)
            dma_offset = min(dma_offset, dma)
            subnet_offsets[subnet_id] = (tiu_offset, dma_offset)

        self.index2loc_range: Dict[int, int] = {}

        last_af_index = -1

        self.cmdkey2loc: Dict[int, List[ValueView]] = OrderedDict()

        for loc_index, loc in enumerate(self.final_mlir.loc.tensor_loc):
            # before execution
            (
                subnet_id,
                tiu,
                dma,
                core_id,
            ) = loc.tuple_key_before
            x_index = y_index = None
            tiu_offset, dma_offset = subnet_offsets[subnet_id]

            if tiu - tiu_offset > 0:
                x_index = tdb.cmd2index[(subnet_id, None, tiu - tiu_offset, core_id)]
            if dma - dma_offset > 0:
                y_index = tdb.cmd2index[(subnet_id, dma - dma_offset, None, core_id)]

            bf_index = max_with_none(x_index, y_index, last_af_index + 1)

            self.cmdkey2loc.setdefault(bf_index, []).extend(
                Operand(opd, opd_index, loc_index, loc.loc_name, bf_index)
                for opd_index, opd in enumerate(loc.operands)
            )

            # after execution
            (
                subnet_id,
                tiu,
                dma,
                core_id,
            ) = (loc.subnet_id, *loc.tiu_dma_id_after, loc.core_id)
            x_index = y_index = None

            if tiu - tiu_offset > 0:
                x_index = tdb.cmd2index[(subnet_id, None, tiu - tiu_offset, core_id)]
            if dma - dma_offset > 0:
                y_index = tdb.cmd2index[(subnet_id, dma - dma_offset, None, core_id)]

            last_af_index = af_index = max_with_none(x_index, y_index)

            self.cmdkey2loc.setdefault(af_index, []).extend(
                Result(opd, opd_index, loc_index, loc.loc_name, af_index)
                for opd_index, opd in enumerate(loc.results)
            )

            for index in range(bf_index, af_index + 1):
                assert index not in self.index2loc_range, index

                self.index2loc_range[index] = loc_index

    def get_mlir_by_atomic(self, op: BaseTpuOp):
        loc = self.get_loc_by_atomic(op)
        file_line = self.final_mlir.get_fileline_by_locname(loc.loc_name)
        return self.final_mlir.lines[file_line]

    def get_mlir_context_by_atomic(self, op: BaseTpuOp, pre=2, next=2) -> List[str]:
        loc = self.get_loc_by_atomic(op)
        file_line = self.final_mlir.get_fileline_by_locname(loc.loc_name)
        return self.final_mlir.lines[max(0, file_line - 1 - pre) : file_line - 1 + next]

    def get_locindex_by_atomic(self, op: BaseTpuOp) -> int:
        index = self.cmd2index[op.tuple_key]
        loc_index = self.index2loc_range[index]
        return loc_index

    def get_loc_by_atomic(self, op: BaseTpuOp) -> CMD:
        loc_index = self.get_locindex_by_atomic(op)
        return self.final_mlir.loc[loc_index]

    def get_loc_context_by_atomic(self, op: BaseTpuOp, pre=2, next=2) -> List[CMD]:
        loc_index = self.get_locindex_by_atomic(op)
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
            op = self.tdb.get_op()
        except StopIteration:
            self.tdb.message("no cmd next.")
            return
        if arg == "":
            self.tdb.message(op.operands)
            return

        try:
            index = int(arg)
            if op.cmd_type == CMDType.cpu:
                if op.cmd_id == 0:
                    data = self.tdb.memory.get_data(op.operands[index])
                else:
                    data = self.tdb.memory.get_cpu_data(op.cmd_id)[op.operands[index]]
            elif op.cmd_type.is_tpu():
                if op.operands[index].is_scalar:
                    data = op.operands[index].data
                else:
                    data = self.tdb.memory.get_data(op.operands[index])
            else:
                self.tdb.error("")
                return
            print(data)
        except (IndexError, SyntaxError, ValueError) as e:
            self.tdb.error(e)

    def do_next(self, arg):
        try:
            op = self.tdb.get_op()
            self.tdb.message(op)
        except StopIteration:
            self.tdb.error("no cmd next.")

    do_op = do_next

    def do_pre(self, arg):
        try:
            op = self.tdb.get_preop()
            self.tdb.message(op)
        except StopIteration:
            self.tdb.message("no cmd pre.")
            return

    def do_out(self, arg):
        try:
            op = self.tdb.get_preop()
        except StopIteration:
            self.tdb.message("no cmd pre.")
            return

        if arg == "":
            self.tdb.message(op.results)
            return

        try:
            index = int(arg)
            if op.cmd_type == CMDType.cpu:
                data = self.tdb.memory.get_cpu_data(op.cmd_id)[index]
            elif op.cmd_type.is_tpu():
                if op.results[index].is_scalar:
                    data = op.results[index].data
                else:
                    data = self.tdb.memory.get_data(op.results[index])
            else:
                self.tdb.error("")
                return
            print(data)
        except (IndexError, SyntaxError, ValueError) as e:
            self.tdb.error(e)

    def after_start(self, tdb: TdbCmdBackend):
        try:
            tdb.message(tdb.get_op())
        except StopIteration:
            pass

    def after_stop(self, tdb: TdbCmdBackend):
        try:
            tdb.message(tdb.get_op())
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
        (subnet_id, tiu_id, dma_id, core_id) = tdb.get_op().tuple_key
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
