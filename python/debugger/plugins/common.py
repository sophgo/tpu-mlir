# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
import cmd
from typing import List

from rich.progress import (
    Progress as Progressbar,
    TimeRemainingColumn,
    TaskProgressColumn,
    TextColumn,
    ProgressColumn,
    BarColumn,
)

import pandas as pd
from ..target_common import CMDType
from ..tdb_support import TdbCmdBackend, TdbPlugin, Displays, TdbStatus


class DisplayPlugin(TdbPlugin, cmd.Cmd):
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
        # parse logic from cmd.onecmd
        # cmd, arg, line = self.parseline(line)
        # if cmd in {'i','info'}:

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


class PrintPlugin(TdbPlugin, cmd.Cmd):
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


class Progress(TdbPlugin):
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

    def after_next(self, tdb: TdbCmdBackend):
        self.progress.start()
        (subnet_id, tiu_id, dma_id, core_id) = tdb.get_op().tuple_key
        if subnet_id not in self.visited_subnet:
            self.progress.print(f"run subnet {subnet_id}")
            self.visited_subnet.add(subnet_id)

        if tiu_id is None:
            tiu_id = "-"
        if dma_id is None:
            dma_id = "-"

        self.progress.update(self.progress_id, description=f"{tiu_id}/{dma_id}")
        self.progress.advance(self.progress_id, 1)
        self.progress.refresh()

    def after_stop(self, tdb: TdbCmdBackend):
        self.progress.stop()


class AutoStaticCheck(TdbPlugin, cmd.Cmd):
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
