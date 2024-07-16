# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

from ..tdb_support import (
    TdbPlugin,
    TdbPluginCmd,
    codelike_format,
    safe_command,
)
from .common import FinalMlirIndexPlugin
from .breakpoints import BreakpointPlugin
from .watchpoints import WatchPlugin


class InfoPlugin(TdbPlugin, TdbPluginCmd):
    name = "info"
    func_names = ["info", "list"]

    def parse_comon(self, arg):
        arg_lis = [i for i in arg.strip().split() if i != ""]
        index = 0
        multi_context = len(arg_lis) > 0
        pre = next = -1
        if multi_context:
            if len(arg_lis) == 1:
                number = int(arg_lis[0])
                pre = number // 2
                next = number - pre
            else:
                pre, next = int(arg_lis[0]), int(arg_lis[1])

            index = pre if self.tdb.cmd_point - pre >= 0 else 0

        return index, multi_context, pre, next

    @property
    def status(self):
        return self.tdb.status

    def info_loc(self, arg):
        if self.status.NO_CAND:
            message = "The program is not being run."
            return message

        index_plugin = self.tdb.get_plugin(
            FinalMlirIndexPlugin
        )  # type: FinalMlirIndexPlugin
        if not index_plugin.enabled:
            return "final.mlir is not assigned, use index <final.mlir> <tensor_location.json> to rebuild index"

        index, multi_context, pre, next = self.parse_comon(arg)
        point = self.tdb.cmd_point

        res = (
            index_plugin.get_loc_context_by_point(point, pre, next)
            if multi_context
            else [index_plugin.get_loc_by_point(point)]
        )
        if res is None:
            self.tdb.error(
                f"{type(self.tdb.get_cmd()).__name__} cmd has no mlir context."
            )
            return ""

        message = codelike_format(res, index)
        return message

    def info_mlir(self, arg):
        if self.status.NO_CAND:
            message = "The program is not being run."
            return message

        index_plugin = self.tdb.get_plugin(
            FinalMlirIndexPlugin
        )  # type: FinalMlirIndexPlugin
        if not index_plugin.enabled:
            return "final.mlir is not assigned, use index <final.mlir> <tensor_location.json> to rebuild index"

        index, multi_context, pre, next = self.parse_comon(arg)
        point = self.tdb.cmd_point

        res = (
            index_plugin.get_mlir_context_by_point(point, pre, next)
            if multi_context
            else [index_plugin.get_mlir_by_point(point)]
        )
        if res is None:
            self.tdb.error(
                f"{type(self.tdb.get_cmd()).__name__} cmd has no mlir context."
            )
            return ""

        message = codelike_format(res, index)
        return message

    def info_asm(self, arg):
        if self.status.NO_CAND:
            message = "The program is not being run."
            return message
        index, multi_context, pre, next = self.parse_comon(arg)
        res = (
            self.tdb.get_op_context(pre, next)
            if multi_context
            else [self.tdb.get_cmd()]
        )
        message = codelike_format(res, index)
        return message

    def info_reg(self, arg):
        if self.status.NO_CAND:
            message = "The program is not being run."
            return message
        index, multi_context, pre, next = self.parse_comon(arg)
        res = (
            self.tdb.get_op_context(pre, next)
            if multi_context
            else [self.tdb.get_cmd()]
        )
        res = [i.reg for i in res]
        message = codelike_format(res, index)
        return message

    def info_buf(self, arg):
        if self.status.NO_CAND:
            message = "The program is not being run."
            return message
        index, multi_context, pre, next = self.parse_comon(arg)
        res = (
            self.tdb.get_op_context(pre, next)
            if multi_context
            else [self.tdb.get_cmd()]
        )
        res = [i.buf for i in res if i.cmd_type == i.cmd_type]
        message = codelike_format(res, index)
        return message

    def info_break(self, arg):
        b: BreakpointPlugin = self.tdb.get_plugin(BreakpointPlugin)
        return str(b.breakpoints)

    def info_watch(self, arg):
        w: WatchPlugin = self.tdb.get_plugin(WatchPlugin)
        return str(w)

    @safe_command
    def do_status(self, arg):
        self.tdb.message(self.tdb.status)

    @safe_command
    def do_mlir(self, arg):
        self.tdb.message(self.info_mlir(arg))

    @safe_command
    def do_asm(self, arg):
        self.tdb.message(self.info_asm(arg))

    @safe_command
    def do_loc(self, arg):
        self.tdb.message(self.info_loc(arg))

    @safe_command
    def do_reg(self, arg):
        print(self.info_reg(arg))

    @safe_command
    def do_buf(self, arg):
        print(self.info_buf(arg))

    @safe_command
    def do_break(self, arg):
        self.tdb.message(self.info_break(arg))

    do_b = do_break

    @safe_command
    def do_watch(self, arg):
        self.tdb.message(self.info_watch(arg))

    do_w = do_watch

    @safe_command
    def do_progress(self, arg):
        self.tdb.message(f"asm: {self.tdb.cmd_point} / {len(self.tdb.cmditer)}")
        index_plugin = self.tdb.get_plugin(
            FinalMlirIndexPlugin
        )  # type: FinalMlirIndexPlugin
        if index_plugin:
            try:
                index = index_plugin.get_locindex_by_atomic()
                self.tdb.message(f"mlir: {index} / {len(index_plugin.final_mlir.loc)}")
            except (IndexError, KeyError) as e:
                self.tdb.error(e)
                self.tdb.error(
                    f"{type(self.tdb.get_cmd()).__name__} cmd has no mlir context."
                )

    def emptyline(self, *args):
        self.do_asm("5")

    def default(self, line: str) -> None:
        self.do_asm(line)
