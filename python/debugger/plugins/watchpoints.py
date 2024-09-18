# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
import copy
from typing import List, Dict

import pandas as pd
import numpy as np

from ..final_mlir import TLValue

from ..target_common import CMDType, op_support
from debugger.tdb_support import (
    BreakpointStop,
    TdbCmdBackend,
    TdbPlugin,
    TdbPluginCmd,
    Watchpoint,
)


class WatchPlugin(TdbPlugin, TdbPluginCmd):
    name = "watch"
    func_names = ["watch", "w"]

    def __init__(self, tdb: "TdbCmdBackend") -> None:
        super().__init__(tdb)
        self.watches: Dict[tuple, Watchpoint] = {}
        self.tdb = tdb
        self.watch_id = 1
        self.watchid2value = {}
        tdb.watches = self.watches
        tdb.watchid2value = self.watchid2value

    def __str__(self) -> str:
        table = [["index", "cmd_type", "cmd_id", "core_id", "enabled", "value"]]

        for k, v in self.watches.items():
            table.append(v.tostrlist())
        df = pd.DataFrame(table)
        return df.to_string(index=False, header=False)

    def emptyline(self) -> bool:
        self.tdb.do_info("w")

    def _decode_watch_index(self, arg):
        if arg:
            try:
                arg = list(map(int, arg.split(",")))
            except ValueError:
                self.tdb.message(f"watch index must be type of int, got {type(arg)}")
                return []
        return arg

    def do_delete(self, arg):
        """stop watching a variable"""
        arg: [List, None] = self._decode_watch_index(arg)
        if arg:
            for i in arg:
                value = self.watchid2value[i][0]
                self.watches.pop(value)
                self.watchid2value.pop(i)
        else:
            self.watches.clear()
            self.watchid2value.clear()

    def do_enable(self, arg):
        """enable breakpoint"""
        arg = self._decode_watch_index(arg)
        for i in arg:
            value = self.watchid2value[i][0]
            self.watches[value].toggle_enable(True)
        return

    def do_disable(self, arg):
        """disable breakpoint"""
        arg = self._decode_watch_index(arg)
        for i in arg:
            value = self.watchid2value[i][0]
            self.watches[value].toggle_enable(False)
        return

    def _watch_in(self, value_index: str):
        try:
            cmd = self.tdb.get_cmd()
        except StopIteration:
            self.tdb.message("no cmd next.")
            return

        try:
            value_index = int(value_index)
            data = self.read_memref_data(cmd, cmd.operands[value_index])
            if data is None:
                self.tdb.error("Not supported net type.")
                return

            if get_tuple(cmd.core_id, cmd.operands[value_index]) not in self.watches:
                self.watches[get_tuple(cmd.core_id, cmd.operands[value_index])] = (
                    Watchpoint(
                        index=self.watch_id,
                        cmd_type=cmd.cmd_type,
                        cmd_id=cmd.cmd_id,
                        core_id=cmd.core_id,
                        value=cmd.operands[value_index],
                    )
                )
                self.watchid2value[self.watch_id] = [
                    get_tuple(cmd.core_id, cmd.operands[value_index]),
                    copy.deepcopy(data),
                ]
                self.watch_id += 1
            else:
                self.tdb.message(
                    f"The value {cmd.operands[value_index]} is already watching!"
                )
        except (IndexError, SyntaxError, ValueError) as e:
            self.tdb.error(e)

    def _watch_out(self, value_index: str):
        try:
            cmd = self.tdb.get_precmd()
        except StopIteration:
            self.tdb.message("no cmd pre.")
            return

        try:
            value_index = int(value_index)
            data = self.read_memref_data(cmd, cmd.results[value_index])
            if data is None:
                self.tdb.error("Not supported net type.")
                return

            if get_tuple(cmd.core_id, cmd.results[value_index]) not in self.watches:
                self.watches[get_tuple(cmd.core_id, cmd.results[value_index])] = (
                    Watchpoint(
                        index=self.watch_id,
                        cmd_type=cmd.cmd_type,
                        cmd_id=cmd.cmd_id,
                        core_id=cmd.core_id,
                        value=cmd.results[value_index],
                    )
                )
                self.watchid2value[self.watch_id] = [
                    get_tuple(cmd.core_id, cmd.results[value_index]),
                    copy.deepcopy(data),
                ]
                self.watch_id += 1
            else:
                self.tdb.message(
                    f"The value {cmd.results[value_index]} is already watching!"
                )
        except (IndexError, SyntaxError, ValueError) as e:
            self.tdb.error(e)

    def default(self, arg: str):
        """
        add variable to watch list
        Example:
            w in/out value_view_index
        """
        if arg == "":
            self.tdb.message("Please enter valid value.")
            return

        try:
            value_type, value_index = arg.split()
        except ValueError:
            self.tdb.message("Please input the valid index of operands or results")
            return

        if value_type == "in":
            self._watch_in(value_index)
            return
        elif value_type == "out":
            self._watch_out(value_index)
            return
        else:
            self.tdb.message("Please enter valid value_view type.")

    def after_step(self, tdb: "TdbCmdBackend"):
        for k, v in self.watchid2value.items():
            cur_watchpoint, old_value = v
            watchpoint: Watchpoint = self.watches[cur_watchpoint]
            if k and watchpoint.enabled:
                try:
                    data = self.read_memref_data(watchpoint, watchpoint.value)
                    if data is None:
                        self.tdb.error("")
                        return
                except Exception as e:
                    self.tdb.error(e)

                assert data is not None
                result = np.array_equal(old_value, data)
                if not result:
                    self.tdb.message(
                        f"\nwatchpoint index {k} has changed!\nold_value: \n{old_value}\nnow_value: \n{data}"
                    )
                    raise BreakpointStop()
        return

    def read_memref_data(self, cmd_or_watchpoint, value):
        if cmd_or_watchpoint.cmd_type == CMDType.cpu:
            if cmd_or_watchpoint.cmd_id == 0:
                data = self.tdb.memory.get_data(value.to_ref())
            else:
                data = self.tdb.memory.get_cpu_data(cmd_or_watchpoint.cmd_id)[value]
        elif cmd_or_watchpoint.cmd_type.is_static():
            if value.is_scalar:
                data = value.data
            else:
                data = self.tdb.memory.get_data(value.to_ref(core_id=cmd_or_watchpoint.core_id))
        else:
            return None
        return data


def get_tuple(core_id: int, value):
    if value.mtype in (op_support.MType.R, op_support.MType.S):
        return (core_id, value)
    else:
        return (0, value)
