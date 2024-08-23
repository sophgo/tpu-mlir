# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

from typing import List, Union, Dict
import re

from debugger.tdb_support import (
    TdbCmdBackend,
    TdbPlugin,
    TdbPluginCmd,
    Breakpoint,
    BreakpointStop,
    breakpoint_cls,
)
import pandas as pd
from ..target_common import CMDType
from .common import FinalMlirIndexPlugin


class AddrBreakpoint(Breakpoint):
    """
    address + [rw]

    - r for read(operand)
    - w for write(results)
    - by default, all read and write action are checked

    b G29323264

    RSLG for local/static/L2 SRAM/DDR memory,
    see op_support.MType for details

    TODO: support address range, or add interrupt in numpy array
    """

    type = "addr"
    pattern = re.compile("^([RSLG])([0-9]+)([rw]{0,2})$")

    def __init__(self, text, cond=None, index=-1) -> None:
        super().__init__(text, cond, index)
        self.mtype, self.address, self.mode = self.pattern.findall(text)[0]
        self.address = int(self.address)
        if self.mode == "":
            self.mode = "rw"

    def should_stop(self, tdb: TdbCmdBackend) -> bool:
        cmd = tdb.get_cmd()
        checked_values = []
        if "r" in self.mode:
            checked_values.extend(cmd.operands)
        elif "w" in self.mode:
            checked_values.extend(cmd.results)

        for cmd in checked_values:
            if cmd.is_scalar:
                continue
            if cmd.mtype.name != self.mtype:
                continue
            if self.address == cmd.r_addr:
                return True

        return False


class CmdIdBreakpoint(Breakpoint):
    type = "cmd-id"
    pattern = re.compile("^[BD][0-9]+")

    def __init__(self, text, cond=None, index=-1) -> None:
        super().__init__(text, cond, index)
        self.match_type = CMDType.tiu if text[0] == "T" else CMDType.dma
        self.match_index = int(text[1:])

    def should_stop(self, tdb: "TdbCmdBackend") -> bool:
        cmd = tdb.get_cmd()

        if self.match_type != cmd.cmd_type:
            return False

        if cmd.cmd_id != self.match_index:
            return False

        return True


class LocBreakpoint(Breakpoint):
    """final.mlir location"""

    type = "loc"
    pattern = re.compile(r"^#loc\([\w]+\)")


class MlirFileLineBreakpoint(Breakpoint):
    type = "file-line(op)"
    pattern = re.compile("^:[0-9]+")

    def __init__(self, text, cond=None, index=-1) -> None:
        super().__init__(text, cond, index)
        self.file_line = int(text[1:])

    def should_stop(self, tdb: TdbCmdBackend) -> bool:
        index: FinalMlirIndexPlugin = tdb.get_plugin(FinalMlirIndexPlugin)
        loc = index.get_loc_by_point(tdb.cmd_point)

        if loc is None:
            return False

        return loc.file_line == self.file_line


class CmdFileLineBreakpoint(Breakpoint):
    """
    set breakpoint for bmodel_dis.py
    """

    type = "file-line(cmd)"
    pattern = re.compile("^::[0-9]+")

    def __init__(self, text, cond=None, index=-1) -> None:
        super().__init__(text, cond, index)
        self.file_line = int(text[2:])


class ValueIdBreakpoint(Breakpoint):
    type = "value-id"
    pattern = re.compile("^%[0-9]+")

    def should_stop(self, tdb: TdbCmdBackend) -> bool:
        index = tdb.get_plugin(FinalMlirIndexPlugin)  # type: FinalMlirIndexPlugin
        if not index.enabled:
            return False

        mlir = index.get_mlir_by_point(tdb.cmd_point)
        if mlir is None:
            return False
        if mlir.find(self.text) >= 0:
            return True
        return False


class DialectOpBreakpoint(Breakpoint):
    type = "dialect"
    pattern = re.compile(r"^(tpu|top)\.\w+")

    def should_stop(self, tdb: TdbCmdBackend) -> bool:
        index = tdb.get_plugin(FinalMlirIndexPlugin)  # type: FinalMlirIndexPlugin
        if not index.enabled:
            return False

        mlir = index.get_mlir_by_point(tdb.cmd_point)
        if mlir is None:
            return False
        if mlir.find(self.text) >= 0:
            return True
        return False


class ASM1684NameBreakpoint(Breakpoint):
    type = "1684-asm"
    pattern = re.compile(r"^\w+")

    @classmethod
    def match_break(cls, text, tdb: TdbCmdBackend) -> bool:
        from ..target_1684.regdef import op_class_dic

        if text in op_class_dic:
            return True
        return False


class ASM1684XNameBreakpoint(Breakpoint):
    type = "1684x-asm"
    pattern = re.compile(r"^\w+")

    @classmethod
    def match_break(cls, text, tdb: TdbCmdBackend) -> bool:
        from ..target_1684x.regdef import op_class_dic

        if text in op_class_dic:
            return True
        return False


class ASM1688NameBreakpoint(Breakpoint):
    type = "1688-asm"
    pattern = re.compile(r"^\w+")

    @classmethod
    def match_break(cls, text, tdb: TdbCmdBackend) -> bool:
        from ..target_1688.regdef import op_class_dic

        if text in op_class_dic:
            return True
        return False


class Breakpoints:
    """Breakpoint manager"""

    def __init__(self, tdb: TdbCmdBackend) -> None:
        self.breaks: Dict[int, Breakpoint] = {}
        self.break_id = 1
        self.tdb = tdb

    def __str__(self) -> str:
        table = [["index", "type", "enable", "text", "hit"]]

        for k, v in self.breaks.items():
            table.append(v.tostrlist())
        df = pd.DataFrame(table)
        return df.to_string(index=False, header=False)

    def _clear(self):
        for b in self.breaks.values():
            b.hit_conut = 0

    def add_break(self, text, cond=None):
        for i in breakpoint_cls:
            if i.match_break(text, self):
                breakpoint = i(text, cond, index=self.break_id)
                self.breaks[self.break_id] = breakpoint
                self.break_id += 1
                return breakpoint

    def delete_break(self, index: Union[int, List[int]]):
        if isinstance(index, int):
            index = [index]
        for i in index:
            if i in self.breaks:
                return self.breaks.pop(index)
        return None

    def should_break(self, tdb: "TdbCmdBackend"):
        for _, v in self.breaks.items():
            if v.enabled and v.should_stop(tdb):
                if v.ignore > 0:
                    v.ignore -= 1
                else:
                    v.hit_conut += 1
                    return v
        return None

    def enable(self, index: Union[int, List[int]]):
        if isinstance(index, int):
            index = [index]
        for i in index:
            if i in self.breaks:
                self.breaks[i].toggle_enable(True)

    def disable(self, index: Union[int, List[int]]):
        if isinstance(index, int):
            index = [index]
        for i in index:
            if i in self.breaks:
                self.breaks[i].toggle_enable(False)

    @classmethod
    def supported_patterns(cls):
        return [f"{i.pattern}({i.type})" for i in breakpoint_cls]


class BreakpointPlugin(TdbPlugin, TdbPluginCmd):
    name = "breakpoint"
    func_names = ["break", "b"]

    def __init__(self, tdb: "TdbCmdBackend") -> None:
        super().__init__(tdb)
        self.breakpoints = Breakpoints(tdb)
        self.stoped_op = None
        tdb.do_delete = self.do_delete
        tdb.do_enable = self.do_enable
        tdb.do_disable = self.do_disable

    def _decode_breakpoint_index(self, arg):
        try:
            arg = list(map(int, arg.split(",")))
        except ValueError:
            self.tdb.message(f"breakpoint index must be type of int, got {type(arg)}")
            return []
        return arg

    def do_delete(self, arg):
        """delete breakpoint"""
        arg = self._decode_breakpoint_index(arg)
        return self.breakpoints.delete_break(arg)

    def do_enable(self, arg):
        """enable breakpoint"""
        arg = self._decode_breakpoint_index(arg)
        return self.breakpoints.enable(arg)

    def do_disable(self, arg):
        """disable breakpoint"""
        arg = self._decode_breakpoint_index(arg)
        return self.breakpoints.disable(arg)

    def emptyline(self) -> bool:
        self.tdb.do_info("b")

    def default(self, arg: str):
        """
        stop before compute the operation with breakpoint

        add breakpoint for
         - loc/tensor name: b #loc2
            - pattern: #loc[0-9]+
         - address: b 4295028736
            - pattern: int
         - address range: b 4295028736-4295028736
            - pattern: [0-9]+-[0-9]+
         - file-line: b :24
            - pattern: :[0-9]+
         - cmd-id: b D23 / b B16
            - pattern: [BD][0-9]+
         - value-id: %173
            - pattern: ^%[0-9]+

         - tpu-dialect op name: tpu.Load, tpu.Conv2D
            - pattern: (tpu|top)
         - tsk_typ/tsk_eu_typ: dma.tensor, sConv, sG
            - other case

        TODO: support analyse break condition
        """
        text = arg.split()[0]
        break_res = self.breakpoints.add_break(text.strip())
        if break_res is not None:
            self.tdb.message(f"{break_res}")
        else:
            self.tdb.error(
                f"Supported Break patterns: {', '.join(self.breakpoints.supported_patterns())}"
            )
        return break_res

    def after_load(self, tdb: TdbCmdBackend):
        self.breakpoints._clear()

    def before_step(
        self,
        tdb: "TdbCmdBackend",
    ):
        cmd = tdb.get_cmd()
        if cmd == self.stoped_op:
            self.stoped_op = None
            return
        break_hit = self.breakpoints.should_break(tdb)
        if break_hit:
            self.stoped_op = cmd
            tdb.message(f"Hit: {break_hit}")
            raise BreakpointStop(break_hit)

    def after_step(
        self,
        tdb: "TdbCmdBackend",
    ):
        return
