# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
from typing import List, Union, Dict, NamedTuple, Type
from functools import partial
import re
import pandas as pd
import numpy as np
from rich import print as pprint
import sys
import cmd
from enum import Enum
from .atomic_dialect import BModel2MLIR
from .target_common import MType, BaseTpuOp, CpuOp, CMDType
from .disassembler import BModel
from .final_mlir import FinalMlirIndex, CMD
from .target_1688.context import BM1688Context


class TdbStatus(Enum):
    # bmodel not loaded
    UNINIT = 0
    # bmodel is loaded but can not run directly
    # but some static function can be used
    IDLE = 1
    # can be run by n/c
    RUNNING = 2


class Display(NamedTuple):
    expr: str
    index: int
    enable: bool = True
    type: str = "eval"  # eval / line


class Displays:
    """Breakpoint manager"""

    def __init__(self) -> None:
        self.display: Dict[int, Display] = {}
        self.display_id = 1

    def __str__(self) -> str:
        table = [["expr", "index", "enable"]]

        for k, v in self.display.items():
            table.append([v.expr, v.index, v.enable])
        df = pd.DataFrame(table)
        return df.to_string(index=False, header=False)

    def add_display(self, expr: str, cond=None):
        display = Display(expr, index=self.display_id)
        self.display[self.display_id] = display
        self.display_id += 1
        return self.display_id - 1

    def delete_display(self, index: int):
        if index in self.display:
            return self.display.pop(index)
        return None

    def enable(self, index: Union[int, List[int]]):
        for i in index:
            if i in self.display:
                self.display[i].enable = True

    def disable(self, index: Union[int, List[int]]):
        for i in index:
            if i in self.display:
                self.display[i].enable = False

    @classmethod
    def get_instance(cls):
        if not hasattr(cls, "_instance"):
            cls._instance = cls()

        return cls._instance


def max_with_none(*args):
    args = [i for i in args if i is not None]
    if len(args) == 1:
        return args[0]
    if len(args) == 0:
        return 0

    return max(args)


def callback(name=None):
    def outer(func):
        def inner(self: "TdbCmdBackend", *args, **kwargs):
            call_name = name if name is not None else func.__name__
            call_name = call_name.replace("do_", "")

            self.plugins._call_loop(f"before_{func.__name__}")
            res = func(self, *args, **kwargs)
            self.plugins._call_loop(f"after_{func.__name__}")
            return res

        return inner

    return outer


class TdbCmdBackend(cmd.Cmd):
    ddr_size = 2**32

    def __init__(
        self,
        completekey="tab",
        stdin=None,
        stdout=None,
        bmodel_file: str = None,
        final_mlir_fn: str = None,
        tensor_loc_file: str = None,
        input_data_fn: str = None,
        reference_data_fn: List[str] = None,
        extra_plugins: List[str] = True,
        extra_check: List[str] = True,
    ):
        super().__init__(completekey, stdin, stdout)
        self.bmodel_file = bmodel_file
        self.final_mlir_fn = final_mlir_fn
        self.tensor_loc_file = tensor_loc_file
        self.input_data_fn = input_data_fn
        if reference_data_fn is None:
            reference_data_fn = []
        elif isinstance(reference_data_fn, str):
            reference_data_fn = [reference_data_fn]
        self.reference_data_fns = reference_data_fn
        self.status = TdbStatus.UNINIT

        # should be import locally to avoid circular import
        from .plugins.breakpoints import Breakpoints
        from .static_check import Checker

        self.checker = Checker(self)
        self.breakpoints = Breakpoints.get_instance()
        self.displays = Displays.get_instance()
        # self.displays.add_display("self.get_op()")
        # self.displays.add_display("self.info('mlir')")
        # self.displays.add_display("self.info('loc')")

        self.static_mode = False
        self.enable_message = True
        self.cmditer: List[Union[BaseTpuOp, CpuOp]]

        self.plugins = PluginCompact(self)
        # default plugins
        self.add_plugin("breakpoint")  # `break/b` command
        self.add_plugin("display")  # `display` command
        self.add_plugin("print")  # `print`` command
        self.add_plugin("info")  # `info`` command

        if len(reference_data_fn) > 0:
            self.add_plugin("data-check")

        for extra_plugin in extra_plugins:
            self.add_plugin(extra_plugin)

        self.add_plugin("static-check")
        self.extra_check = extra_check

        self.message(f"Load plugins: {self.plugins}")

    def _reset(self):
        self._load_bmodel()
        self._load_data()
        self.cmditer = self.atomic_mlir.create_cmdlist()
        self.cmd_point = 0
        # print(len(self.cmditer))
        self.status = TdbStatus.RUNNING
        self.static_mode = False
        self._build_index()
        self.plugins.after_load()

    def _load_bmodel(self):
        bmodel_file = self.bmodel_file
        if bmodel_file is None:
            raise Exception("Nothing to debug.")
        bmodel = BModel(bmodel_file)
        self.message(f"Load {bmodel_file}")
        context = bmodel.context
        self.bmodel = bmodel
        self.message(f"Load {context.device.name} backend")
        self.atomic_mlir = BModel2MLIR(bmodel)
        self.final_mlir = FinalMlirIndex(self.final_mlir_fn, self.tensor_loc_file)
        self.message(f"Build {self.final_mlir_fn} index")
        self.message(f"Decode bmodel back into atomic dialect")
        self.message(f"static_mode = {self.static_mode}")
        # self.final_mlir = ...
        # self.tensor_loc = ...

        self.runner = context.get_runner(self.ddr_size)
        self.LMEM = self.runner.LMEM
        self.DDR = self.runner.DDR
        self.context = context
        self.memory = context.memory
        self.decoder = context.decoder

        self.message(f"initialize memory")
        self.memory.clear_memory()
        coeff = self.atomic_mlir.functions[0].regions[0].data
        if coeff:
            address = coeff.address
            if isinstance(self.context, BM1688Context):
                address = self.context.fix_tag(address)
            addr = address - self.context.memmap[MType.G][0]
            # load constant data
            self.DDR[addr : addr + len(coeff.data)] = memoryview(coeff.data)

    def _load_data(self):
        file = self.input_data_fn
        if file is None:
            self.error(f"file {file} is invalid")
            return
        if file.endswith(".dat"):
            inputs = np.fromfile(file, dtype=np.uint8)
            _offset = 0

            for arg in self.atomic_mlir.functions[0].signature[0]:
                mem = arg.memref
                size = int(np.prod(mem.shape) * mem.itemsize)
                self.memory.set_data(
                    mem, inputs[_offset : _offset + size].view(mem.np_dtype)
                )

                _offset += size
        elif file.endswith(".npz"):
            inputs = np.load(file)
            self.set_inputs_dict(inputs)

    def _build_index(self):
        """ """
        self.cmd2index = {}
        # create subnet tiu, dma id offset
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

        for executed_id, op in enumerate(self.cmditer, start=1):
            # executed_id indicate the index after op execution, start is 1
            # cmd_point = 0 means no cmd is executed
            if isinstance(op, BaseTpuOp):
                self.cmd2index[op.tuple_key] = executed_id
            elif isinstance(op, CpuOp):
                self.cmd2index[("cpu", op.cmd_id, op.subnet_id)] = executed_id
            else:
                import pdb

                pdb.set_trace()

        self.before_index2loc = {}
        self.after_index2loc = {}

        self.index2loc_range = {}

        last_af_index = 0
        for loc_index, loc in enumerate(self.final_mlir.loc.tensor_loc):
            # before execution
            (
                subnet_id,
                tiu,
                dma,
                core_id,
            ) = (loc.subnet_id, *loc.tiu_dma_id_before, loc.core_id)
            x_index = y_index = None
            tiu_offset, dma_offset = subnet_offsets[subnet_id]

            if tiu - tiu_offset > 0:
                x_index = self.cmd2index[(subnet_id, tiu - tiu_offset, None, core_id)]
            if dma - dma_offset > 0:
                y_index = self.cmd2index[(subnet_id, None, dma - dma_offset, core_id)]

            bf_index = max_with_none(x_index, y_index, last_af_index + 1)

            # assert bf_index not in self.before_index2loc, bf_index
            self.before_index2loc.setdefault(bf_index, []).append(loc_index)

            # after execution
            (
                subnet_id,
                tiu,
                dma,
                core_id,
            ) = (loc.subnet_id, *loc.tiu_dma_id_after, loc.core_id)
            x_index = y_index = None

            if tiu - tiu_offset > 0:
                x_index = self.cmd2index[(subnet_id, tiu - tiu_offset, None, core_id)]
            if dma - dma_offset > 0:
                y_index = self.cmd2index[(subnet_id, None, dma - dma_offset, core_id)]

            last_af_index = af_index = max_with_none(x_index, y_index)
            # assert af_index not in self.after_index2loc, af_index
            self.after_index2loc.setdefault(af_index, []).append(loc_index)

            for index in range(bf_index, af_index + 1):
                # assert self.index2loc_range[index] == -1
                assert index not in self.index2loc_range, index
                # if index == 17:
                #     import pdb; pdb.set_trace()

                self.index2loc_range[index] = loc_index

    def add_plugin(self, plugin_name: str):
        plugin = self.plugins.add_plugin(plugin_name)

        if isinstance(plugin, cmd.Cmd):
            assert not hasattr(self, f"do_{plugin_name}"), plugin_name

            func_names = getattr(plugin, "func_names", [plugin_name])
            for func_name in func_names:
                setattr(self, f"do_{func_name}", plugin.onecmd)
                setattr(self, f"complete_{func_name}", plugin.completenames)
                setattr(self, f"help_{func_name}", partial(plugin.do_help, ""))

    def get_plugin(self, name) -> "TdbPlugin":
        return self.plugins[name]

    def set_inputs(self, *inputs):
        # args = self.file.module.functions[0].signature[0]
        args = self.atomic_mlir.functions[0].signature
        assert len(inputs) == len(args)
        for id, input in enumerate(inputs):
            self.set_input(id, input)

    def set_input(self, id, input):
        args = self.atomic_mlir.functions[0].signature[0]  # input_tensors
        mem = args[id].memref
        self.memory.set_data(mem, input)

    def get_names(self):
        # This method used to pull in base class attributes
        # at a time dir() didn't do it yet.
        return dir(self)

    def get_op_context(self, pre=2, next=2):
        pre = max(0, self.cmd_point - pre)
        return self.cmditer[pre : self.cmd_point + next]

    def get_op(self):
        if self.status == TdbStatus.UNINIT:
            raise StopIteration()

        if self.cmd_point >= len(self.cmditer):
            raise StopIteration()
        return self.cmditer[self.cmd_point]

    def get_preop(self):
        if self.cmd_point == 0:
            raise StopIteration()
        return self.cmditer[self.cmd_point - 1]

    def get_nextop(self):
        op = self.get_op()
        self.cmd_point += 1
        return op

    def get_mlir_by_atomic(self, op: BaseTpuOp):
        loc = self.get_loc_by_atomic(op)
        file_line = self.final_mlir.get_fileline_by_loc(loc.loc_name)
        return self.final_mlir.lines[file_line]

    def get_mlir_context_by_atomic(self, op: BaseTpuOp, pre=2, next=2) -> List[str]:
        loc = self.get_loc_by_atomic(op)
        file_line = self.final_mlir.get_fileline_by_loc(loc.loc_name)
        return self.final_mlir.lines[max(0, file_line - 1 - pre) : file_line - 1 + next]

    def get_loc_by_atomic(self, op: BaseTpuOp) -> CMD:
        index = self.cmd2index[op.tuple_key]
        loc_index = self.index2loc_range[index]
        return self.final_mlir.loc[loc_index]

    def get_loc_context_by_atomic(self, op: BaseTpuOp, pre=2, next=2) -> List[CMD]:
        index = self.cmd2index[op.tuple_key]
        loc_index = self.index2loc_range[index]

        return self.final_mlir.loc[max(0, loc_index - pre) : loc_index + next]

    def next(self):
        """
        every do_<func> used next() should catch BreakpointStop Exception
        and stop to wait user interaction
        """
        op = self.get_op()

        try:
            self.plugins.before_next(self)
        except BreakpointStop as e:
            raise e
        # clear breakpoint mark
        if not self.static_mode:
            cmd, cmd_type = op.cmd, op.cmd_type
            if not self.decoder.is_end(cmd):
                if cmd_type == CMDType.tiu:
                    self.runner.tiu_compute(cmd)
                elif cmd_type == CMDType.dma:
                    self.runner.dma_compute(cmd)
                elif cmd_type == CMDType.cpu:
                    self.runner.cpu_compute(cmd)
                else:
                    self.error("skip unknown CMDType")
                # elif self.decoder.is_dynamic(cmd):
                #     self.runner.dynamic_compute(cmd)

        try:
            self.plugins.after_next(self)
        except BreakpointStop as e:
            raise e

        self.get_nextop()

    def set_inputs_dict(self, inputs):
        args = self.atomic_mlir.functions[0].signature[0]
        from utils.lowering import lowering

        for id, arg in enumerate(args):  # type: List[int, tensor_cls]
            input = lowering(
                inputs[arg.name],
                pdtype=arg.dtype.name,
                pshape=arg.shape[0],
                pzero_point=arg.zero_point,
                pscale=arg.scale,
            )
            self.set_input(id, input)

    # cmd basic functions
    def message(self, msg):
        if self.enable_message:
            pprint(msg, file=self.stdout)

    def error(self, msg):
        if self.enable_message:
            pprint("***", msg, file=self.stdout)

    def _complete_expression(self, text, line, begidx, endidx):
        # Complete an arbitrary expression.
        # if not self.curframe:
        #     return []
        # Collect globals and locals.  It is usually not really sensible to also
        # complete builtins, and they clutter the namespace quite heavily, so we
        # leave them out.
        ns = {**sys._getframe().f_globals, **sys._getframe().f_locals}
        if "." in text:
            # Walk an attribute chain up to the last part, similar to what
            # rlcompleter does.  This will bail if any of the parts are not
            # simple attribute access, which is what we want.
            dotted = text.split(".")
            try:
                obj = ns[dotted[0]]
                for part in dotted[1:-1]:
                    obj = getattr(obj, part)
            except (KeyError, AttributeError):
                return []
            prefix = ".".join(dotted[:-1]) + "."
            return [prefix + n for n in dir(obj) if n.startswith(dotted[-1])]
        else:
            # Complete a simple name.
            return [n for n in ns.keys() if n.startswith(text)]

    def cmdloop(self, intro=None):
        """Repeatedly issue a prompt, accept input, parse an initial prefix
        off the received input, and dispatch to action methods, passing them
        the remainder of the line as argument.

        """

        self.preloop()
        if self.use_rawinput and self.completekey:
            try:
                import readline

                self.old_completer = readline.get_completer()
                readline.set_completer(self.complete)
                readline.parse_and_bind(self.completekey + ": complete")
            except ImportError:
                pass
        try:
            if intro is not None:
                self.intro = intro
            if self.intro:
                self.stdout.write(str(self.intro) + "\n")
            stop = None
            while not stop:
                if self.cmdqueue:
                    line = self.cmdqueue.pop(0)
                else:
                    if self.use_rawinput:
                        try:
                            line = input(self.prompt)
                        except KeyboardInterrupt:
                            line = ""
                            print(file=self.stdout)
                            continue
                        except EOFError:
                            line = "EOF"
                    else:
                        self.stdout.write(self.prompt)
                        self.stdout.flush()
                        line = self.stdin.readline()
                        if not len(line):
                            line = "EOF"
                        else:
                            line = line.rstrip("\r\n")
                line = self.precmd(line)
                stop = self.onecmd(line)
                stop = self.postcmd(stop, line)
            self.postloop()
        finally:
            if self.use_rawinput and self.completekey:
                try:
                    import readline

                    readline.set_completer(self.old_completer)
                except ImportError:
                    pass


class BreakpointStop(Exception):
    pass


class Breakpoint:
    type: str = None
    pattern: re.Pattern = None

    def __init__(self, text, cond=None, index=-1) -> None:
        self.enabled = True
        self.text = text
        self.cond = cond
        self.index = index

        self.hit_conut = 0

    def __init_subclass__(cls) -> None:
        breakpoint_cls.append(cls)

    def __str__(self) -> str:
        return "\t".join(self.tostrlist())

    def tostrlist(self) -> List[str]:
        enable_str = "y" if self.enabled else "n"
        return [f"{self.index}", self.type, enable_str, self.text, f"{self.hit_conut}"]

    def should_stop(self, tdb: TdbCmdBackend) -> bool:
        pass

    def toggle_enable(self, flag: bool):
        self.enabled = flag

    @classmethod
    def match_break(cls, text) -> bool:
        if isinstance(cls.pattern, re.Pattern):
            return cls.pattern.search(text) is not None
        return False


class TdbPlugin:
    """
    subclasses of class `TdbPlugin` can also extend cmd.Cmd,
    and implement do_{name} and complete_{name} to extend the functionality of tdb.
    """

    name = None
    func_names: List[str]

    def __init_subclass__(cls) -> None:
        register_plugins[cls.name] = cls
        # if issubclass(cmd.Cmd):
        #     cls.stdout = sys.stdout

    def __init__(self, tdb: TdbCmdBackend) -> None:
        self.tdb = tdb
        if isinstance(self, cmd.Cmd):
            self.stdout = sys.stdout

    def __str__(self) -> str:
        if isinstance(self, cmd.Cmd):
            return f"{self.name}*"
        return self.name

    @classmethod
    def from_tdb(cls, tdb: TdbCmdBackend):
        if not hasattr(cls, "_instance"):
            cls._instance = cls(tdb)

        return cls._instance

    def before_next(self, tdb: TdbCmdBackend):
        pass

    def after_stop(self, tdb: TdbCmdBackend):
        pass

    def after_next(self, tdb: TdbCmdBackend):
        pass

    def after_load(self, tdb: TdbCmdBackend):
        pass

    def after_end_execution(self, tdb: TdbCmdBackend):
        pass


class PluginCompact:
    name = "compact_list"

    def __init__(self, tdb: TdbCmdBackend) -> None:
        self.plugins: Dict[str, TdbPlugin] = {}
        self.tdb = tdb

    def __repr__(self) -> str:
        return f"{', '.join([str(i) for i in self.plugins.values()])}"

    def __getitem__(self, key):
        return self.plugins.get(key, None)

    def __contains__(self, arg):
        return arg in self.plugins

    def _call_loop(self, name: str):
        for plugin in self.plugins.values():
            getattr(plugin, name)(self.tdb)

    def before_next(self, _: TdbCmdBackend = None):
        self._call_loop(name="before_next")

    def after_next(self, _: TdbCmdBackend = None):
        self._call_loop(name="after_next")

    def after_stop(self, _: TdbCmdBackend = None):
        self._call_loop(name="after_stop")

    def after_load(self, _=None):
        self._call_loop(name="after_load")

    def after_end_execution(self):
        self._call_loop(name="after_end_execution")

    def add_plugin(self, plugin: Union[str, TdbPlugin]):
        if isinstance(plugin, str):
            try:
                plugin = register_plugins[plugin].from_tdb(self.tdb)
            except KeyError as e:
                raise KeyError(register_plugins.keys()) from e
        self.plugins[plugin.name] = plugin
        return plugin


breakpoint_cls: List[Type[Breakpoint]] = []
register_plugins: Dict[str, TdbPlugin] = {}
