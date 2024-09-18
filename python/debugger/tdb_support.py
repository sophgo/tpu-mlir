# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
from argparse import ArgumentParser, Namespace
import collections
from typing import List, Union, Dict, NamedTuple, Type, Optional
from functools import partial
import re
import os
import pandas as pd
import numpy as np
from rich import print as pprint, get_console
import sys
import cmd
from enum import Enum
from .atomic_dialect import BModel2MLIR
from .target_common import MType, BaseTpuCmd, CpuCmd, CMDType, DynIrCmd, StaticCmdGroup, CModelRunner
from .disassembler import BModel
from .target_1688.context import BM1688Context
from .target_1690.context import BM1690Context
from .target_2380.context import SG2380Context
from .target_mars3.context import MARS3Context
import pandas as pd
import builtins


class TdbStatus(Enum):
    # bmodel not loaded
    UNINIT = 0
    # interactive mode with bmodel loaded
    IDLE = 1
    # running in continue/run cmd
    RUNNING = 2
    # end of executation
    END = 3

    @property
    def NO_CAND(self):
        # no cmd to be executed
        return self == TdbStatus.UNINIT or self == TdbStatus.END


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


def add_callback(name=None, *filter):
    def outer(func):
        def inner(self: "TdbCmdBackend", *args, **kwargs):
            call_name = name if name is not None else func.__name__
            call_name = call_name.replace("do_", "")

            use_cb = len(filter) == 0 or self.status in filter
            skip = False
            try:
                if use_cb:
                    self.plugins._call_loop(f"before_{call_name}")
            except (KeyboardInterrupt, StopIteration) as e:
                self.status = TdbStatus.IDLE
                raise e
            except BufferError as e:
                skip = True

            if not skip:
                res = func(self, *args, **kwargs)
            else:
                res = None
                if call_name == 'step':
                    self.cmd_point += 1

            try:
                if use_cb:
                    self.plugins._call_loop(f"after_{call_name}")
            except (KeyboardInterrupt, StopIteration) as e:
                raise e
            return res

        return inner

    return outer


def safe_command(func):
    def outer(self, *args):
        try:
            func(self, *args)
        except Exception as e:
            self.tdb.error(e)

    return outer


def complete_file(self, text: str, line, begidx, endidx):
    text = text.split(" ")[-1]
    head = os.path.dirname(text)
    if len(head) != 0:
        head = f"{head}/"
    tail = os.path.basename(text)
    if head.strip() == "":
        res = []
        for f in os.listdir("."):
            if not f.startswith(tail):
                continue
            if os.path.isdir(f):
                res.append(f"{f}/")
            else:
                res.append(f)
        return res

    return [f"{f}" for f in os.listdir(head) if f.startswith(tail)]


def commom_args(parser: ArgumentParser):
    parser.add_argument(
        "context_dir",
        type=str,
        default="./",
        nargs="?",
        help="The path of BModel.",
    )
    parser.add_argument(
        "--cache_mode",
        choices=['online', 'generate', 'offline'],
        default='online',
    )
    parser.add_argument(
        "--run_by_atomic",
        action="store_true",
        default=False,
        help="whether run by atomic cmds, set False as default which means run_by_op. \
              This option only effect PCIE and SOC mode, disable this may decrease time cost,\
              but may have timeout error when an op contain too many atomic cmds. "                                                                                   ,
    )

    parser.add_argument(
        "--ddr_size",
        type=int,
        nargs="?",
        default=2**32,
        help="The ddr_size of cmodel.",
    )


class TdbCmdBackend(cmd.Cmd):
    def __init__(
        self,
        bmodel_file: str = None,
        final_mlir_fn: str = None,
        tensor_loc_file: str = None,
        input_data_fn: str = None,
        reference_data_fn: Optional[List[str]] = None,
        extra_plugins: List[str] = [],
        extra_check: List[str] = [],
        completekey="tab",
        stdin=None,
        stdout=None,
        is_soc=False,  # this option is valid only when USING_CMODEL = False
        args=None,
    ):
        super().__init__(completekey, stdin, stdout)
        if isinstance(args, Namespace):
            args = args.__dict__
        args = dict(args)
        assert args is not None
        self.bmodel_file = bmodel_file
        self.final_mlir_fn = final_mlir_fn
        self.tensor_loc_file = tensor_loc_file
        self.input_data_fn = input_data_fn
        if reference_data_fn is None:
            reference_data_fn = []
        elif isinstance(reference_data_fn, str):
            reference_data_fn = [reference_data_fn]
        self.reference_data_fns = reference_data_fn
        self.ddr_size: int = args.get("ddr_size", 2**32)
        self.status = TdbStatus.UNINIT
        builtins.tdb = self
        self.run_by_atomic: bool = args.get("run_by_atomic", False)
        self.fast_checker = not self.run_by_atomic
        self.fast_checker_enabled = False
        self.is_soc = is_soc
        self.cache_mode: str = args.get("cache_mode", "online")
        self.args = args
        # should be import locally to avoid circular import
        from .static_check import Checker


        # initialize registered plugins
        from . import plugins as _

        self.checker = Checker(self)
        self.displays = Displays.get_instance()

        self.static_mode = False
        self.enable_message = True
        self.cmditer: List[Union[BaseTpuCmd, CpuCmd, DynIrCmd]]

        self.plugins = PluginCompact(self)
        # default plugins
        self.add_plugin("breakpoint")  # `break/b` command
        self.add_plugin("display")  # `display` command
        self.add_plugin("print")  # `print`` command
        self.add_plugin("info")  # `info`` command
        self.add_plugin("final-mlir")
        self.add_plugin("reload")  # `reload` command
        self.add_plugin("watch")

        if len(reference_data_fn) > 0:
            self.add_plugin("data-check")

        for extra_plugin in extra_plugins:
            self.add_plugin(extra_plugin)

        self.add_plugin("static-check")
        self.extra_check = extra_check

        self.message(f"Load plugins: {self.plugins}")
        self.message(
            f"Type `s` to initialize bmodel, type `r` to run full commands, or `help` to see other commands."
        )

    @add_callback("load")
    def _reset(self):
        self._load_cmds()
        self._load_runner()
        self._load_data()

        self.status = TdbStatus.IDLE
        self.static_mode = False
        self._build_index()

        if self.get_plugin("progress") is None:
            self.message(
                "You are in quiet mode, add `-v/--verbose` argument to open prograss bar,\n"
                "or use `info progress` to show progress information."
            )

    def _load_cmds(self):
        bmodel_file = self.bmodel_file
        if bmodel_file is None:
            raise Exception("Nothing to debug.")
        bmodel = BModel(bmodel_file)
        self.message(f"Load {bmodel_file}")
        self.context = bmodel.context
        self.bmodel = bmodel
        self.message(f"Load {self.context.device.name} backend")
        self.atomic_mlir = BModel2MLIR(bmodel)
        self.cmditer = self.atomic_mlir.create_cmdlist()
        self.cmd_point = 0

    def _load_runner(self):
        # cmd_point point at the cmd that to be executed (but not)
        # executed_id
        context = self.context
        self.message(f"Build {self.final_mlir_fn} index")
        self.message(f"Decode bmodel back into atomic dialect")
        self.message(f"static_mode = {self.static_mode}")

        self.runner = context.get_runner(self.ddr_size)
        self.message(f"use {self.runner}")
        self.context = context

        if self.fast_checker and hasattr(self.runner, "fast_checker"):
            self.runner.fast_checker = True
            self.fast_checker_enabled = True

        if self.is_soc:
            return

        if hasattr(self.runner, "use_pcie"):
            self.runner.use_pcie()

        self.memory = context.memory
        self.decoder = context.decoder

        self.message(f"initialize memory")
        self.memory.clear_memory()
        coeff = self.atomic_mlir.functions[0].regions[0].data
        self.memory.set_neuron_size(self.bmodel.neuron_size)
        coeff_data = np.frombuffer(coeff.data, dtype=np.uint8)
        self.memory.set_coeff_size(coeff_data.size)

        if coeff:
            address = coeff.address
            if isinstance(self.context, BM1688Context) or isinstance(
                self.context, BM1690Context) or isinstance(
                self.context, SG2380Context) or isinstance(
                self.context, MARS3Context):
                address = self.context.fix_addr(address)
            addr_offset_ddr = address - self.context.memmap[MType.G][0]
            # load constant data
            if self.runner.using_cmodel:
                self.LMEM = self.runner.LMEM
                self.SMEM = self.runner.SMEM
                self.DDR = self.runner.DDR
                self.DDR[addr_offset_ddr : addr_offset_ddr + len(coeff.data)] = (
                    memoryview(coeff.data)
                )
            else:
                self.memory.set_data_to_address(
                    coeff.address, coeff_data
                )

    def _load_data(self):
        if self.is_soc:
            return
        file = self.input_data_fn
        if file is None or (isinstance(file, str) and not os.path.isfile(file)):
            self.error(f"input data file `{file}` is invalid")
            return

        if isinstance(file, dict):
            self.set_inputs_dict(file)
        elif file.endswith(".dat"):
            inputs = np.fromfile(file, dtype=np.uint8)
            _offset = 0

            for arg in self.atomic_mlir.functions[0].signature[0]:
                mem = arg.memref
                size = int(np.prod(mem.shape) * mem.itemsize)
                self.memory.set_data(
                    mem, inputs[_offset : _offset + size].view(mem.np_dtype)
                )  #  load input tensor

                _offset += size
        elif file.endswith(".npz"):
            inputs = np.load(file)
            self.set_inputs_dict(inputs)

    def _build_index(self):
        """
        build a pandas DataFrame with following columns:
            executed-id, subnet-id, core-id, cmd-id, cmd-id-dep, cmd-type, cmd-index
                                                                  dyn
                                                                  static
                                                                  cpu
                                                     maybe None

        cmd-index == op.tuple_key
        TODO
        """
        cmd_records = []
        self.tiu_max = collections.defaultdict(lambda: 0)
        self.dma_max = collections.defaultdict(lambda: 0)
        for executed_id, op in enumerate(self.cmditer, start=1):
            record = {
                "cmd_index": op.tuple_key,
                "executed_id": executed_id,
                "subnet_id": op.subnet_id,
                "core_id": op.core_id,
                "cmd_id": op.cmd_id,
                # "cmd_id_dep": None,
                "cmd_type": op.cmd_type,
                "op_name": op.name,
                # "is_sys": False,
            }
            cmd_records.append(record)
            if op.cmd_type == CMDType.tiu:
                self.tiu_max[op.core_id] += 1
            elif op.cmd_type == CMDType.dma:
                self.dma_max[op.core_id] += 1

        op_df = pd.DataFrame.from_records(cmd_records)
        op_df["executed_id"] = op_df["executed_id"].astype(int)
        self.op_df = op_df.set_index("cmd_index", drop=False)

    def add_plugin(self, plugin_name: str):
        plugin = self.plugins.add_plugin(plugin_name)

        if isinstance(plugin, TdbPluginCmd):
            assert not hasattr(self, f"do_{plugin_name}"), plugin_name

            func_names = getattr(plugin, "func_names", [plugin_name])
            for func_name in func_names:
                setattr(self, f"do_{func_name}", plugin.onecmd)
                setattr(self, f"complete_{func_name}", plugin.complete_plugin)
                setattr(self, f"help_{func_name}", partial(plugin.do_help, ""))

    def get_plugin(self, name: Union[str, Type["TdbPlugin"]]) -> "TdbPlugin":
        """
        if plugin not registed, return None for result.

        self.plugins is a instance of class PluginCompact which implement __getitem__,
        not dict
        """
        if isinstance(name, str):
            return self.plugins[name]
        else:
            return self.plugins[name.name]

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

    def get_op_context(self, pre=2, next=2, cmd_point=None):
        if cmd_point is None:
            cmd_point = self.cmd_point
        pre = max(0, cmd_point - pre)
        return self.cmditer[pre:cmd_point + next + 1]

    def get_cmd(self):
        # This method return next cmd which is about to execute but not executed yet
        if self.status == TdbStatus.UNINIT:
            raise StopIteration()

        if self.cmd_point >= len(self.cmditer):
            raise StopIteration()
        return self.cmditer[self.cmd_point]

    def get_precmd(self):
        # This method return the pre cmd which is just executed
        if self.cmd_point == 0:
            raise StopIteration()
        return self.cmditer[self.cmd_point - 1]

    def get_stack_cmds(self, cmd_point=None, cmditer=None):
        if cmditer is None:
            cmditer = self.cmditer
        if cmd_point is None:
            cmd_point = self.cmd_point

        tiu = []
        dma = []
        all = []

        if self.plugins.has_plugin('final-mlir'):
            index_plugin: "FinalMlirIndexPlugin" = self.plugins['final-mlir']
            df = index_plugin.data_frame
        else:
            df = None

        offset = 0

        while cmd_point + offset < len(cmditer):
            cur = cmditer[cmd_point + offset]
            if cmditer[cmd_point + offset].cmd_type == CMDType.tiu:
                tiu.append(cmditer[cmd_point + offset])
            else:
                dma.append(cmditer[cmd_point + offset])
            all.append(cur)

            if not self.fast_checker_enabled:
                break

            if isinstance(self.runner, CModelRunner):
                break

            if self.run_by_atomic:
                break

            if df is not None:
                result_not_empty = (len(df.loc[df["executed_id"] == cmd_point + 1 + offset,
                                               "results"].tolist()[0]) > 0)
            else:
                break

            if result_not_empty:
                break
            offset += 1

        return StaticCmdGroup(tiu, dma, all)

    @add_callback("step")
    def step(self):
        """
        every do_<func> used next() should catch BreakpointStop Exception
        and stop to wait user interaction
        """
        cmd = self.get_cmd()
        forward_cmds = 1
        try:
            if not self.static_mode:
                cmd_type = cmd.cmd_type
                if cmd_type.is_static():
                    if not self.context.is_sys(cmd):
                        cmds = self.get_stack_cmds()
                        forward_cmds = len(cmds.all)
                        self.runner.cmds_compute(cmds)
                elif cmd_type == CMDType.cpu:
                    self.runner.cpu_compute(cmd)
                elif cmd_type == CMDType.dyn_ir:
                    self.runner.dynamic_compute(cmd)
                else:
                    self.error("skip unknown CMDType")
        except ValueError as e:
            self.error(e)
            raise BreakpointStop()
        self.cmd_point += forward_cmds

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
            if isinstance(msg, Exception):
                get_console().print_exception(show_locals=True)
            else:
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
        self.ignore = 0

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
    def match_break(cls, text, tdb: TdbCmdBackend) -> bool:
        if isinstance(cls.pattern, re.Pattern):
            return cls.pattern.search(text) is not None
        return False


class Watchpoint:

    def __init__(self, index, cmd_type, cmd_id, core_id, value):
        self.enabled = True
        self.index = index
        self.cmd_type = cmd_type
        self.cmd_id = cmd_id
        self.core_id = core_id
        self.value = value

    def __str__(self) -> str:
        return "\t".join(self.tostrlist())

    def tostrlist(self) -> List[str]:
        enable_str = "y" if self.enabled else "n"
        return [
            str(self.index),
            str(self.cmd_type),
            str(self.cmd_id),
            str(self.core_id),
            enable_str,
            str(self.value),
        ]

    def toggle_enable(self, flag: bool):
        self.enabled = flag


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
            return "{}*".format(self.name)
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

    def after_step(self, tdb: TdbCmdBackend):
        pass

    def after_load(self, tdb: TdbCmdBackend):
        pass

    def after_end_execution(self, tdb: TdbCmdBackend):
        pass


class TdbPluginCmd(cmd.Cmd):

    def complete_plugin(self, text="", line="", begidx=0, endidx=0) -> List[str]:
        i, n = 0, len(line)
        while i < n and line[i] in self.identchars:
            i = i + 1
        i += 1
        line = line[i:]
        endidx -= i
        begidx -= i

        if begidx > 0:
            cmd, text, foo = self.parseline(line)
            if cmd == "":
                compfunc = self.completedefault
            else:
                try:
                    compfunc = getattr(self, "complete_" + cmd)
                except AttributeError:
                    compfunc = self.completedefault
        else:
            compfunc = self.completenames
        completion_matches = compfunc(text, line, begidx, endidx)
        return completion_matches


class PluginCompact:
    name = "compact_list"

    def __init__(self, tdb: TdbCmdBackend) -> None:
        self.plugins: Dict[str, TdbPlugin] = {}
        self.tdb = tdb

    def __repr__(self) -> str:
        return ", ".join([str(i) for i in self.plugins.values()])

    def __getitem__(self, key):
        return self.plugins.get(key, None)

    def __contains__(self, arg):
        return arg in self.plugins

    def _call_loop(self, name: str):
        for plugin in self.plugins.values():
            fn = getattr(plugin, name, None)
            if fn is not None:
                fn(self.tdb)

    def before_next(self, _: TdbCmdBackend = None):
        self._call_loop(name="before_next")

    def after_step(self, _: TdbCmdBackend = None):
        self._call_loop(name="after_step")

    def after_stop(self, _: TdbCmdBackend = None):
        self._call_loop(name="after_stop")

    def after_load(self, _=None):
        self._call_loop(name="after_load")

    def after_end_execution(self):
        self._call_loop(name="after_end_execution")

    def has_plugin(self, plugin_name: str):
        return plugin_name in self.plugins

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


def codelike_format(res: list, index=None):
    messages = []
    for i, c in enumerate(res):
        if i == index:
            c = f" \33[1;31m=>\033[0m {c}"
        else:
            c = f"    {c}"
        messages.append(c)
    lis_message = "\n".join(messages)
    return lis_message
