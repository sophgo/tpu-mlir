#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
from debugger.target_common import MType
from debugger.tdb_support import TdbCmdBackend, TdbStatus, BreakpointStop, CMDType
import os
import cmd
from debugger.disassembler import (
    BModel,
)
from debugger.atomic_dialect import BModel2MLIR

import sys
import pprint
import code
import traceback
import numpy as np
from rich import print


class Tdb(cmd.Cmd):
    ddr_size = 2**32
    prompt = "(Tdb) "
    """
    TPU debugger.

    1. Use as an interactive tool:
    >> tdb.py the/path/of/bmodel/context

    2. Use as a package:
    from tdb import Tdb
    tdb = Tdb()
    tdb.load_bmodel("./onnx_test/AddConst_f32.bmodel")
    tdb.start()
    input = np.arange(1 * 16 * 28 * 28, dtype=np.float32)[::-1].reshape([1, 16, 28, 28])
    tdb.set_inputs(input)
    tdb.do_continue("")
    tdb.get_return()
    """

    def __init__(self, completekey="tab", stdin=None, stdout=None):
        cmd.Cmd.__init__(self, completekey, stdin, stdout)
        self.disassembler = None
        self.record_status = False
        self.milir_module = None
        self.current_function = None
        self.status = {}
        self.bmodel: BModel = None
        self.inputs = None
        self.enable_message = True
        self.current_line = -1
        self.breakpoint = []
        # The temporary breakpoint must be hit and then destroyed after it is used.
        self.temporary_breakpoint = False

    def __reset(self):
        self.runner.memory.clear_memory()
        self.status = {}
        self.current_line = -1
        self.current_function = None
        self._make_continue_iter()

    def load_bmodel(self, bmodel_file: str = ""):
        if bmodel_file is None:
            raise Exception("Nothing to debug.")
        bmodel = BModel(bmodel_file)
        context = bmodel.context
        self.bmodel = bmodel
        self.milir_module = BModel2MLIR(bmodel)
        self.runner = context.get_runner(Tdb.ddr_size)
        self.LMEM = self.runner.LMEM
        self.DDR = self.runner.DDR
        self.context = context
        self.memory = context.memory
        self.decoder = context.decoder

    def message(self, msg):
        if self.enable_message:
            print(msg, file=self.stdout)

    def error(self, msg):
        if self.enable_message:
            print("***", msg, file=self.stdout)

    def do_break(self, arg):
        if arg:
            try:
                self.breakpoint.append(int(arg))
                self.message(f"Add breakpoint at line: {arg}")
            except Exception:
                bb = [int(x) for x in arg.split(" ")]
                self.breakpoint.extend(bb)
                self.message(f"Add breakpoint at line: {bb}")
        else:
            self.message(f"{self.breakpoint}")

    do_b = do_break

    def clear_log(self, arg):
        import os

        os.system("clear")

    do_g = clear_log

    def do_clear(self, arg):
        if not arg:
            try:
                reply = input("Clear all breaks? ")
            except EOFError:
                reply = "no"
            reply = reply.strip().lower()
            if reply in ("y", "yes"):
                for bp in self.breakpoint:
                    self.message(f"Deleted {bp}")
                self.breakpoint = []
            return

    def print_line(self, offset=0, padding=0):
        line_num = self.current_line + offset
        try:
            self.message(
                f"[bold green]{line_num:{padding}} [/bold green] {self.get_op(offset)}"
            )
        except ValueError:
            self.message("Start.")

    def get_asm_context(self, offset=5):
        op_len = len(self.current_function.regions[0].blocks[0].operations)
        lines = self.current_line + np.arange(-offset, offset + 1)
        lines = lines[lines >= 0]
        lines = lines[lines < op_len]
        width = int(np.ceil(np.log10(lines.max())))  # get line number width
        msg = []
        for i in lines:
            ri = i - self.current_line  # relative line number
            if i == self.current_line and i in self.breakpoint:
                msg.append(f"[bold red]B+> {i:{width}} [/bold red] {self.get_op(ri)}")
            elif i == self.current_line:
                msg.append(f"[bold blue]--> {i:{width}} [/bold blue] {self.get_op(ri)}")
            elif i in self.breakpoint:
                msg.append(f"[bold red] BB {i:{width}} [/bold red] {self.get_op(ri)}")
            else:
                msg.append(
                    f"[bold green]{' '*4}{i:{width}} [/bold green] {self.get_op(ri)}"
                )
        return "\n".join(msg)

    def print_context(self, offset=5):
        self.message(self.get_asm_context(offset))

    def do_l(self, arg):
        """l(list)
        show +/-5 lines code around the current line.
        """
        if arg == "":
            arg = 5
        else:
            try:
                arg = int(arg)
                assert arg > 0
            except Exception:
                self.error(f"invalid input: {arg}. input should be a positive integer.")
                return
        self.print_context(arg)

    def do_ll(self, _):
        """list all
        show all the instruction in bmodel.
        """
        op_len = len(self.current_function.regions[0].blocks[0].operations)
        self.print_context(op_len)

    def do_next(self, _):
        """n(ext)
        Continue execution until the next line in the current function
        is reached or it returns.
        """
        try:
            self.print_line()
            self.next()
        except StopIteration:
            self.message("End of execution!")

    do_n = do_next

    def should_stop(self):
        if self.current_line in self.breakpoint:
            if self.temporary_breakpoint:
                self.breakpoint.remove(self.current_line)
            return True
        for f in filter(callable, self.breakpoint):
            if f():
                if self.temporary_breakpoint:
                    self.breakpoint.remove(f)
                self.message(f"Hit breakpoint: {f}")
                return True
        return False

    def _make_continue_iter(self):
        class continueIter:
            def __iter__(_self):
                return _self

            def __next__(_self):
                try:
                    if not self.temporary_breakpoint:
                        self.next()
                    while not self.should_stop():
                        self.next()
                except Exception:
                    raise StopIteration

        self.continues = continueIter()

    def do_continue(self, arg):
        """c(ontinue)
        Continue execution, only stop when a breakpoint is encountered.
        """
        try:
            next(self.continues)
            self.print_context()
        except StopIteration:
            self.message("End of execution.")

    do_c = do_continue

    def do_quit(self, arg):
        """q(uit)\nexit
        Quit from the debugger. The program being executed is aborted.
        """
        return True

    do_q = do_quit
    do_exit = do_quit

    def _getval(self, arg):
        try:
            return eval(arg)
        except Exception:
            exc_info = sys.exc_info()[:2]
            self.error(traceback.format_exception_only(*exc_info)[-1].strip())
            raise

    def do_p(self, arg):
        """p expression
        Print the value of the expression.
        """
        if not arg:
            self.print_line()
            return
        try:
            self.message(repr(self._getval(arg)))
        except Exception:
            pass

    def do_pp(self, arg):
        """pp expression
        Pretty-print the value of the expression.
        """
        try:
            self.message(pprint.pformat(self._getval(arg)))
        except Exception:
            pass

    def do_interact(self, arg):
        """interact

        Start an interactive interpreter whose global namespace
        contains all the (global and local) names found in the current scope.
        """
        code.interact("*interactive*")

    def do_help(self, arg):
        """h(elp)
        Without argument, print the list of available commands.
        With a command name as argument, print help about that command.
        "help pdb" shows the full pdb documentation.
        "help exec" gives help on the ! command.
        """
        if not arg:
            return cmd.Cmd.do_help(self, arg)
        try:
            try:
                topic = getattr(self, "help_" + arg)
                return topic()
            except AttributeError:
                command = getattr(self, "do_" + arg)
        except AttributeError:
            self.error("No help for %r" % arg)
        else:
            if sys.flags.optimize >= 2:
                self.error(
                    "No help for %r; please do not run Python with -OO "
                    "if you need command help" % arg
                )
                return
            self.message(command.__doc__.rstrip())

    do_h = do_help

    def load_data(self, file: str = ""):
        if file:
            self.inputs = file
        if file.endswith(".dat"):
            inputs = np.fromfile(file, dtype=np.uint8)
            _offset = 0
            for arg in self.current_function.signature[0]:
                mem = self.bmodel.tensor2memref(arg)
                size = int(np.prod(mem.shape) * mem.itemsize)
                self.memory.set_data(
                    mem, inputs[_offset : _offset + size].view(mem.np_dtype)
                )

                _offset += size
        elif file.endswith(".npz"):
            inputs = np.load(file)
            self.set_inputs_dict(inputs)

    def set_inputs_dict(self, inputs):
        args = self.current_function.signature[0]
        from utils.lowering import lowering

        for id, arg in enumerate(args):  # type: List[int, tensor_cls]
            input = lowering(
                inputs[arg.name],
                pdtype=arg.dtype.name,
                pshape=arg.shape,
                pzero_point=arg.zero_point,
                pscale=arg.scale,
            )
            self.set_input(id, input)

    def set_inputs(self, *inputs):
        # args = self.file.module.functions[0].signature[0]
        args = self.current_function.signature[0]
        assert len(inputs) == len(args)
        for id, input in enumerate(inputs):
            self.set_input(id, input)

    def set_input(self, id, input):
        args = self.current_function.signature[0]
        mem = self.bmodel.tensor2memref(args[id])
        self.memory.set_data(mem, input)

    def get_return(self):
        outputs = self.current_function.signature[1]
        mems = [self.bmodel.tensor2memref(x) for x in outputs]
        return [self.memory.get_data(mem) for mem in mems]

    def get_op(self, offset=0):
        ops = self.current_function.regions[0].blocks[0].operations
        line = self.current_line + offset
        if line >= len(ops):
            raise StopIteration("End of execution.")
        if line < 0:
            raise ValueError("Ahead of execution.")
        return ops[line]

    def get_all_ops(self):
        return self.milir_module.functions[0].regions[0].blocks[0].operations

    def push_status(self):
        if not self.record_status:
            return
        op = self.get_op()
        import pdb

        pdb.set_trace()

        self.status[self.current_line] = [x.data for x in op.results[0]]

    def pop_status(self):
        if not self.record_status:
            raise Exception("No records, can not go back.")

        op = self.get_op()
        if self.current_line not in self.status:
            raise Exception("can not go back.")
        data = self.status[self.current_line]
        for k, v in zip(op.results, data):
            k.data = v
        del self.status[self.current_line]

    def start(self):
        self.__reset()
        coeff = self.milir_module.functions[0].regions[0].data
        if coeff:
            address = coeff.address
            if self.context.device.name == "BM1688":
                address = self.context.opparam.MemRef.fix_tag(
                    address, self.context.base_addr
                )
            addr = address - self.context.memmap[MType.G][0]

            self.DDR[addr : addr + len(coeff.data)] = memoryview(coeff.data)
        if self.milir_module is None:
            raise Exception("please load one file.")
        self.current_function = self.milir_module.functions[0]

    def do_run(self, arg):
        """r(un)
        run from the beginning.
        """
        self.start()
        if self.inputs:
            self.load_data(self.inputs)
        self.do_continue("")

    do_r = do_run

    def next(self):
        """n(ext)
        run next instruction.
        """
        if self.current_line == -1:
            self.current_line += 1
            return
        self.push_status()
        try:
            op = self.get_op()
            cmd, cmd_type = op.cmd, op.cmd_type
            # sys = (self.context.dma_sys, self.context.tiu_sys)
            if not self.decoder.is_end(cmd):
                if cmd_type == CMDType.tiu:
                    self.runner.tiu_compute(cmd)
                elif cmd_type == CMDType.dma:
                    self.runner.dma_compute(cmd)
                elif cmd_type == CMDType.cpu:
                    self.runner.cpu_compute(cmd)
                else:
                    self.error("skip unknown CMDType")
            self.current_line += 1
        except ValueError as e:
            raise e

    def back(self):
        if self.current_line > 0:
            self.pop_status()
            self.current_line -= 1
        else:
            raise Exception("begin of execution.")


class TdbInterface(TdbCmdBackend):
    """
    do_break
    do_b
    do_g
    do_clear
    do_l
    do_ll
    do_next
    do_n
    do_continue
    do_c
    do_quit
    do_q
    do_exit
    do_p
    do_pp
    do_interact
    do_help
    do_h
    do_run
    do_r
    """

    ddr_size = 2**32
    prompt = "(tdb) "
    """
    TPU debugger.

    1. Use as an interactive tool:
    >> tdb.py the/path/of/bmodel/context

    2. Use as a package:
    from tdb import Tdb
    tdb = Tdb()
    tdb.load_bmodel("./onnx_test/AddConst_f32.bmodel")
    tdb.start()
    input = np.arange(1 * 16 * 28 * 28, dtype=np.float32)[::-1].reshape([1, 16, 28, 28])
    tdb.set_inputs(input)
    tdb.do_continue("")
    tdb.get_return()
    """

    def complete_check(self, text, line, begidx, endidx):
        lis = self.checker.check_list() + ["?"]
        return [i for i in lis if i.startswith(text)]

    def do_status(self, arg):
        self.message(self.status)

    def default(self, line):
        self.do_py(line)

    def do_py(self, arg):
        try:
            self.message(eval(arg))
        except BaseException as e:
            self.error(e)

    def complete_py(self, text, line, begidx, endidx):
        return self._complete_expression(text, line, begidx, endidx)

    def do_run(self, _):
        """run from begining"""
        if self.status == TdbStatus.RUNNING:
            try:
                res = input(
                    """The program being debugged has been started already.\nStart it from the beginning? (y or any)"""
                )
                if not res.strip().lower().startswith("y"):
                    self.message("Program not restarted.")
                    return False
            except KeyboardInterrupt:
                self.message("Cancel")
                return False
        self._reset()

        while True:
            try:
                self.next()
            except (KeyboardInterrupt, BreakpointStop, StopIteration):
                self.plugins.after_stop(self)
                break
        self.plugins.after_end_execution()

    do_r = do_run

    def do_start(self, arg):
        if self.status == TdbStatus.RUNNING:
            try:
                res = input(
                    """The program being debugged has been started already.\nStart it from the beginning? (y or any)"""
                )
                if not res.strip().lower().startswith("y"):
                    self.message("Program not restarted.")
                    return False
            except Exception:
                self.message("Quit")
                return False
        self._reset()
        self.plugins.after_stop()

    do_s = do_start

    def do_continue(self, arg):
        """continue running"""
        if self.status != TdbStatus.RUNNING:
            self.message("The program is not being run.")
            return

        while True:
            try:
                _ = self.next()
            except (BreakpointStop, KeyboardInterrupt):
                self.plugins.after_stop()
                break
            except StopIteration:
                self.plugins.after_stop()
                self.status = TdbStatus.IDLE
                self.message("End of Execution")
                break

        self.plugins.after_end_execution()

    do_c = do_continue

    def do_next(self, arg):
        if self.status != TdbStatus.RUNNING:
            self.message("The program is not being run.")
            return

        if arg == "":
            arg = 1
        else:
            try:
                arg = int(arg)
            except Exception:
                arg = 1

        for i in range(arg):
            try:
                self.next()
                self.plugins.after_stop()
            except (BreakpointStop, KeyboardInterrupt):
                self.plugins.after_stop()
                return
            except StopIteration:
                self.plugins.after_stop()
                self.status = TdbStatus.IDLE
                message = "End of Execution"
                self.message(message)
                return

    do_n = do_next

    def do_quit(self, arg):
        if self.status != TdbStatus.RUNNING:
            exit(0)
        try:
            res = input(
                """The program being debugged has been started already.\nQuit? (y or any)"""
            )
            if res.strip().lower().startswith("y"):
                exit(0)
        except EOFError:
            exit(0)
        except KeyboardInterrupt:
            self.message("Quit")
            return False

    do_q = do_quit
    do_EOF = do_quit

    def postcmd(self, stop, line):
        pass

    def do_plugin(self, arg):
        return self.message(self.plugins.plugins.keys())


def main():
    import argparse

    parser = argparse.ArgumentParser(description="TPU Debugger.")
    parser.add_argument(
        "context_dir",
        type=str,
        default="./",
        nargs="?",
        help="The path of BModel.",
    )
    parser.add_argument(
        "--inputs",
        type=str,
        nargs="?",
        default=None,
        help="The inputs data of the BModel.",
    )
    parser.add_argument(
        "--ref_data",
        nargs="*",
        type=str,
        # default=None,
        help="The inputs data of the BModel.",
    )
    parser.add_argument(
        "--plugins",
        type=str,
        nargs="?",
        default=None,
        help="The inputs data of the BModel.",
    )
    parser.add_argument(
        "--checks",
        type=str,
        nargs="?",
        default=None,
        help="The inputs data of the BModel.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", default=False)

    return parser.parse_args()


if __name__ == "__main__":
    args = main()
    context_dir = args.context_dir
    bmodel_file = os.path.join(context_dir, "compilation.bmodel")
    final_mlir_fn = os.path.join(context_dir, "final.mlir")
    tensor_loc_file = os.path.join(context_dir, "tensor_location.json")
    input_data_fn = args.inputs
    if input_data_fn is None:
        input_data_fn = os.path.join(context_dir, "input_ref_data.dat")
    reference_data_fn = args.ref_data

    # context_output_ref_fn = os.path.join(context_dir, "output_ref_data.dat")
    # if reference_data_fn is None and os.path.exists(context_output_ref_fn):
    #     reference_data_fn = context_output_ref_fn

    extra_plugins = args.plugins
    if extra_plugins is None:
        extra_plugins = []
    else:
        extra_plugins = extra_plugins.split(",")

    if args.verbose:
        extra_plugins.append("progress")

    extra_check = args.checks
    if extra_check is None:
        extra_check = []
    else:
        extra_check = extra_check.split(",")

    tdb = TdbInterface(
        bmodel_file=bmodel_file,
        final_mlir_fn=final_mlir_fn,
        tensor_loc_file=tensor_loc_file,
        input_data_fn=input_data_fn,
        reference_data_fn=reference_data_fn,
        extra_plugins=extra_plugins,
        extra_check=extra_check,
    )

    tdb.cmdloop()
