#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
from debugger.tdb_support import (
    TdbCmdBackend,
    TdbStatus,
    BreakpointStop,
    commom_args,
    add_callback,
)
import os


class TdbInterface(TdbCmdBackend):
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

    def default(self, line):
        self.do_py(line)

    def do_status(self, arg):
        self.message(self.status)

    def do_py(self, arg):
        try:
            self.message(eval(arg))
        except SyntaxError:
            try:
                exec(arg)
                globals().update(locals())
            except BaseException as e:
                self.error(e)
        except BaseException as e:
            self.error(e)

    def complete_py(self, text, line, begidx, endidx):
        return self._complete_expression(text, line, begidx, endidx)

    @add_callback("skip")
    def do_skip(self, arg):
        try:
            val = int(arg)
            self.cmd_point += val
        except ValueError as e:
            self.error(e)

    @add_callback("end_execution", TdbStatus.END)
    @add_callback("stop")
    @add_callback("run")
    def do_run(self, _):
        """run from begining"""
        if not self.status.NO_CAND:
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

        self.status = TdbStatus.RUNNING
        while True:
            try:
                self.step()
            except (KeyboardInterrupt, BreakpointStop):
                self.status = TdbStatus.IDLE
                break
            except StopIteration:
                self.status = TdbStatus.END
                self.message("End of Execution")
                break

    do_r = do_run

    @add_callback("start")
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

    do_s = do_start

    @add_callback("end_execution", TdbStatus.END)
    @add_callback("stop")
    @add_callback("continue")
    def do_continue(self, arg):
        """continue running"""
        if self.status == TdbStatus.UNINIT:
            self.message("The program is not being run.")
            return

        self.status = TdbStatus.RUNNING

        while True:
            try:
                _ = self.step()
            except (BreakpointStop, KeyboardInterrupt):
                self.status = TdbStatus.IDLE
                break
            except StopIteration:
                self.status = TdbStatus.END
                self.message("End of Execution")
                break

    do_c = do_continue

    @add_callback("end_execution", TdbStatus.END)
    @add_callback("stop")
    @add_callback("next")
    def do_next(self, arg):
        if self.status == TdbStatus.UNINIT:
            self.message("The program is not being run.")
            return

        if arg == "":
            arg = 1
        else:
            try:
                arg = int(arg)
            except Exception:
                arg = 1

        for _ in range(arg):
            try:
                self.step()
            except (BreakpointStop, KeyboardInterrupt):
                break
            except StopIteration:
                self.status = TdbStatus.END
                self.message("End of Execution.")
                break

    do_n = do_next

    @add_callback("quit")
    def do_quit(self, arg):
        if self.status.NO_CAND:
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

    def do_plugin(self, arg):
        return self.message(self.plugins.plugins.keys())

    def postcmd(self, stop, line):
        """ignore return value of do_command"""
        pass


def parse_args(args=None):
    import argparse

    parser = argparse.ArgumentParser(description="TPU Debugger.")
    commom_args(parser)
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
        help="The reference data of the BModel.",
    )
    parser.add_argument(
        "--plugins",
        type=str,
        nargs="?",
        default=None,
        help="The extra plugins to be added.",
    )
    parser.add_argument(
        "--edit",
        action="store_true",
        help="to keep all intermediate files for debug",
    )

    parser.add_argument(
        "--quiet", action="store_true", default=False, help="disable progress bar"
    )

    return parser.parse_args(args)


def get_tdb(args=None):
    args = parse_args(args)
    context_dir = args.context_dir
    if os.path.isfile(context_dir) and context_dir.endswith(".bmodel"):
        bmodel_file = context_dir
        final_mlir_fn = None
        tensor_loc_file = None
    else:
        bmodel_file = os.path.join(context_dir, "compilation.bmodel")
        final_mlir_fn = os.path.join(context_dir, "final.mlir")
        tensor_loc_file = os.path.join(context_dir, "tensor_location.json")

    if final_mlir_fn is not None and (
        not os.path.exists(final_mlir_fn) or not os.path.exists(tensor_loc_file)
    ):
        final_mlir_fn = tensor_loc_file = None

    input_data_fn = args.inputs
    if input_data_fn is None and os.path.isdir(context_dir):
        input_data_fn = os.path.join(context_dir, "input_ref_data.dat")

    reference_data_fn = args.ref_data

    extra_plugins = args.plugins
    if extra_plugins is None:
        extra_plugins = []
    else:
        extra_plugins = extra_plugins.split(",")

    if not args.quiet:
        extra_plugins.append("progress")

    if args.edit:
        extra_plugins.append("edit-bmodel")

    tdb = TdbInterface(
        bmodel_file=bmodel_file,
        final_mlir_fn=final_mlir_fn,
        tensor_loc_file=tensor_loc_file,
        input_data_fn=input_data_fn,
        reference_data_fn=reference_data_fn,
        extra_plugins=extra_plugins,
        args=args,
    )
    return tdb


if __name__ == "__main__":
    tdb = get_tdb()
    tdb.cmdloop()
