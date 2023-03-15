#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
from torch.nn.modules import linear
from bmodel_dis import Bmodel2MLIR, opdef_1684x, tensor2memref
from utils.bmodel_dis.opparam_1684x import MType, Memory, memmap
from utils.cmodel import lib, gen_lookup_table
import cmd, sys, pprint, code, traceback, ctypes
import numpy as np
from rich import print
from os import path

# common cases
def bind_compute():
    """
    ENGINE_BD   = 0,
    ENGINE_GDMA = 1,
    ENGINE_GDE  = 2,
    ENGINE_SORT = 3,
    ENGINE_NMS  = 4,
    ENGINE_CDMA = 5,
    """
    # TODO
    # atomic_sort
    # atomic_gde
    # atomic_nms
    def call_ins(command, engine_type):
        return lib.execute_command(
            0,
            np.packbits(
                command.reshape(-1, 8),
                axis=-1,
                bitorder="little",
            ).ctypes.data,
            engine_type,
        )

    def bdc_compute(cls):
        return call_ins(cls.cmd, 0)

    def gdma_compute(cls):
        return call_ins(cls.cmd, 1)

    for _, v in opdef_1684x.bdc_cmd.items():
        for op in v:
            setattr(op, "compute", bdc_compute)

    for _, v in opdef_1684x.dma_cmd.items():
        for op in v:
            setattr(op, "compute", gdma_compute)


# provide load input and bmodel to Global memory
# input should support assignment


# lib.cmodel_multi_thread_cxt_deinit(0)
# breakpoint


def c_array_to_ndarray(x, shape):
    if isinstance(x, int):
        x = ctypes.c_void_p(x)
    if isinstance(shape, int):
        shape = (shape,)
    try:
        p = ctypes.cast(x, ctypes.POINTER(ctypes.c_uint8))
    except:
        raise Exception(f"unsupported memory access: {x}")
    finally:
        return np.ctypeslib.as_array(p, shape=shape)


class Tdb(cmd.Cmd):
    ddr_size = 2**30
    prompt = "(Tdb) "
    """
    TPU debugger.

    1. Use as an interactive tool:
    >> tdb.py the/path/of/bmodel/context

    2. Use as a package:
    from tdb import Tdb
    tdb = Tdb()
    tdb.load_file("./onnx_test/AddConst_f32.bmodel")
    tdb.start()
    input = np.arange(1 * 16 * 28 * 28, dtype=np.float32)[::-1].reshape([1, 16, 28, 28])
    tdb.set_inputs(input)
    tdb.do_continue("")
    tdb.get_return()
    """

    def __init__(self, completekey="tab", stdin=None, stdout=None):
        cmd.Cmd.__init__(self, completekey, stdin, stdout)
        self.ddr = np.ndarray([])
        self.lmem = np.ndarray([])
        self.record_status = False
        self.model = None
        self.current_function = None
        self.status = {}
        self.file = None
        self.inputs = None
        self.current_line = 0
        self.breakpoint = []
        self.inital_runtime()

    def __del__(self):
        self.close()

    def close(self):
        lib.cmodel_deinit(0)

    def reset(self):
        self.ddr.fill(0)
        self.lmem.fill(0)
        self.status = {}
        self.current_line = 0
        self.current_function = None
        # self.breakpoint = []

    def inital_runtime(self):
        bind_compute()
        lib.cmodel_init(0, Tdb.ddr_size)
        self.ddr = c_array_to_ndarray(lib.get_global_memaddr(0), Tdb.ddr_size)
        self.lmem = c_array_to_ndarray(
            lib.get_local_mem(0).contents.raw_ptr, (64, 16, 1024 * 16)
        )
        self.smem = c_array_to_ndarray(lib.get_static_memaddr_by_node(0), (16 * 1024,))
        self.ddr.fill(0)
        self.lmem.fill(0)
        lut = np.array(gen_lookup_table(), np.uint32).view(np.uint8)
        self.smem[: len(lut)] = lut[...]
        self.mem_manager = Memory(self.lmem, self.ddr)

    def message(self, msg):
        print(msg, file=self.stdout)

    def error(self, msg):
        print("***", msg, file=self.stdout)

    def do_break(self, arg):
        if arg:
            try:
                self.breakpoint.append(int(arg))
                self.message(f"Add breakpoint at line: {arg}")
            except:
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
        self.message(
            f"[bold green]{line_num:{padding}} [/bold green] {self.get_op(offset)}"
        )

    def print_context(self, offset=5):
        op_len = len(self.current_function.regions[0].blocks[0].operations)
        lines = self.current_line + np.arange(-offset, offset)
        lines = lines[lines >= 0]
        lines = lines[lines < op_len]
        width = int(np.ceil(np.log10(lines.max())))
        for i in lines:
            if i == self.current_line and i in self.breakpoint:
                self.message(f"[bold red]->B {i:{width}} [/bold red] {self.get_op(i)}")
            elif i == self.current_line:
                self.message(
                    f"[bold blue]--> {i:{width}} [/bold blue] {self.get_op(i)}"
                )
            elif i in self.breakpoint:
                self.message(f"[bold red] BB {i:{width}} [/bold red] {self.get_op(i)}")
            else:
                self.print_line(i - self.current_line, width + 4)

    def do_l(self, _):
        """l(list)
        show +/-5 lines code around the current line.
        """
        self.print_context()

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
        except ValueError:
            print("End of execution!")

    do_n = do_next

    def do_continue(self, arg):
        """c(ontinue)
        Continue execution, only stop when a breakpoint is encountered.
        """
        try:
            self.get_op()
        except ValueError:
            self.message("End of execution!")
        try:
            self.next()
            while self.current_line not in self.breakpoint:
                self.next()
            if self.current_line in self.breakpoint:
                self.print_context()
        except ValueError:
            pass

    do_c = do_continue

    def do_quit(self, arg):
        """q(uit)\nexit
        Quit from the debugger. The program being executed is aborted.
        """
        self.close()
        return True

    do_q = do_quit
    do_exit = do_quit

    def _getval(self, arg):
        try:
            return eval(arg)
        except:
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
        except:
            pass

    def do_pp(self, arg):
        """pp expression
        Pretty-print the value of the expression.
        """
        try:
            self.message(pprint.pformat(self._getval(arg)))
        except:
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

    def load_file(self, file: str = ""):
        if file == None:
            raise Exception("Nothing to debug.")
        self.file = file

    def load_data(self, file: str = ""):
        if file:
            self.inputs = file
        if file.endswith(".dat"):
            inputs = np.fromfile(file, dtype=np.uint8)
            _offset = 0
            for arg in self.current_function.signature[0]:
                mem = tensor2memref(arg)
                assert mem.mtype == MType.G
                offset = mem.mtype.r_addr
                size = int(np.prod(mem.shape) * mem.itemsize)
                self.ddr[offset : offset + size] = inputs[_offset : _offset + size]
                _offset += size
        elif file.endswith(".npz"):
            inputs = np.load(file)
            self.set_inputs(*inputs.value())

    def set_inputs(self, *inputs):
        # args = self.file.module.functions[0].signature[0]
        args = self.current_function.signature[0]
        assert len(inputs) == len(args)
        for id, input in enumerate(inputs):
            self.set_input(id, input)

    def set_input(self, id, input):
        args = self.current_function.signature[0]
        mem = tensor2memref(args[id])
        self.set_data(mem, input)

    def set_data(self, des, src: np.ndarray):
        m_type = des.mtype
        if m_type == MType.G:
            offset = m_type.r_addr
            assert src.dtype == des.np_dtype
            src_u8 = np.ascontiguousarray(src.flatten()).view(np.uint8)
            self.ddr[offset : offset + src_u8.size] = src_u8.flatten()
        if m_type == "L":
            raise Exception("Not implemented.")

    def get_data(self, mem):
        return self.mem_manager.get_data(mem)

    def get_return(self):
        outputs = self.current_function.signature[1]
        mems = [tensor2memref(x) for x in outputs]
        return [self.get_data(mem) for mem in mems]

    def get_op(self, offset=0):
        ops = self.current_function.regions[0].blocks[0].operations
        line = self.current_line + offset
        if line >= len(ops):
            raise ValueError("end of execution.")
        if line < 0:
            raise ValueError("ahead of execution.")
        return ops[line]

    def push_status(self):
        if not self.record_status:
            return
        op = self.get_op()
        self.status[self.current_line] = self.get_data(op.results[0])

    def pop_status(self):
        if not self.record_status:
            raise Exception("No records, can not go back.")

        op = self.get_op()
        if self.current_line not in self.status:
            raise Exception("can not go back.")
        status = self.status[self.current_line]
        if isinstance(op, opdef_1684x.dma_base):
            self.set_data(op.results[0], status)
        elif isinstance(op, opdef_1684x.bdc_base):
            self.lmem[...] = status[...]
        del self.status[self.current_line]

    def start(self):
        self.reset()
        self.model = Bmodel2MLIR(self.file)
        coeff = self.model.module.functions[0].regions[0].data
        if coeff:
            addr = coeff.address - memmap[MType.G][0]
            # load constant data
            self.ddr[addr : addr + len(coeff.data)] = memoryview(coeff.data)
        if self.model is None:
            raise Exception("please load one file.")
        self.current_function = self.model.module.functions[0]

    def do_restart(self, arg):
        self.start()
        if self.inputs:
            self.load_data(args.inputs)
        print("restart debugger.")

    do_r = do_restart

    def next(self):
        self.push_status()
        try:
            op = self.get_op()
            op.compute()
            self.current_line += 1
        except ValueError as e:
            raise e

    def back(self):
        if self.current_line > 0:
            self.pop_status()
            self.current_line -= 1
        else:
            raise Exception("begin of execution.")


def __main():
    import argparse

    parser = argparse.ArgumentParser(description="TPU Debugger.")
    parser.add_argument(
        "bmodel",
        type=str,
        nargs="?",
        help="The path of BModel.",
    )
    parser.add_argument(
        "inputs",
        type=str,
        nargs="?",
        help="The inputs data of the BModel.",
    )
    return parser.parse_args()


if __name__ == "__main__":

    args = __main()
    tdb = Tdb()
    if args.bmodel:
        if path.isfile(args.bmodel):
            print(f"load bmodel: {args.inputs}")
            tdb.load_file(args.bmodel)
            tdb.start()
        elif path.isdir(args.bmodel):
            inputs = path.join(args.bmodel, "input_ref_data.dat")
            bmodel = path.join(args.bmodel, "compilation.bmodel")
            assert path.isfile(bmodel)
            assert path.isfile(inputs)
            assert args.inputs == None
            args.inputs = inputs
            print(f"load bmodel: {args.inputs}")
            tdb.load_file(bmodel)
            tdb.start()
    if args.inputs:
        print(f"load input: {args.inputs}")
        tdb.load_data(args.inputs)

    # Some handy functions for calling in interactive mode.
    def ins(index):
        """
        p ins(0)
        """
        op = tdb.get_op()
        value = op.operands[index]
        print(value)
        return tdb.get_data(value)

    def outs(index):
        """
        p outs(0)
        """
        op = tdb.get_op()
        value = op.results[index]
        print(value)
        return tdb.get_data(value)

    class op:
        def __init__(self, index=0) -> None:
            self.index = index

        def __repr__(self) -> str:
            print(tdb.get_op(self.index))

        def ins(self, index):
            op = tdb.get_op(self.index)
            value = op.operands[index]
            print(value)
            return tdb.get_data(value)

        def ous(self, index):
            op = tdb.get_op(self.index)
            value = op.results[index]
            print(value)
            return tdb.get_data(value)

    tdb.cmdloop()
