#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
from bmodel_dis import Bmodel2MLIR, opdef_1684x, tensor2memref
from utils.bmodel_dis.opparam_1684x import MType, Memory, memmap
from utils.cmodel import lib, gen_lookup_table
import cmd
import ctypes
import traceback
import sys
import pprint
import numpy as np
import code

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


line_prefix = "\n-> "


class Tdb(cmd.Cmd):
    ddr_size = 2**30
    prompt = "(Tdb) "
    """
    TPU debuuger
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
        self.current_line = 0
        self.breakpoint = []
        self.inital_runtime()

    def __del__(self):
        self.close()

    def close(self):
        lib.cmodel_deinit(0)

    def reset(self):
        self.ddr[...] = 0
        self.lmem[...] = 0
        self.status = {}
        self.current_line = 0
        self.current_function = None
        self.breakpoint = []

    def inital_runtime(self):
        bind_compute()
        lib.cmodel_init(0, Tdb.ddr_size)
        self.ddr = c_array_to_ndarray(lib.get_global_memaddr(0), Tdb.ddr_size)
        self.lmem = c_array_to_ndarray(
            lib.get_local_mem(0).contents.raw_ptr, (64, 16, 1024 * 16)
        )
        self.smem = c_array_to_ndarray(lib.get_static_memaddr_by_node(0), (16 * 1024,))
        self.ddr[...] = 0
        self.lmem[...] = 0
        lut = np.array(gen_lookup_table(), np.uint32).view(np.uint8)
        self.smem[: len(lut)] = lut[...]
        self.mem_manager = Memory(self.lmem, self.ddr)

    def message(self, msg):
        print(msg, file=self.stdout)

    def error(self, msg):
        print("***", msg, file=self.stdout)

    def do_break(self, arg):
        self.breakpoint.append(int(arg))

    do_b = do_break

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

    def do_next(self, arg):
        """n(ext)
        Continue execution until the next line in the current function
        is reached or it returns.
        """
        self.next()
        return 1

    do_n = do_next

    def do_continue(self, arg):
        """c(ont(inue))
        Continue execution, only stop when a breakpoint is encountered.
        """
        try:
            self.next()
            while self.current_line not in self.breakpoint:
                self.next()
        except ValueError:
            pass

    do_c = do_cont = do_continue

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

    def load_file(self, file=None):
        if file:
            self.file = file
        if self.file == None:
            raise Exception("Nothing to debug.")

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
        if self.current_line + offset < len(ops):
            return ops[self.current_line + offset]
        raise ValueError("end of execution.")

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


# Usage:
# tdb = Tdb()
# tdb.load_file("./onnx_test/AddConst_f32.bmodel")
# tdb.start()
# input = np.arange(1 * 16 * 28 * 28, dtype=np.float32)[::-1].reshape([1, 16, 28, 28])
# tdb.set_inputs(input)
# tdb.next()
# tdb.next()
# tdb.next()
# tdb.next()
# tdb.get_return()

if __name__ == "__main__":
    Tdb().cmdloop()
