from ..tdb_support import TdbPlugin, TdbPluginCmd, codelike_format
from .common import FinalMlirIndexPlugin
from .breakpoints import BreakpointPlugin


class InfoPlugin(TdbPlugin, TdbPluginCmd):
    name = "info"

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
        op = self.tdb.get_op()
        res = (
            index_plugin.get_loc_context_by_atomic(op, pre, next)
            if multi_context
            else [index_plugin.get_loc_by_atomic(op)]
        )
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
        op = self.tdb.get_op()
        res = (
            index_plugin.get_mlir_context_by_atomic(op, pre, next)
            if multi_context
            else [index_plugin.get_mlir_by_atomic(op)]
        )
        message = codelike_format(res, index)
        return message

    def info_asm(self, arg):
        if self.status.NO_CAND:
            message = "The program is not being run."
            return message
        index, multi_context, pre, next = self.parse_comon(arg)
        res = (
            self.tdb.get_op_context(pre, next) if multi_context else [self.tdb.get_op()]
        )
        message = codelike_format(res, index)
        return message

    def info_reg(self, arg):
        if self.status.NO_CAND:
            message = "The program is not being run."
            return message
        index, multi_context, pre, next = self.parse_comon(arg)
        res = (
            self.tdb.get_op_context(pre, next) if multi_context else [self.tdb.get_op()]
        )
        res = [i.cmd for i in res]
        message = codelike_format(res, index)
        return message

    def info_buf(self, arg):
        if self.status.NO_CAND:
            message = "The program is not being run."
            return message
        index, multi_context, pre, next = self.parse_comon(arg)
        res = (
            self.tdb.get_op_context(pre, next) if multi_context else [self.tdb.get_op()]
        )
        res = [i.buf for i in res if i.cmd_type == i.cmd_type]
        message = codelike_format(res, index)
        return message

    def info_break(self, arg):
        b: BreakpointPlugin = self.tdb.get_plugin(BreakpointPlugin)
        return str(b.breakpoints)

    def do_status(self, arg):
        self.tdb.message(self.tdb.status)

    def do_mlir(self, arg):
        self.tdb.message(self.info_mlir(arg))

    def do_asm(self, arg):
        self.tdb.message(self.info_asm(arg))

    def do_loc(self, arg):
        self.tdb.message(self.info_loc(arg))

    def do_reg(self, arg):
        print(self.info_reg(arg))

    def do_buf(self, arg):
        print(self.info_buf(arg))

    def do_break(self, arg):
        self.tdb.message(self.info_break(arg))

    do_b = do_break

    def do_progress(self, arg):
        self.tdb.message(f"asm: {self.tdb.cmd_point} / {len(self.tdb.cmditer)}")
        index_plugin = self.tdb.get_plugin(
            FinalMlirIndexPlugin
        )  # type: FinalMlirIndexPlugin
        if index_plugin:
            index = index_plugin.get_locindex_by_atomic(self.tdb.get_op())
            self.tdb.message(f"mlir: {index} / {len(index_plugin.final_mlir.loc)}")
