import cmd
from ..tdb_support import TdbPlugin, TdbStatus


class InfoPlugin(TdbPlugin, cmd.Cmd):
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

    def codelike_format(self, res: list, index=None):
        messages = []
        for i, c in enumerate(res):
            if i == index:
                c = f" => {c}"
            else:
                c = f"    {c}"
            messages.append(c)
        lis_message = "\n".join(messages)
        return lis_message

    @property
    def status(self):
        return self.tdb.status

    def info_loc(self, arg):
        if self.status != TdbStatus.RUNNING:
            message = "The program is not being run."
            return message
        index, multi_context, pre, next = self.parse_comon(arg)
        op = self.tdb.get_op()
        res = (
            self.tdb.get_loc_context_by_atomic(op, pre, next)
            if multi_context
            else [self.tdb.get_loc_by_atomic(op)]
        )
        message = self.codelike_format(res, index)
        return message

    def info_mlir(self, arg):
        if self.status != TdbStatus.RUNNING:
            message = "The program is not being run."
            return message
        index, multi_context, pre, next = self.parse_comon(arg)
        op = self.tdb.get_op()
        res = (
            self.tdb.get_mlir_context_by_atomic(op, pre, next)
            if multi_context
            else [self.tdb.get_mlir_by_atomic(op)]
        )
        message = self.codelike_format(res, index)
        return message

    def info_asm(self, arg):
        if self.status != TdbStatus.RUNNING:
            message = "The program is not being run."
            return message
        index, multi_context, pre, next = self.parse_comon(arg)
        res = (
            self.tdb.get_op_context(pre, next) if multi_context else [self.tdb.get_op()]
        )
        message = self.codelike_format(res, index)
        return message

    def info_reg(self, arg):
        if self.status != TdbStatus.RUNNING:
            message = "The program is not being run."
            return message
        index, multi_context, pre, next = self.parse_comon(arg)
        res = (
            self.tdb.get_op_context(pre, next) if multi_context else [self.tdb.get_op()]
        )
        res = [i.cmd for i in res]
        message = self.codelike_format(res, index)
        return message

    def info_buf(self, arg):
        if self.status != TdbStatus.RUNNING:
            message = "The program is not being run."
            return message
        index, multi_context, pre, next = self.parse_comon(arg)
        res = (
            self.tdb.get_op_context(pre, next) if multi_context else [self.tdb.get_op()]
        )
        res = [i.buf for i in res if i.cmd_type == i.cmd_type]
        message = self.codelike_format(res, index)
        return message

    def info_break(self, arg):
        return str(self.tdb.breakpoints)

    def do_status(self, arg):
        self.tdb.message(self.tdb.status)

    def do_mlir(self, arg):
        self.tdb.message(self.info_mlir(arg))

    def do_asm(self, arg):
        self.tdb.message(self.info_asm(arg))

    def do_loc(self, arg):
        self.tdb.message(self.info_loc(arg))

    def do_reg(self, arg):
        self.tdb.message(self.info_reg(arg))

    def do_buf(self, arg):
        self.tdb.message(self.info_buf(arg))

    def do_break(self, arg):
        self.tdb.message(self.info_break(arg))

    do_b = do_break
