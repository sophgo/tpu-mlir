# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

from typing import Any, List, Tuple
from collections import namedtuple
from .disassembler import (
    BModel,
    Net,
    Parameter,
    StaticCmdGroup,
    SubNet,
    Tensor,
    CmdGroup,
)

from .target_common import (
    CModelContext,
    BaseTpuOp,
    use_backend,
)
from .target_1688.context import BM1688Context
import functools
import textwrap

INDENT_SPACE = "  "


def BModel2MLIR(bmodel_net: BModel):
    with use_backend(bmodel_net.chip) as context:
        if isinstance(context, BM1688Context):
            coeff = bmodel_net.net[0].parameter[0].coeff_mem
            if coeff and context.base_addr[0] != context.base_addr[1]:
                context.base_addr[1] += len(coeff.data)

        with atomic_context(bmodel_net, context):
            atomic_mlir = MlirModule(bmodel_net)
    return atomic_mlir


def decode_cmdgroup(
    context: CModelContext, cmd_group: CmdGroup, subnet_id: int, core_id=0
) -> StaticCmdGroup:
    context = context
    decoder = context.decoder

    tiu = decoder.decode_tiu_cmds(cmd_group.tiu_cmd, core_id=core_id)
    dma = decoder.decode_dma_cmds(cmd_group.dma_cmd, core_id=core_id)

    cmdgroup = StaticCmdGroup(tiu, dma, context.merge_instruction(tiu, dma))
    # hack injection subnet_id
    for cmd in cmdgroup.all:
        cmd.subnet_id = subnet_id
    return cmdgroup


class _AtomicContext:
    def __call__(self, bmodel_net: BModel, cmodel_context: CModelContext) -> Any:
        self.bmodel_net = bmodel_net
        self.cmodel_context = cmodel_context
        return self

    def __enter__(self):
        pass

    def __exit__(self, *exc_info):
        self.bmodel_net = None
        self.cmodel_context = None


atomic_context = _AtomicContext()


class Node:
    def __init__(self) -> None:
        self.parent = None

    def with_parent(self, parent):
        self.parent = parent
        return self

    def __str__(self):
        raise NotImplementedError()


class Value(Node):
    def __init__(self, x: Tensor) -> None:
        super().__init__()
        self.memref = x.memref

    @property
    def name(self):
        return self.memref.name

    @property
    def type_str(self):
        return self.memref.type_str

    def __str__(self):
        return f"{self.name}: {self.type_str}"


class Block(Node):
    CMD = namedtuple("cmd", ["tiu", "dma", "all"])

    def __init__(self, subnet: SubNet, indent=0):
        super().__init__()
        self.subnet_id = subnet.id
        self.indent = indent
        self.operations: List[BaseTpuOp] = []

        bmodel_net = atomic_context.bmodel_net
        context = bmodel_net.context

        decoder = context.decoder
        decode_cmd_params = decoder.decode_cmd_params
        decode_cpu_params = decoder.decode_cpu_params

        self.run_mode = subnet.run_mode
        self.cmds = []
        self.cpu_cmds = []

        if subnet.run_mode == subnet.run_mode.CPU:
            # TODO
            self.cpu_cmds = [bmodel_net.decode_cpu_op(i) for i in subnet.cpu_params]
        elif subnet.run_mode == subnet.run_mode.TPU_DYNAMIC:
            # TODO
            self.ir_cmds = bmodel_net.decode_dynamic_ir(subnet.ir_buffer)
        elif bmodel_net.core_num > 1:
            self.cmds = [
                decode_cmdgroup(context, cmd, self.subnet_id, core_id)
                for core_id, x in enumerate(subnet.core_commands)
                for cmd in x.gdma_tiu_commands
            ]
        else:
            self.cmds = [
                decode_cmdgroup(context, x, self.subnet_id) for x in subnet.cmd_group
            ]

        if bmodel_net.core_num > 1:
            from .target_1688.multi_core import MultiCoreCmd

            sorter = MultiCoreCmd(
                [[decode_cmd_params(cmd) for cmd in group.all] for group in self.cmds]
            )
            self.cmds = sorter.consume_cmd()
            self.operations.extend(self.cmds)

        else:
            for x in self.cmds:
                for op in x.all:
                    self.operations.append(decode_cmd_params(op))

        for cpu_cmd_id, cpu_x in enumerate(self.cpu_cmds):
            # per cpuop, per subnet
            input_memref = [i.memref for i in subnet.input_tensor]
            output_memref = [i.memref for i in subnet.output_tensor]
            self.operations.append(
                decode_cpu_params(
                    op_type=cpu_x.op_type,
                    param=cpu_x.cpu_cmd,
                    input_memref=input_memref,
                    output_memref=output_memref,
                    subnet_id=subnet.id,
                    cmd_id=cpu_cmd_id,
                )
            )

        self.args = subnet.input_tensor
        self.terminator = subnet.output_tensor
        self.successor = subnet.next_subnet_ids

    @functools.lru_cache()
    def __str__(self):
        """
        ^bb0(%a: i64, %cond: i1): // Code dominated by ^bb0 may refer to %a
            cf.cond_br %cond, ^bb1, ^bb2
        """
        if self.run_mode == self.run_mode.CPU:
            ops_str = "\n".join([i.op_type.name for i in self.cpu_cmds])
        elif self.run_mode == self.run_mode.TPU_STATIC:
            ops_str = "\n".join((f"{x}" for x in self.operations))
        # elif self.run_mode == self.run_mode.TPU_DYNAMIC:
        else:
            ops_str = f"// not resovled yet for mode {self.run_mode.name}"

        comment = f"    //  run_mode={self.run_mode.name}"

        ops_str = textwrap.indent(ops_str, INDENT_SPACE)

        args = []
        for arg in self.args:
            value = Value(arg)
            args.append(str(value))
        args_str = ", ".join(args)

        if all((x == -1 for x in self.successor)):
            tem = [Value(x) for x in self.terminator]
            rets = (
                "return "
                + ", ".join((x.name for x in tem))
                + ": "
                + ", ".join((x.type_str for x in tem))
            )
        else:
            rets = f"Successor {self.successor}"  # TODO
        rets = textwrap.indent(rets, INDENT_SPACE)

        return f"^bb{self.subnet_id}({args_str}){comment}\n{ops_str}\n{rets}"


class Region(Node):
    def __init__(self, net_stage: Parameter, indent=0):
        super().__init__()
        self.indent = indent
        self.blocks = [Block(x, indent) for x in net_stage.sub_net]
        self.signature: Tuple[Value, Value] = (
            net_stage.input_tensor,
            net_stage.output_tensor,
        )
        self.data = net_stage.coeff_mem if net_stage.coeff_mem else None

    def __str__(self):
        blocks = "\n".join((f"{b}" for b in self.blocks))
        return f"{blocks}"


class Function(Node):
    def __init__(self, net: Net, indent=0):
        super().__init__()
        self.indent = indent
        self.name = net.name
        self.regions = [Region(x, indent) for x in net.parameter]

    @property
    def signature(self):
        return self.regions[0].signature

    def dump_head(self):
        #
        operands = ", ".join((str(Value(x)) for x in self.signature[0]))
        returns = ", ".join((Value(x).type_str for x in self.signature[1]))
        return f"func.func @{self.name}({operands}) -> ({returns}) ({{"

    def dump_tail(self):
        def fmt_names(x: List[Tensor]):
            names = (f'"{n.name}"' for n in x)
            return f"[{', '.join(names)}]"

        arg = f"arg_attrs = {fmt_names(self.signature[0])}"
        ret = f"res_attrs = {fmt_names(self.signature[1])}"
        attr = f"{{function_type = {{{arg}, {ret}}}}}"
        return f"}}) {attr}"

    def __str__(self):
        head = self.dump_head()
        tail = self.dump_tail()
        ops_str = ("\n}, {\n").join((str(r) for r in self.regions))
        ops_str = textwrap.indent(ops_str, INDENT_SPACE)
        return f"{head}\n{ops_str}\n{tail}"


class MlirModule(Node):
    def __init__(self, bmodel: BModel):
        super().__init__()
        self.bmodel = bmodel
        self.chip = bmodel.chip
        self.version = bmodel.version
        self.type = bmodel.type

        self.functions = [Function(x, 1) for x in self.bmodel.net]

    def create_cmdlist(self) -> List[BaseTpuOp]:
        res = []
        for func in self.functions:
            for region in func.regions:
                for block in region.blocks:
                    res.extend(block.operations)
        return res

    def dump_head(self):
        attrs = f'attributes {{chip = "{self.chip}", version = {self.version}}}'
        return f"module {attrs} {{"

    def dump_tail(self):
        return "} loc(#loc)"

    def __str__(self):
        head = self.dump_head()
        tail = self.dump_tail()
        func_str = "\n".join((f"{x}" for x in self.functions))
        func_str = textwrap.indent(func_str, INDENT_SPACE)
        return f"{head}\n{func_str}\n{tail}"
