# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

from enum import Enum
from typing import Any, List, Tuple
from collections import namedtuple
from .disassembler import (
    BModel,
    Net,
    Parameter,

    SubNet,
    Tensor,
    CmdGroup,
)

from .target_common import (
    BModelContext,
    BaseTpuCmd,
    BaseCmd,
    use_backend,
    StaticCmdGroup,
)
from .target_1684.context import BM1684Context
from .target_1688.context import BM1688Context
from .target_1690.context import BM1690Context
from .target_2380.context import SG2380Context
from .target_mars3.context import MARS3Context
import functools
import textwrap

INDENT_SPACE = "  "


def BModel2MLIR(bmodel_net: BModel):
    with use_backend(bmodel_net.chip) as context:
        if isinstance(context, BM1688Context) or isinstance(context, SG2380Context) or isinstance(context, MARS3Context):
            coeff = bmodel_net.net[0].parameter[0].coeff_mem
            if coeff and context.base_addr[0] != context.base_addr[1]:
                context.base_addr[1] += len(coeff.data)

        with atomic_context(bmodel_net, context):
            atomic_mlir = MlirModule(bmodel_net)
    return atomic_mlir


def decode_cmdgroup(
    context: BModelContext,
    cmd_group: CmdGroup,
    subnet_id: int,
    core_id=0,
) -> StaticCmdGroup:
    decoder = context.decoder

    tiu = decoder.decode_tiu_cmds(
        cmd_group.tiu_cmd.bytes,
        core_id=core_id,
        subnet_id=subnet_id,
    )
    dma = decoder.decode_dma_cmds(
        cmd_group.dma_cmd.bytes,
        core_id=core_id,
        subnet_id=subnet_id,
    )

    cmdgroup = StaticCmdGroup(tiu, dma, context.merge_instruction(tiu, dma))
    return cmdgroup


class _AtomicContext:
    def __call__(self, bmodel_net: BModel, bmodel_context: BModelContext) -> Any:
        self.bmodel_net = bmodel_net
        self.bmodel_context = bmodel_context
        return self

    def __enter__(self):
        pass

    def __exit__(self, *exc_info):
        self.bmodel_net = None
        self.bmodel_context = None


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

    def __init__(self, subnet: SubNet, indent=0, ctx_addr=0, ctx_size=0):
        super().__init__()
        self.subnet = subnet
        self.subnet_id = subnet.id
        self.indent = indent
        self.operations: List[BaseCmd] = []

        bmodel_net = atomic_context.bmodel_net
        context = atomic_context.bmodel_context

        decoder = context.decoder

        # used for __str__ for better represent multi core cmd list
        self.group_by_core = False
        self.run_mode = subnet.run_mode
        self.cmds = []
        self.cpu_cmds = []
        self.ir_cmds = []

        input_memref = [i.memref for i in subnet.input_tensor]
        output_memref = [i.memref for i in subnet.output_tensor]
        self.args = subnet.input_tensor
        self.terminator = subnet.output_tensor
        self.successor = subnet.next_subnet_ids

        if subnet.run_mode == subnet.run_mode.CPU:
            self.cpu_cmds.extend(
                [bmodel_net.decode_cpu_op(i) for i in subnet.cpu_param]
            )
            for cpu_cmd_id, cpu_x in enumerate(self.cpu_cmds):
                # per cpuop, per subnet
                self.operations.append(
                    decoder.decode_cpu_cmd(
                        op_type=cpu_x.op_type,
                        buf=cpu_x.cpu_cmd,
                        input_memref=input_memref,
                        output_memref=output_memref,
                        subnet_id=subnet.id,
                        cmd_id=cpu_cmd_id,
                    )
                )
            return

        if subnet.run_mode == subnet.run_mode.TPU_DYNAMIC:
            self.ir_cmds.extend(bmodel_net.decode_dynamic_ir(subnet.ir_buffer))
            for ir_cmd_id, x in enumerate(self.ir_cmds):
                self.operations.append(
                    decoder.decode_ir_cmd(
                        subnet.ir_buffer,
                        subnet.ir_len,
                        input_memref=input_memref,
                        output_memref=output_memref,
                        subnet_id=subnet.id,
                        cmd_id=ir_cmd_id,
                    )
                )
            return

        if subnet.run_mode == subnet.run_mode.TPU_STATIC:
            if bmodel_net.core_num > 1:
                self.cmds = [
                    decode_cmdgroup(context, cmd, self.subnet_id, core_id)
                    for core_id, x in enumerate(subnet.core_commands)
                    for cmd in x.gdma_tiu_commands
                ]

                if isinstance(context, BM1690Context):
                    from .target_1690.multi_core import MultiCore, MsgCore
                elif isinstance(context, BM1688Context):
                    from .target_1688.multi_core import MultiCore, MsgCore
                elif isinstance(context, SG2380Context):
                    from .target_2380.multi_core import MultiCore, MsgCore
                core_nums = len(self.cmds)
                self.cores_cmds = [
                    MultiCore(core_id, core_nums, core_cmds.all, indent)
                    for core_id, core_cmds in enumerate(self.cmds)
                ]

                # resort print order
                msgcore_nums = len(self.cores_cmds[0].msgcores)
                msgcore_id = 0
                # insert into operations
                self.group_by_core = True
                self.core_operations: List[MsgCore] = []
                while msgcore_id < msgcore_nums:
                    for core_id, core_cmds in enumerate(self.cores_cmds):
                        if core_cmds.msgcores and msgcore_id <= len(core_cmds.msgcores):
                            self.core_operations.append(core_cmds.msgcores[msgcore_id])
                    msgcore_id += 1

                for msgcore in self.core_operations:
                    if msgcore.no_sys_cmds:
                        self.operations.extend(msgcore.no_sys_cmds)
                    self.operations.extend(msgcore.sys_cmds)
                return

            if subnet.cmd_group:
                self.cmds = [
                    decode_cmdgroup(context, x, self.subnet_id)
                    for x in subnet.cmd_group
                ]

                if isinstance(context, BM1684Context):
                    # tricky make cmd_id of cmd groups after first but in the same subnet add offset from previou cmd_id
                    if len(self.cmds[0].tiu) > 0:
                        tiu_offset = self.cmds[0].tiu[-1].cmd_id
                    if len(self.cmds[0].dma) > 0:
                        dma_offset = self.cmds[0].dma[-1].cmd_id

                    for cmd_group in self.cmds[1:]:
                        for cmd in cmd_group.tiu:
                            cmd.cmd_id += tiu_offset
                            cmd.cmd_id_dep += dma_offset

                        for cmd in cmd_group.dma:
                            cmd.cmd_id += dma_offset
                            cmd.cmd_id_dep += tiu_offset

                        tiu_offset = cmd.cmd_id
                        dma_offset = cmd.cmd_id
            else:
                self.cmds = [
                    decode_cmdgroup(context, cmd, self.subnet_id, core_id)
                    for core_id, x in enumerate(subnet.core_commands)
                    for cmd in x.gdma_tiu_commands
                ]

            for x in self.cmds:
                self.operations.extend(x.all)

    @functools.lru_cache()
    def __str__(self):
        """
        ^bb0(%a: i64, %cond: i1): // Code dominated by ^bb0 may refer to %a
            cf.cond_br %cond, ^bb1, ^bb2
        """
        if self.run_mode == self.run_mode.CPU:
            ops_str = "\n".join([i.op_type.name for i in self.cpu_cmds])
        elif self.run_mode == self.run_mode.TPU_STATIC:
            if self.group_by_core:
                ops_str = "\n".join((f"{x}" for x in self.core_operations))
            else:
                ops_str = "\n".join((f"{x}" for x in self.operations))
        elif self.run_mode == self.run_mode.TPU_DYNAMIC:
            ops_str = "\n".join((f"{x}" for x in self.operations))
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
        self.ctx_addr = net_stage.ctx_addr
        self.ctx_size = net_stage.ctx_size
        self.blocks = [
            Block(x, indent, self.ctx_addr, self.ctx_size) for x in net_stage.sub_net
        ]
        self.signature: Tuple[List[Tensor], List[Tensor]] = (
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
    """
    with atomic_context(bmodel_net, context):
        atomic_mlir = MlirModule(bmodel_net)

    or

    atomic_mlir = MlirModule.from_context(bmodel_net, context)
    """

    def __init__(self, bmodel: BModel):
        super().__init__()
        self.bmodel = bmodel
        self.chip = bmodel.chip
        self.version = bmodel.version
        self.type = bmodel.type
        self.addr_mode = bmodel.addr_mode
        self.functions = [Function(x, 1) for x in self.bmodel.net]

    def create_cmdlist(self) -> List[BaseTpuCmd]:
        res = []
        for func in self.functions:
            for region in func.regions:
                for block in region.blocks:
                    res.extend(block.operations)
        return res

    def dump_head(self):
        attrs = f'attributes {{chip = "{self.chip}", version = {self.version}, addr_mode = {self.addr_mode}}}'
        return f"module {attrs} {{"

    def dump_tail(self):
        return "} loc(#loc)"

    def __str__(self):
        head = self.dump_head()
        tail = self.dump_tail()
        func_str = "\n".join((f"{x}" for x in self.functions))
        func_str = textwrap.indent(func_str, INDENT_SPACE)
        return f"{head}\n{func_str}\n{tail}"

    @staticmethod
    def from_context(bmodel_net: BModel, context):
        with atomic_context(bmodel_net, context):
            return MlirModule(bmodel_net)
