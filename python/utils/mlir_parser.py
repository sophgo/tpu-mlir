#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
#
# Licensed under the Apache License v2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ==============================================================================

import os
import sys
import mlir
from mlir.ir import *


class Operation:

    def __init__(self, op, body, idx):
        self.name = Operation.name(op)
        self.type = Operation.type(op)
        self.loc = Operation.loc(op)
        self.shape = Operation.shape(op)
        self.opds = Operation.operands(op, body, idx)
        self.attrs = Operation.attrs(op)

    def __str__(self):
        return self.name + "," + self.type + "," + self.loc + "," + str(self.shape) + "," + str(
            self.opds)

    @staticmethod
    def name(op):
        return mlir.ir.StringAttr(op.attributes['name']).value

    @staticmethod
    def type(op):
        return op.operation.name

    @staticmethod
    def attrs(op):
        arr_map = {}
        for i in range(len(op.attributes)):
            attr = op.attributes[i]
            k, v = str(attr.name), str(attr.attr)
            arr_map[k] = v
        return arr_map

    @staticmethod
    def loc(op):
        return op.get_asm().split('=')[0].strip('% ')

    @staticmethod
    def shape(op):
        shape_type = mlir.ir.ShapedType(op.operands[0].type)
        shape = [shape_type.get_dim_size(i) for i in range(shape_type.rank)]
        return shape

    @staticmethod
    def operands(op, body, idx):
        opds = []
        for opd in op.operands:
            for j in reversed(range(idx)):
                prev_op = body.operations[j]
                if prev_op.result == opd:
                    if Operation.type(prev_op) not in [
                            'tpu.none', 'tpu.load_weight', 'tpu.weight_file'
                    ]:
                        opds.append(Operation.name(prev_op))
        return opds


class MlirParser:
    def __init__(self, mlir_file):
        with open(mlir_file, 'r') as f:
            context = f.read()
        self.ctx = mlir.ir.Context()
        self.ctx.allow_unregistered_dialects = True
        self.module = mlir.ir.Module.parse(context, self.ctx)
        self.body = self.module.body.operations[0].regions[0].blocks[0]
        self.attrs = Operation.attrs(self.module.operation)
        self.ops = []
        for i in range(len(self.body.operations)):
            op = self.body.operations[i]
            type = Operation.type(op)
            if type in ['top.None','top.Weight', 'func.return']:
                continue
            self.ops.append(Operation(op, self.body, i))
        self.inputs = []
        for op in self.ops:
            if op.type == 'top.Input':
                self.inputs.append(op)

if __name__ == '__main__':
    parser = MlirParser(sys.argv[1])
    print(parser.module)
