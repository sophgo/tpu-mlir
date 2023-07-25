#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import inspect

import setuptools.sandbox
from utils.mlir_parser import MlirParser, Operation
import argparse, re, ast, math, os

# 1684X
NPU_NUM = 64
base_eu_num = 16
FW_MAX_SHAPE_DIMS = 8
tpu_local_mem_size_per_npu = 1 << 18  # 256kB
QUARTER_LOCAL_MEM_SIZE = tpu_local_mem_size_per_npu >> 2
tpu_bank_num = 16
bank_size = tpu_local_mem_size_per_npu / tpu_bank_num
eu_num_map = {
    "f32": base_eu_num,
    "i32": base_eu_num,
    "si32": base_eu_num,
    "ui32": base_eu_num,
    "f16": base_eu_num * 2,
    "bfF16": base_eu_num * 2,
    "i16": base_eu_num * 2,
    "ui16": base_eu_num * 2,
    "si16": base_eu_num * 2,
    "i8": base_eu_num * 4,
    "ui8": base_eu_num * 4,
    "si8": base_eu_num * 4,
}

data_type_t = {
    "i8": (0 << 1) | 1,
    "ui8": (0 << 1) | 0,
    "i16": (3 << 1) | 1,
    "ui16": (3 << 1) | 0,
    "f16": (1 << 1) | 1,
    "bf16": (5 << 1) | 1,
    "i32": (4 << 1) | 1,
    "ui32": (4 << 1) | 0,
    "f32": (2 << 1) | 1
}


def ALIGN(x, a):
    return math.ceil(x / a) * a


def EU_NUM(dtype):
    return eu_num_map[dtype]


def DIV_UP(a, b):
    if a == 0:
        return 0
    else:
        return (a - 1) / b + 1


def prod(num_list):
    mul = 1
    for i in num_list:
        mul *= i
    return mul


def pipeline(array, num):
    for i in range(num - 1, 0, -1):
        array[i] = array[i - 1]
    return array


def tpu_data_type_size(dtype):
    if dtype == "f32" or dtype == "i32" or dtype == "ui32":
        return 4
    elif dtype == "f16" or dtype == "bf16" or dtype == "i16" or dtype == "ui16":
        return 2
    elif dtype == "i8" or dtype == "ui8":
        return 1
    else:
        print("Not Supported DType!!")


def tpu_is_data_type_signed(dtype):
    return dtype & 0x1


#################################used for top.transpose for tiu ops#########################################
def get_optimized_factorization(polynomial):
    factor = 0
    if polynomial > NPU_NUM:
        for i in range(NPU_NUM, NPU_NUM // 2, -1):
            if polynomial % i == 0:
                factor = i
                break

    return factor


def get_transpose_info(fixed_dim, left_dim, right_dim, type_len):
    factor = 0
    use_left_dim = False
    use_right_dim = False
    min_transpose_size = 0
    overflow = 0

    if left_dim > right_dim:
        factor = get_optimized_factorization(left_dim)
        if factor > 0 and factor != left_dim and fixed_dim < left_dim:
            use_left_dim = True
        else:
            factor = get_optimized_factorization(right_dim)
            if factor > 0 and factor != right_dim and fixed_dim == 1:
                use_right_dim = True
    else:
        factor = get_optimized_factorization(right_dim)
        if factor > 0 and factor != right_dim and fixed_dim < right_dim:
            use_right_dim = True
        else:
            factor = get_optimized_factorization(left_dim)
            if factor > 0 and factor != left_dim and fixed_dim == 1:
                use_left_dim = True

    if (use_left_dim == False and use_right_dim == False) or (fixed_dim >= left_dim and fixed_dim >= right_dim):
        factor = get_optimized_factorization(fixed_dim)
        if factor > 0:
            n = 1
            c = factor
            h = left_dim
            w = right_dim
            min_transpose_size = h * w * type_len
            overflow = min_transpose_size / QUARTER_LOCAL_MEM_SIZE
            max_trans_counts = fixed_dim / factor
        elif fixed_dim <= NPU_NUM:
            n = 1
            c = fixed_dim
            h = left_dim
            w = right_dim
            min_transpose_size = h * w * type_len
            overflow = min_transpose_size / QUARTER_LOCAL_MEM_SIZE
            max_trans_counts = 1
        trans_method = 3  # TRANS_NPU_H_SWITCH_W
    elif use_left_dim:
        n = 1
        c = factor
        h = left_dim / factor
        w = right_dim
        trans_method = 1  # TRANS_NPU_N_SWITCH_W
        min_transpose_size = h * w * type_len
        overflow = min_transpose_size / QUARTER_LOCAL_MEM_SIZE
        max_trans_counts = fixed_dim
    elif use_right_dim:
        n = left_dim
        c = factor
        h = right_dim / factor
        w = 1
        trans_method = 1  # TRANS_NPU_N_SWITCH_W
        min_transpose_size = h * n * type_len
        overflow = min_transpose_size / QUARTER_LOCAL_MEM_SIZE
        max_trans_counts = fixed_dim
    if overflow > 1 or overflow == 0:
        trans_method = 0  # TRANS_GENERAL
    return [int(n), int(c), int(h), int(w)], trans_method, max_trans_counts


##################################################################################################################
#####################################  used for top.reduce for tiu ops  ##########################################
def find_smaller_factor(num):
    sqrt_int = int(math.sqrt(num))
    if num % sqrt_int == 0:
        return sqrt_int
    for factor in range(sqrt_int - 1, 0, -1):
        if num % factor == 0:
            break
    return factor


def process_shape(axis_list, axis_num, in_shape_orig, shape_dims_orig):
    is_reduce_orig = [False] * FW_MAX_SHAPE_DIMS
    is_reduce = [False] * FW_MAX_SHAPE_DIMS
    in_shape = [0] * FW_MAX_SHAPE_DIMS
    for i in range(FW_MAX_SHAPE_DIMS):
        in_shape[i] = 1
        is_reduce[i] = is_reduce_orig[i] = False
    for i in range(axis_num):
        is_reduce_orig[axis_list[i]] = True

    pos, reduce_pos = 0, 0
    for i in range(shape_dims_orig):
        if in_shape_orig[i] == 1:
            is_reduce_orig[i] = False
            continue
        if is_reduce_orig[i]:
            axis_list[reduce_pos] = pos
            reduce_pos += 1
            is_reduce[pos] = True
        in_shape[pos] = in_shape_orig[i]
        pos += 1

    if pos < 4:
        for i in range(3, -1, -1):
            if i < 4 - pos:
                in_shape[i] = 1
            else:
                in_shape[i] = in_shape[i + pos - 4]
        for i in range(reduce_pos):
            axis_list[i] += 4 - pos
        pos = 4
    elif pos > 4:
        shape_dims = pos
        pos = 0
        reduce_pos = 0
        minimum_merged_dims = shape_dims
        cur_dims = 0
        for i in range(1, shape_dims):
            if (not is_reduce[i - 1]) and (not is_reduce[i]):
                minimum_merged_dims -= 1

        for i in range(1, shape_dims):
            if (not is_reduce[i - 1]) and (not is_reduce[i]) and (shape_dims - cur_dims > 4):
                in_shape[pos] *= in_shape[i]
                cur_dims += 1
            else:
                if is_reduce[i - 1]:
                    axis_list[reduce_pos] = pos
                    reduce_pos += 1
                pos += 1
                in_shape[pos] = in_shape[i]
        if is_reduce[shape_dims - 1]:
            axis_list[reduce_pos] = pos
            reduce_pos += 1
        pos += 1

    shape_sum = 0
    for i in range(FW_MAX_SHAPE_DIMS):
        shape_sum += in_shape[i]
    if shape_sum - in_shape[3] == 7 and axis_list[0] == 3:
        imm_res1 = int((tpu_local_mem_size_per_npu - 3 * bank_size) / 2)
        imm_res2 = int((imm_res1 / bank_size) * bank_size - 64)
        reduce_hw_min = int((imm_res2 / 64) * 16)

        if in_shape[3] > reduce_hw_min:
            shape_h = int(find_smaller_factor(in_shape[3]))
            shape_w = int(in_shape[3] / shape_h)
            in_shape[2] = shape_h
            in_shape[3] = shape_w
            axis_list[0] = 2
            axis_list[1] = 3
            axis_num = 2
            return pos, in_shape

    axis_num = reduce_pos
    return pos, in_shape


#################################################################################################################


class OpParse(object):
    total_layer_ops = 0
    total_tiu_ops = 0
    total_tiu_utility = None

    def __init__(self, op):
        self.op = op
        self.name = OpParse.name(op)
        self.type = Operation.type(op)
        self.loc = Operation.loc(op)
        self.attrs = Operation.attrs(op)
        self.append_attr = Operation.append_attr(op, self.attrs)
        self.in_list = list(map(lambda x: str(getattr(x, "type")), self.op.operands))
        self.out_list = list(map(lambda x: str(getattr(x, "type")), self.op.results))

        cal_type = re.search(r"(?<=\.)\S+$", self.type).group(
            0).lower()  # match, for example:Top.Mul, "Mul" will be matched and trans into lower case
        self.getLayerOps(cal_type)
        self.getTiuOps(cal_type)
        self.getTiuUtility()

    @staticmethod
    def name(op):  # rewrite method
        loc = op.location
        if loc == "loc(unknown)":
            return None
        # loc(fused["pool1", "pool1_mask"]) => pool1
        return re.search(r'\"(.*)\"', str(loc)).group(1)

    def getLayerOps(self, cal_type):
        simple_detail = LayerOps(self.op)
        cur_layer_ops = int(getattr(simple_detail, cal_type)())
        self.cur_layer_ops = cur_layer_ops
        OpParse.total_layer_ops += cur_layer_ops
        return self.cur_layer_ops, OpParse.total_layer_ops

    def getTiuOps(self, cal_type):
        complex_detail = TiuOps(self.op)
        cur_tiu_ops = int(getattr(complex_detail, cal_type)())
        self.cur_tiu_ops = cur_tiu_ops
        OpParse.total_tiu_ops += cur_tiu_ops
        return self.cur_tiu_ops, OpParse.total_tiu_ops

    def getTiuUtility(self):
        if self.cur_layer_ops != 0 and self.cur_tiu_ops != 0:
            self.cur_tiu_utility = round(self.cur_layer_ops / self.cur_tiu_ops, 2)
            # assert self.cur_tiu_utility <= 1
            if self.cur_tiu_utility > 1:
                print(self.op)
                print(self.cur_tiu_utility)
                print("\n")
        else:
            self.cur_tiu_utility = "None"

        if OpParse.total_layer_ops != 0 and OpParse.total_tiu_ops != 0:
            OpParse.total_tiu_utility = round(OpParse.total_layer_ops / OpParse.total_tiu_ops, 2)
        else:
            OpParse.total_tiu_utility = "None"
        return self.cur_tiu_utility, OpParse.total_tiu_utility

    def keys(self):
        return ["name", "type", "loc", "in_list", "out_list", "cur_layer_ops", "total_layer_ops", "cur_tiu_ops",
                "total_tiu_ops", "cur_tiu_utility", "total_tiu_utility"]

    def values(self):
        return [getattr(self, k) for k in self.keys()]

    @staticmethod
    def getShapeType(OperandOrResult):
        shape_type = re.search(r"(?<=<).*(?=>)", str(OperandOrResult.type)).group(0)
        shape_type_list = shape_type.split("x")
        cur_shape, cur_type = list(map(int, shape_type_list[:-1])), shape_type_list[-1]
        return cur_shape, cur_type


class LayerOps(object):
    def __init__(self, op):
        self.op = op
        assert len(self.op.results) >= 1
        self.out_shape, self.dtype = OpParse.getShapeType(self.op.results[0])
        self.attrs = Operation.attrs(op)
        if 'do_relu' in self.attrs:
            if self.attrs['do_relu'].lower() == "false":
                self.do_relu = False
            else:
                self.do_relu = True
        else:
            self.do_relu = False

    def getNumElements(self, shape_in_list):
        ele = 1
        for i in range(len(shape_in_list)):
            ele *= shape_in_list[i]
        return ele

    def add(self):
        extra = 0
        if self.do_relu:
            extra += 1
        return self.getNumElements(self.out_shape) * (len(self.op.operands) - 1 + extra)

    def addconst(self):
        extra = 0
        if self.do_relu:
            extra += 1
        return self.getNumElements(self.out_shape) * (1 + extra)

    def arange(self):
        return 0

    def arg(self):
        in_shape, _ = OpParse.getShapeType(self.op.operands[0])
        return self.getNumElements(in_shape)

    def attention(self):
        return 0

    def avgpool(self):  # max,avg
        kernel_shape = ast.literal_eval(self.attrs["kernel_shape"])
        kernel_mul = 1
        for i in kernel_shape:
            kernel_mul *= i
        if self.do_relu:
            return self.getNumElements(self.out_shape) * (kernel_mul + 1)
        else:
            return self.getNumElements(self.out_shape) * kernel_mul

    def adaptiveavgpool(self):
        kernel_shape = ast.literal_eval(self.attrs["kernel_shape"])
        kernel_mul = 1
        for i in kernel_shape:
            kernel_mul *= i
        if self.do_relu:
            return self.getNumElements(self.out_shape) * (kernel_mul + 1)
        else:
            return self.getNumElements(self.out_shape) * kernel_mul

    def maxpool(self):  # max,avg
        kernel_shape = ast.literal_eval(self.attrs["kernel_shape"])
        kernel_mul = 1
        for i in kernel_shape:
            kernel_mul *= i
        if self.do_relu:
            return self.getNumElements(self.out_shape) * (kernel_mul + 1)
        else:
            return self.getNumElements(self.out_shape) * kernel_mul

    def batchnorm(self):
        return 2 * self.getNumElements(self.out_shape)

    def cast(self):
        return self.getNumElements(self.out_shape)

    def clip(self):
        return 2 * self.getNumElements(self.out_shape)

    def compare(self):
        return self.getNumElements(self.out_shape)

    def compareconst(self):
        return self.getNumElements(self.out_shape)

    def concat(self):
        return 0

    def constantfill(self):
        return 0

    def conv(self):
        group = ast.literal_eval(self.attrs["group"].split(":")[0])
        kernel_shape = ast.literal_eval(self.attrs["kernel_shape"])
        kernel_mul = 1
        for i in kernel_shape:
            kernel_mul *= i

        extra = 0
        in_shape, in_type = OpParse.getShapeType(self.op.operands[0])
        if len(self.op.operands) > 2:
            extra += 1
        if self.do_relu:
            extra += 1

        ic = in_shape[1]

        return self.getNumElements(self.out_shape) * (2 * kernel_mul * ic / group + extra)

    def copy(self):
        return self.getNumElements(self.out_shape)

    def cos(self):
        return self.getNumElements(self.out_shape) * 4

    def cosh(self):
        return self.getNumElements(self.out_shape) * 4

    def csc(self):
        return self.getNumElements(self.out_shape)

    def custom(self):
        return 0

    def deconv(self):
        group = ast.literal_eval(self.attrs["group"].split(":")[0])
        kernel_shape = ast.literal_eval(self.attrs["kernel_shape"])
        kernel_mul = 1
        for i in kernel_shape:
            kernel_mul *= i

        extra = 0
        in_shape, in_type = OpParse.getShapeType(self.op.operands[0])
        if len(self.op.operands) > 2:
            extra += 1
        if self.do_relu:
            extra += 1

        oc = self.out_shape[1]

        return self.getNumElements(in_shape) * (2 * kernel_mul * oc / group + extra)

    def deformconv2d(self):
        group = ast.literal_eval(self.attrs["group"].split(":")[0])
        kernel_shape = ast.literal_eval(self.attrs["kernel_shape"])
        kernel_mul = 1
        for i in kernel_shape:
            kernel_mul *= i

        extra = 0
        in_shape, in_type = OpParse.getShapeType(self.op.operands[0])
        if len(self.op.operands) > 2:
            extra += 1
        if self.do_relu:
            extra += 1

        ic = in_shape[1]

        return self.getNumElements(self.out_shape) * (2 * kernel_mul * ic / group + extra + 1)

    def depth2space(self):
        return 0

    def dequantizelinear(self):
        return self.getNumElements(self.out_shape)

    def detectionoutput(self):
        return self.getNumElements(self.out_shape)

    def div(self):
        return self.getNumElements(self.out_shape)

    def elu(self):
        return self.getNumElements(self.out_shape)

    def erf(self):
        return self.getNumElements(self.out_shape)

    def exp(self):
        return self.getNumElements(self.out_shape) * 4

    def flatten(self):
        return 0

    def floor(self):
        return self.getNumElements(self.out_shape)

    def frcndetection(self):
        return self.getNumElements(self.out_shape)

    def gather(self):
        return 0

    def gatherelements(self):
        return 0

    def gathernd(self):
        return 0

    def gelu(self):
        return self.getNumElements(self.out_shape) * 5

    def gridsampler(self):
        mode = ast.literal_eval(self.attrs["mode"].split(":")[0])
        if mode == 0:
            grid_shape, _ = OpParse.getShapeType(self.op.operands[1])
            return 21 * grid_shape[1] * grid_shape[2]
        else:
            return 0

    def groupnorm(self):
        return self.getNumElements(self.out_shape) * (10 + len(self.op.operands) - 1)

    def layernorm(self):
        if len(self.op.operands) >= 3:
            extra = 2
        elif len(self.op.operands) == 2:
            extra = 1
        return self.getNumElements(self.out_shape) * (10 + extra)

    def leakyrelu(self):
        return self.getNumElements(self.out_shape)

    def log(self):
        return 4 * self.getNumElements(self.out_shape)

    def matmul(self):
        extra = 0
        if self.do_relu:
            extra += 1
        if len(self.op.operands) > 2:
            extra += 1

        if self.attrs['right_transpose'].lower() == "false":
            right_transpose = False
        else:
            right_transpose = True

        a_shape, _ = OpParse.getShapeType(self.op.operands[0])
        b_shape, _ = OpParse.getShapeType(self.op.operands[1])

        a_dims = len(a_shape)
        b_dims = len(b_shape)
        o_dims = len(self.out_shape)

        if b_dims == 1:
            b_shape.append(1)
            self.out_shape.append(1)
            b_dims += 1
            o_dims += 1

        if a_dims == 1:
            a_shape.insert(0, 1)
            self.out_shape.insert(0, 1)
            a_dims += 1
            o_dims += 1

        if right_transpose:
            n = b_shape[b_dims - 2]
            k = b_shape[b_dims - 1]
        else:
            n = b_shape[b_dims - 1]
            k = b_shape[b_dims - 2]

        assert (n == self.out_shape[o_dims - 1])

        batch = 1
        for i in range(b_dims - 2):
            batch *= b_shape[i]
        if batch > 1 or o_dims <= 2:
            m = self.out_shape[o_dims - 2]
        else:
            m = 1
            for j in range(0, o_dims - 1):
                m *= self.out_shape[j]

        return batch * (2 * k + extra) * n * m

    def mul(self):
        extra = 0
        if self.do_relu:
            extra += 1
        input_nums = len(self.op.operands)
        return self.getNumElements(self.out_shape) * (input_nums - 1 + extra)

    def mulconst(self):
        extra = 0
        if self.do_relu:
            extra += 1
        return self.getNumElements(self.out_shape) * (1 + extra)

    def permute(self):
        return 0

    def relu(self):
        return self.getNumElements(self.out_shape)

    def reshape(self):
        return 0

    def roipooling(self):
        return self.getNumElements(self.out_shape)

    def scale(self):
        extra = 0
        if self.do_relu:
            extra += 1
        return self.getNumElements(self.out_shape) * (2 + extra)

    def sigmoid(self):
        return 4 * self.getNumElements(self.out_shape)

    def silu(self):
        return 5 * self.getNumElements(self.out_shape)

    def slice(self):
        return 0

    def softmax(self):
        in_shape, in_type = OpParse.getShapeType(self.op.operands[0])
        extra = 0
        if self.attrs["log"] == "true":
            extra += 1
        return self.getNumElements(in_shape) * (5 + extra)

    def squeeze(self):
        return 0

    def sqrt(self):
        return self.getNumElements(self.out_shape)

    def upsample(self):
        if self.do_relu:
            return 2 * self.getNumElements(self.out_shape)
        else:
            return self.getNumElements(self.out_shape)

    def weigh(self):
        return 0

    def input(self):
        return 0

    def scatternd(self):
        return 0

    def sub(self):
        extra = 0
        if self.do_relu:
            extra += 1
        return self.getNumElements(self.out_shape) * (len(self.op.operands) - 1 + extra)

    def subconst(self):
        extra = 0
        if self.do_relu:
            extra += 1
        return self.getNumElements(self.out_shape) * (1 + extra)

    def abs(self):
        return self.getNumElements(self.out_shape)

    def reduce(self):
        return self.getNumElements(self.out_shape)

    def normalize(self):
        return self.getNumElements(self.out_shape) * 2

    def max(self):
        return self.getNumElements(self.out_shape)

    def maxconst(self):
        return self.getNumElements(self.out_shape)

    def min(self):
        return self.getNumElements(self.out_shape)

    def minconst(self):
        return self.getNumElements(self.out_shape)

    def transpose(self):
        return 0

    def sin(self):
        return self.getNumElements(self.out_shape) * 4

    def sinh(self):
        return self.getNumElements(self.out_shape) * 4


class TiuOps(object):  # currently only support input shape=(n,c,h,w)
    def __init__(self, op):
        self.op = op
        assert len(self.op.results) >= 1
        self.out_shape, self.dtype = OpParse.getShapeType(self.op.results[0])
        self.attrs = Operation.attrs(op)
        if 'do_relu' in self.attrs:
            if self.attrs['do_relu'].lower() == "false":
                self.do_relu = False
            else:
                self.do_relu = True
        else:
            self.do_relu = False

    def reshape_to_4d(self, shape):
        if len(shape) < 4:
            shape = [1] * (4 - len(shape)) + shape
        elif len(shape) > 4:
            shape = [prod(shape[:-3])] + shape[-3:]
        return shape

    def atomic_ar(self, shape, in_type, atomic_op_name="arith.add"):  # for atomic_ar, shape is same before and after op
        n, c, h, w = shape
        factor = 1
        if atomic_op_name == "arith.div":
            factor = 5

        c = ALIGN(c, NPU_NUM)
        hw = ALIGN(h * w, EU_NUM(in_type))
        return n * c * hw * factor

    def atomic_pord(self, out_shape, in_type, atomic_op_name, as_quotation=False, kernel_mul=None):
        n, c, h, w = out_shape

        # get upper function name, such as scale or batchnorm
        stack = inspect.stack()
        upper_func_name = stack[1].function
        if not as_quotation:
            if upper_func_name == "scale" or upper_func_name == "batchnorm":
                in_shape1, in_type1 = OpParse.getShapeType(self.op.operands[1])
                kernel_mul = in_shape1[-1] * in_shape1[-2]
            else:
                kernel_shape = ast.literal_eval(self.attrs["kernel_shape"])
                kernel_mul = 1
                for i in kernel_shape:
                    kernel_mul *= i
        else:
            assert kernel_mul is not None

        factor = 1
        if atomic_op_name == "pord.avgpooling" or atomic_op_name == "pord.maxpooling":
            factor = len(self.op.results)  # TODO: fix the factor
        elif atomic_op_name == "pord.depthwise":
            factor = 2
        elif atomic_op_name == "pord.depthwiserelu":
            factor = 3
        else:
            # roi_pooling
            kernel_mul = 1
            factor = 2 * 4 - 1  # bilinar, ignore coords generate

        # in_shape, in_type = OpParse.getShapeType(self.op.operands[0])
        c = ALIGN(c, NPU_NUM)
        hw = ALIGN(h * w, EU_NUM(in_type))
        return n * c * hw * (factor * kernel_mul + 1)

    def atomic_conv(self, in_shape, in_type, out_shape, atomic_op_name=None):
        group = ast.literal_eval(self.attrs["group"].split(":")[0])
        kernel_shape = ast.literal_eval(self.attrs["kernel_shape"])
        kernel_mul = 1
        for i in kernel_shape:
            kernel_mul *= i
        extra = 0
        # in_shape, dtype = OpParse.getShapeType(self.op.operands[0])
        if len(self.op.operands) > 2:
            extra += 1
        if self.do_relu:
            extra += 1

        n, ic, ih, iw = in_shape
        n, oc, oh, ow = out_shape
        ohw = oh * ow

        ic = ALIGN(ic, NPU_NUM)
        ohw = ALIGN(ohw, EU_NUM(in_type))
        oc = ALIGN(oc, NPU_NUM)
        return n * oc * ohw * (2 * ic * kernel_mul / group + extra)

    def atomic_matmul(self, atomic_op_name=None):
        extra = 0
        if len(self.op.operands) > 2:
            extra += 1

        if self.attrs['right_transpose'].lower() == "false":
            right_transpose = False
        else:
            right_transpose = True

        a_shape, a_type = OpParse.getShapeType(self.op.operands[0])
        b_shape, b_type = OpParse.getShapeType(self.op.operands[1])

        a_dims = len(a_shape)
        b_dims = len(b_shape)
        o_dims = len(self.out_shape)

        if b_dims == 1:
            b_shape.append(1)
            self.out_shape.append(1)
            b_dims += 1
            o_dims += 1

        if a_dims == 1:
            a_shape.insert(0, 1)
            self.out_shape.insert(0, 1)
            a_dims += 1
            o_dims += 1

        if right_transpose:
            n = b_shape[b_dims - 2]
            k = b_shape[b_dims - 1]
        else:
            n = b_shape[b_dims - 1]
            k = b_shape[b_dims - 2]

        assert (n == self.out_shape[o_dims - 1])

        batch = 1
        for i in range(b_dims - 2):
            batch *= b_shape[i]
        if batch > 1 or o_dims <= 2:
            m = self.out_shape[o_dims - 2]
        else:
            m = 1
            for j in range(0, o_dims - 1):
                m *= self.out_shape[j]

        k = ALIGN(k, EU_NUM(a_type))
        m = ALIGN(m, NPU_NUM)

        return batch * (2 * k + extra) * n * m

    def atomic_cmp(self, in_shape, in_type, atomic_op_name=None):
        n, c, h, w = in_shape
        c = ALIGN(c, NPU_NUM)
        hw = ALIGN(h * w, EU_NUM(in_type))
        return n * c * hw * 2

    def atomic_sfu(self, in_shape, in_type, atomic_op_name):
        n, c, h, w = in_shape
        res_num = len(self.op.results)
        if atomic_op_name == "sfu.taylor_4x" or atomic_op_name == "sfu.taylor":
            in_shape1, dtype1 = OpParse.getShapeType(self.op.operands[0])
            factor = 2 * in_shape1[
                0] - 1  # TODO:change the factor,but don't know the operands[0] is for layer input or atomic input?
        elif atomic_op_name == "sfu.normalize":
            factor = 1
        elif atomic_op_name == "sfu.rsqrt":
            factor = 3  # iteration times

        c = ALIGN(c, NPU_NUM)
        hw = ALIGN(h * w, EU_NUM(in_type))
        return res_num * n * c * hw * factor

    def atomic_vc(self, in_type, out_shape, atomic_op_name=None):
        n, c, h, w = out_shape
        c = ALIGN(c, NPU_NUM)
        w = ALIGN(w, EU_NUM(in_type))
        return n * c * h * w

    def atomic_lin(self, in_type, out_shape, atomic_op_name=None):
        n, c, h, w = out_shape
        c = ALIGN(c, NPU_NUM)
        w = ALIGN(w, EU_NUM(in_type))
        return n * c * h * w * 2

    def atomic_rqdq(self, in_type, out_shape, atomic_op_name=None):
        n, c, h, w = out_shape
        factor = 3  # mul, add, shift
        c = ALIGN(c, NPU_NUM)
        hw = ALIGN(h * w, EU_NUM(in_type))
        return n * c * hw * factor

    def atomic_sg(self, in_type, out_shape, atomic_op_name=None):
        n, c, h, w = out_shape
        factor = 1
        c = ALIGN(c, NPU_NUM)
        w = ALIGN(w, EU_NUM(in_type))
        return n * c * h * w * factor

    def atomic_sgl(self, in_type, out_shape, atomic_op_name=None):
        n, c, h, w = out_shape
        factor = 1
        c = ALIGN(c, NPU_NUM)
        w = ALIGN(w, EU_NUM(in_type))
        return n * c * h * w * factor

    def atomic_transbc(self, in_type, out_shape, atomic_op_name):
        n, c, h, w = out_shape
        factor = 1
        c = ALIGN(c, NPU_NUM)
        if atomic_op_name in (
                "tsbc.l_copy",
                "tsbc.l_bc",
                "tsbc.s_bc",
                "tsbc.s_distribute",
        ):
            hw = ALIGN(h * w, EU_NUM(in_type))
        else:
            hw = h * ALIGN(w, EU_NUM(in_type))
        return n * c * hw * factor

    def atomic_lar(self, in_type, out_shape, atomic_op_name=None):
        n, c, h, w = out_shape
        c = ALIGN(c, NPU_NUM)
        w = ALIGN(w, EU_NUM(in_type))
        return n * c * h * w

    ####################  single op  ############################
    def conv(self):
        in_shape, in_type = OpParse.getShapeType(self.op.operands[0])
        return self.atomic_conv(in_shape, in_type, self.out_shape)

    def compare(self):
        in_shape, in_type = OpParse.getShapeType(self.op.operands[0])
        in_shape = self.reshape_to_4d(in_shape)
        return self.atomic_cmp(in_shape, in_type)

    def compareconst(self):
        in_shape, in_type = OpParse.getShapeType(self.op.operands[0])
        in_shape = self.reshape_to_4d(in_shape)
        return self.atomic_cmp(in_shape, in_type)

    def add(self):
        self.out_shape = self.reshape_to_4d(self.out_shape)
        return self.atomic_ar(self.out_shape, self.dtype)

    def addconst(self):
        self.out_shape = self.reshape_to_4d(self.out_shape)
        return self.atomic_ar(self.out_shape, self.dtype)

    def matmul(self):
        return self.atomic_matmul()

    def normalize(self):
        in_shape, in_type = OpParse.getShapeType(self.op.operands[0])
        return self.atomic_sfu(in_shape, in_type, "sfu.normalize")

    def mul(self):
        self.out_shape = self.reshape_to_4d(self.out_shape)
        return self.atomic_ar(self.out_shape, self.dtype)

    def mulconst(self):
        self.out_shape = self.reshape_to_4d(self.out_shape)
        return self.atomic_ar(self.out_shape, self.dtype)

    def sub(self):
        self.out_shape = self.reshape_to_4d(self.out_shape)
        return self.atomic_ar(self.out_shape, self.dtype)

    def subconst(self):
        self.out_shape = self.reshape_to_4d(self.out_shape)
        return self.atomic_ar(self.out_shape, self.dtype)

    def max(self):
        self.out_shape = self.reshape_to_4d(self.out_shape)
        return self.atomic_ar(self.out_shape, self.dtype)

    def maxconst(self):
        self.out_shape = self.reshape_to_4d(self.out_shape)
        return self.atomic_ar(self.out_shape, self.dtype)

    def min(self):
        self.out_shape = self.reshape_to_4d(self.out_shape)
        return self.atomic_ar(self.out_shape, self.dtype)

    def minconst(self):
        self.out_shape = self.reshape_to_4d(self.out_shape)
        return self.atomic_ar(self.out_shape, self.dtype)

    def div(self):
        self.out_shape = self.reshape_to_4d(self.out_shape)
        return self.atomic_ar(self.out_shape, self.dtype, "arith.div")

    def cast(self):
        self.out_shape = self.reshape_to_4d(self.out_shape)
        return self.atomic_ar(self.out_shape, self.dtype)

    def copy(self):
        self.out_shape = self.reshape_to_4d(self.out_shape)
        return self.atomic_ar(self.out_shape, self.dtype)

    def abs(self):
        self.out_shape = self.reshape_to_4d(self.out_shape)
        return self.atomic_ar(self.out_shape, self.dtype)

    def avgpool(self):
        in_shape, in_type = OpParse.getShapeType(self.op.operands[0])
        return self.atomic_pord(self.out_shape, in_type, "pord.avgpooling")

    def adaptiveavgpool(self):
        in_shape, in_type = OpParse.getShapeType(self.op.operands[0])
        return self.atomic_pord(self.out_shape, in_type, "pord.avgpooling")

    def maxpool(self):
        in_shape, in_type = OpParse.getShapeType(self.op.operands[0])
        return self.atomic_pord(self.out_shape, in_type, "pord.maxpooling")

    def roipool(self):
        in_shape, in_type = OpParse.getShapeType(self.op.operands[0])
        return self.atomic_pord(self.out_shape, in_type, "pord.roiavgpooling")

    #################################################
    def arange(self):
        return self.atomic_ar(self.out_shape, self.dtype)

    def attention(self):
        # return self.atomic_ar("arith.add") * 6 + self.atomic_matmul() * 6 + self.atomic_transbc(
        #     "tsbc.l_bc") + self.atomic_ar("arith.mul") * 5
        return 0

    def arg(self):
        in_shape, in_type = OpParse.getShapeType(self.op.operands[0])
        dim = len(in_shape)
        axis = int(self.attrs["axis"])

        n, c, h, w = 1, 1, 1, 1
        if axis == dim - 1:
            for i in range(axis):
                c *= in_shape[i]
            w = in_shape[axis]
            axis = 3
        else:
            for i in range(axis + 1, dim):
                w *= in_shape[i]
            for i in range(axis):
                c *= in_shape[i]

            h = in_shape[axis]
            axis = 2

        on, oc, oh, ow = n, c, h, w
        if axis == 0:
            on = 1
        elif axis == 1:
            oc = 1
        elif axis == 2:
            oh = 1
        elif axis == 3:
            ow = 1
        in_shape = [n, c, h, w]
        out_shape = [on, oc, oh, ow]
        return self.atomic_ar(in_shape, in_type, in_shape, self.dtype) * 2 + self.atomic_pord(in_shape, in_type,
                                                                                              out_shape, self.dtype,
                                                                                              "pord.maxpooling") * 2

    def batchnorm(self):  # calculation according to the test case
        in_shape, in_type = OpParse.getShapeType(self.op.operands[0])
        return self.atomic_pord(in_shape, in_type, "pord.avgpooling")

    def deconv(self):
        in_shape, in_type = OpParse.getShapeType(self.op.operands[0])
        return self.atomic_conv(self.out_shape, in_type, in_shape)

    def deformconv2d(self):
        in_shape, in_type = OpParse.getShapeType(self.op.operands[0])
        group = ast.literal_eval(self.attrs["group"].split(":")[0])
        kernel_shape = ast.literal_eval(self.attrs["kernel_shape"])
        kernel_mul = 1
        for i in kernel_shape:
            kernel_mul *= i
        extra = 0
        # in_shape, dtype = OpParse.getShapeType(self.op.operands[0])
        if len(self.op.operands) > 2:
            extra += 1
        if self.do_relu:
            extra += 1

        n, ic, ih, iw = in_shape
        n, oc, oh, ow = self.out_shape
        ohw = oh * ow

        ic = ALIGN(ic, NPU_NUM)
        ohw = ALIGN(ohw, EU_NUM(in_type))
        oc = ALIGN(oc, NPU_NUM)
        return n * oc * ohw * (2 * ic * kernel_mul / group + extra + 1)

    def clip(self):  # calculation according to the test case
        in_shape, in_type = OpParse.getShapeType(self.op.operands[0])
        return self.atomic_ar(in_shape, in_type, "arith.min") * 2

    def concat(self):
        return 0

    def constantfill(self):
        return 0

    def csc(self):
        return self.getNumElements(self.out_shape)

    def custom(self):
        return 0

    def depth2space(self):
        return 0

    def dequantizelinear(self):
        return self.atomic_rqdq(self.dtype, self.out_shape)

    def detectionoutput(self):
        return self.getNumElements(self.out_shape)

    def flatten(self):  # calculation according to the test case. No nodechip file
        return 0

    def floor(self):  # calculation according to the test case. No nodechip file
        return 0  # ignor tpu.cast

    def frcndetection(self):
        return self.getNumElements(self.out_shape)

    def gather(self):  # calculation according to the nodechip file.
        return 0

    def gatherelements(self):
        return 0

    def gathernd(self):
        in_shape, in_type = OpParse.getShapeType(self.op.operands[0])
        in_shape = self.reshape_to_4d(in_shape)
        return self.atomic_ar(in_shape, in_type) * 3 + self.atomic_transbc(in_type, self.out_shape, "tsbc.cw_ts")

    def gelu(self):
        return self.active("gelu")

    def gridsampler(self):
        mode = ast.literal_eval(self.attrs["mode"].split(":")[0])
        pad_mode = ast.literal_eval(self.attrs["padding_mode"].split(":")[0])
        if self.attrs["align_corners"] == "false":
            align_corners = False
        else:
            align_corners = True

        if mode == 0:
            in_shape, in_type = OpParse.getShapeType(self.op.operands[0])
            ops = 0
            if align_corners:
                ops += self.atomic_ar(in_shape, in_type) * 2
            else:
                ops += self.atomic_ar(in_shape, in_type) * 3

            ops += self.atomic_ar(in_shape, in_type) * 13 + self.atomic_cmp(in_shape, in_type)
        else:
            return 0

    def groupnorm(self):
        in_shape, in_type = OpParse.getShapeType(self.op.operands[0])
        in_shape = self.reshape_to_4d(in_shape)
        return self.atomic_ar(in_shape, in_type) * 12

    def layernorm(self):
        in_shape, in_type = OpParse.getShapeType(self.op.operands[0])
        in_shape = self.reshape_to_4d(in_shape)
        return self.atomic_ar(in_shape, in_type) * 12

    def leakyrelu(self):  # calculation according to the test case.  No corresponding nodechip file
        in_shape, in_type = OpParse.getShapeType(self.op.operands[0])
        return self.atomic_ar(in_shape, in_type, "arith.mul") + self.atomic_cmp(in_shape, in_type)

    def log(self):  # calculation according to the test case. No corresponding nodechip file
        return self.active("ln")

    def relu(self):  # calculation according to the test case.
        return self.atomic_ar(self.out_shape, self.dtype, "arith.max")

    def reshape(self):  # calculation according to the nodechip_reshape_local file , simply use a copy op
        in_shape, in_type = OpParse.getShapeType(self.op.operands[0])
        in_shape = self.reshape_to_4d(in_shape)
        return self.atomic_ar(in_shape, in_type)

    def reduce(self):
        def reduce_onedim(new_shape_onedim, in_type_onedim, method_onedim, axis_list_onedim,
                          shape_dims_onedim):
            ops_onedim = 0
            axis = axis_list_onedim[0]
            if axis == 2:
                pord_out = [new_shape_onedim[0], new_shape_onedim[1], 1, new_shape_onedim[3]]
                if method_onedim.lower() == "reducemean" or method_onedim.lower() == "reducesum":
                    ops_onedim += self.atomic_pord(pord_out, in_type_onedim, "pord.avgpooling", as_quotation=True,
                                                   kernel_mul=new_shape_onedim[2])
                elif method_onedim.lower() == "reducemax" or method_onedim.lower() == "reducemin":
                    if method_onedim.lower() == "reducemin":
                        ops_onedim += self.atomic_ar(new_shape_onedim, in_type_onedim) * 2
                    ops_onedim += self.atomic_pord(pord_out, in_type_onedim, "pord.maxpooling", as_quotation=True,
                                                   kernel_mul=new_shape_onedim[2])
                return ops_onedim

            if axis != 3:
                trans_order = [0] * FW_MAX_SHAPE_DIMS
                trans_shape = [1] * FW_MAX_SHAPE_DIMS
                trans_order[3] = axis
                for i in range(shape_dims_onedim - 1):
                    trans_order[i] = i + (i >= axis)
                for i in range(shape_dims_onedim):
                    trans_shape[i] = new_shape_onedim[trans_order[i]]
                permute_ops, permute_results = self.permute(new_shape_onedim, in_type_onedim, trans_order,
                                                            as_quotation=True)
                ops_onedim += permute_ops
                axis_num = 1
                axis_list_onedim[0] = 3
                for shape in permute_results:
                    if shape == [0, 0, 0, 0] or shape == [1, 1, 1, 1]:
                        continue
                    _, new_shape_onedim = process_shape(axis_list_onedim, axis_num, trans_shape, 4)

                    pord_out = [new_shape_onedim[0], new_shape_onedim[1], new_shape_onedim[2], 1]
                    if method_onedim.lower() == "reducemean" or method_onedim.lower() == "reducesum":
                        ops_onedim += self.atomic_pord(pord_out, in_type_onedim, "pord.avgpooling", as_quotation=True,
                                                       kernel_mul=new_shape_onedim[3])
                    elif method_onedim.lower() == "reducemax" or method_onedim.lower() == "reducemin":
                        if method_onedim.lower() == "reducemin":
                            ops_onedim += self.atomic_ar(new_shape_onedim, in_type_onedim) * 2
                        ops_onedim += self.atomic_pord(pord_out, in_type_onedim, "pord.maxpooling", as_quotation=True,
                                                       kernel_mul=new_shape_onedim[3])
                return ops_onedim

            pord_out = [new_shape_onedim[0], new_shape_onedim[1], new_shape_onedim[2], 1]
            if method_onedim.lower() == "reducemean" or method_onedim.lower() == "reducesum":
                ops_onedim += self.atomic_pord(pord_out, in_type_onedim, "pord.avgpooling", as_quotation=True,
                                               kernel_mul=new_shape_onedim[3])
            elif method_onedim.lower() == "reducemax" or method_onedim.lower() == "reducemin":
                if method_onedim.lower() == "reducemin":
                    ops_onedim += self.atomic_ar(new_shape_onedim, in_type_onedim) * 2
                ops_onedim += self.atomic_pord(pord_out, in_type_onedim, "pord.maxpooling", as_quotation=True,
                                               kernel_mul=new_shape_onedim[3])
            return ops_onedim

        in_shape, in_type = OpParse.getShapeType(self.op.operands[0])
        axis_list_orig = ast.literal_eval(self.attrs["axes"])
        shape_dims_orig = len(in_shape)
        axis_num = len(axis_list_orig)
        method = ast.literal_eval(self.attrs["mode"])

        axis_list = [0] * FW_MAX_SHAPE_DIMS
        loop = 0
        for i in range(axis_num):
            if in_shape[axis_list_orig[i]] == 1:
                continue
            axis_list[loop] = axis_list_orig[i]
            loop += 1
        axis_num = loop

        new_shape_dims, new_shape = process_shape(axis_list, axis_num, in_shape, shape_dims_orig)
        if new_shape_dims < 4:
            new_shape_dims, new_shape = process_shape(axis_list, axis_num, in_shape, shape_dims_orig)
        elif new_shape_dims > 4:
            print("Not implement this process yet.")
            exit()
        assert new_shape_dims == 4

        ops = 0
        if axis_num == 1:  # precise calculation
            ops += reduce_onedim(new_shape, in_type, method, axis_list, 4)
        elif axis_num == 2:  # not precise calculation
            is_reduce = [False] * 4
            for i in range(axis_num):
                is_reduce[axis_list[i]] = True
            shape = [0] * 4
            shape[axis_list[1]] = 1
            ops += reduce_onedim(new_shape, in_type, method, axis_list[-1:], 4)
            dims, new_shape_2nd = process_shape(axis_list, axis_num, shape, 4)
            ops += reduce_onedim(new_shape_2nd, in_type, method, axis_list[:1], 4)
        elif axis_num == 3:  # not precise calculation
            ops += reduce_onedim(new_shape, in_type, method, [axis_list[0]], 4)
            ops += reduce_onedim(new_shape, in_type, method, [axis_list[1]], 4)
            ops += reduce_onedim(new_shape, in_type, method, [axis_list[2]], 4)
        elif axis_num == 4:  # not precise calculation
            ops += reduce_onedim(new_shape, in_type, method, [axis_list[0]], 4)
            ops += reduce_onedim(new_shape, in_type, method, [axis_list[1]], 4)
            ops += reduce_onedim(new_shape, in_type, method, [axis_list[2]], 4)
            ops += reduce_onedim(new_shape, in_type, method, [axis_list[3]], 4)

        return ops

    def active(self, active_type, in_shape=None, in_type=None, if_local_layer=False,
               as_quotation=False):  # need to consider the optimization for ACTIVE
        if in_shape is None and in_type is None:
            in_shape, in_type = OpParse.getShapeType(self.op.operands[0])

        if not as_quotation:
            length = 1
            for i in range(len(in_shape)):
                length *= in_shape[i]
            tensor_w = int(max(DIV_UP(min(length, 16384), NPU_NUM),
                               DIV_UP(128, eu_num_map[in_type] * tpu_data_type_size(in_type))))
            slice = min(min(length, NPU_NUM * tensor_w), 16384)
            max_rows_per_time = int(bank_size / (tensor_w * tpu_data_type_size(in_type)))
            rows = int(DIV_UP(length, slice))
            rows_secs = int(DIV_UP(rows, max_rows_per_time))
            # at least loop two times to overlap all bdc time
            rows_slice = int(DIV_UP(rows, max(rows_secs, 2)))

            cur_idx = [0] * 3
            cur_rows = [0] * 3
            cur_cols = [0] * 3
            stage_idx, draning_idx = 0, 0
            ops = 0
            while cur_idx[2] < length:
                if draning_idx < 1:
                    cur_len = min(length - cur_idx[0], rows_slice * slice)
                    cur_cols[0] = min(cur_len, slice)
                    cur_rows[0] = max(1, cur_len / cur_cols[0])
                if stage_idx > 0 and draning_idx < 2:
                    cur_shape = [cur_rows[1], int(DIV_UP(cur_cols[1], tensor_w)), 1, tensor_w]

                    if active_type == "rsqrt":
                        ops += self.active("rsqrt", cur_shape, in_type, as_quotation=True)
                    elif active_type == "sqrt":
                        ops += self.active("sqrt", cur_shape, in_type, as_quotation=True)
                    elif active_type == "square":
                        ops += self.active("square", cur_shape, in_type, as_quotation=True)
                    elif active_type == "floor" or active_type == "ceil" or active_type == "round":
                        ops += self.active("floor", cur_shape, in_type, as_quotation=True)
                    elif active_type == "absval":
                        ops += self.active("absval", cur_shape, in_type, as_quotation=True)
                    elif active_type == "hswish":  # only support float type
                        ops += self.active("hswish", cur_shape, in_type, as_quotation=True)
                    elif active_type == "hsigmoid":  # only support float type
                        ops += self.active("hsigmoid", cur_shape, in_type, as_quotation=True)
                    elif active_type == "is_finite":
                        ops += self.active("is_finite", cur_shape, in_type, as_quotation=True)
                    elif active_type == "sign":
                        ops += self.active("sign", cur_shape, in_type, as_quotation=True)
                    elif active_type == "soft_sign":
                        ops += self.active("soft_sign", cur_shape, in_type, as_quotation=True)

                    if if_local_layer:
                        local_n_shape = 1
                    else:
                        local_n_shape = int(cur_shape[0])

                    new_cur_shape = [local_n_shape, cur_shape[1], cur_shape[2], cur_shape[3]]
                    for i in range(0, int(cur_shape[0]), int(new_cur_shape[0])):
                        if active_type == "sigmoid":
                            ops += self.active("sigmoid", new_cur_shape, in_type, as_quotation=True)
                        elif active_type == "mish":
                            ops += self.active("mish", new_cur_shape, in_type, as_quotation=True)
                        elif active_type == "sinh" or active_type == "cosh":
                            ops += self.active("sinh", new_cur_shape, in_type, as_quotation=True)
                        elif active_type == "tanh":
                            ops += self.active("tanh", new_cur_shape, in_type, as_quotation=True)
                        elif active_type == "exp":
                            ops += self.active("exp", new_cur_shape, in_type, as_quotation=True)
                        elif active_type == "silu":  # sigmoid+mul
                            ops += self.active("silu", new_cur_shape, in_type, as_quotation=True)
                        elif active_type == "swish":  # mul+sigmoid+mul
                            ops += self.active("swish", new_cur_shape, in_type, as_quotation=True)
                        elif active_type == "elu":
                            ops += self.active("elu", new_cur_shape, in_type, as_quotation=True)
                        elif active_type == "erf":
                            ops += self.active("erf", new_cur_shape, in_type, as_quotation=True)
                        elif active_type == "gelu":
                            ops += self.active("gelu", new_cur_shape, in_type, as_quotation=True)
                        elif active_type == "ln":  # log
                            ops += self.active("ln", new_cur_shape, in_type, as_quotation=True)
                        elif active_type == "log_sigmoid":
                            ops += self.active("log_sigmoid", new_cur_shape, in_type, as_quotation=True)
                        elif active_type == "arcsinh" or active_type == "arccosh":
                            ops += self.active("arcsinh", new_cur_shape, in_type, as_quotation=True)
                        elif active_type == "arctanh":
                            ops += self.active("arctanh", new_cur_shape, in_type, as_quotation=True)
                        elif active_type == "soft_plus":
                            ops += self.active("soft_plus", new_cur_shape, in_type, as_quotation=True)
                        elif active_type == "tan":
                            ops += self.active("tan", new_cur_shape, in_type, as_quotation=True)
                        elif active_type == "sin":
                            ops += self.active("sin", new_cur_shape, in_type, as_quotation=True)
                        elif active_type == "cos":
                            ops += self.active("cos", new_cur_shape, in_type, as_quotation=True)
                        elif active_type == "arcsin":
                            ops += self.active("arcsin", new_cur_shape, in_type, as_quotation=True)
                        elif active_type == "arccos":
                            ops += self.active("arccos", new_cur_shape, in_type, as_quotation=True)

                cur_idx = pipeline(cur_idx, 3)
                cur_cols = pipeline(cur_cols, 3)
                cur_rows = pipeline(cur_rows, 3)
                if draning_idx < 1:
                    cur_idx[0] += cur_cols[0] * cur_rows[0]
                    if cur_idx[0] >= length:
                        draning_idx += 1
                else:
                    draning_idx += 1
                stage_idx += 1
            return ops

        else:
            assert in_shape is not None
            quotation_ops = 0
            ar_ops = self.atomic_ar(in_shape, in_type)
            div_ops = ar_ops * 5
            taylor_ops = self.atomic_sfu(in_shape, in_type, "sfu.taylor_4x")
            rsqrt_ops = self.atomic_sfu(in_shape, in_type, "sfu.rsqrt")
            normalize_ops = self.atomic_sfu(in_shape, in_type, "sfu.normalize")
            cmp_ops = self.atomic_cmp(in_shape, in_type)
            lin_ops = self.atomic_lin(in_type, in_shape)

            if active_type == "rsqrt":
                quotation_ops += rsqrt_ops
            elif active_type == "sqrt":
                quotation_ops += rsqrt_ops + cmp_ops * 2 + ar_ops
            elif active_type == "square":
                quotation_ops += ar_ops
            elif active_type == "floor" or active_type == "ceil" or active_type == "round":
                quotation_ops += ar_ops
            elif active_type == "absval":
                quotation_ops += ar_ops
            elif active_type == "hswish":  # only support float type
                quotation_ops += ar_ops * 5
            elif active_type == "hsigmoid":  # only support float type
                quotation_ops += ar_ops * 4
            elif active_type == "is_finite":
                quotation_ops += ar_ops * 2
            elif active_type == "sign":
                dtype_signed = tpu_is_data_type_signed(data_type_t[in_type])
                cur_ops = cmp_ops
                if dtype_signed:
                    cur_ops *= 2  # one more equal_select atomic op
                quotation_ops += cur_ops
            elif active_type == "soft_sign":
                quotation_ops += ar_ops + div_ops
            elif active_type == "sigmoid":
                quotation_ops += ar_ops * 12 + div_ops + taylor_ops
            elif active_type == "mish":
                quotation_ops += ar_ops * 15 + div_ops + taylor_ops + cmp_ops * 2 + lin_ops
            elif active_type == "sinh" or active_type == "cosh":
                quotation_ops += ar_ops * (10 + 2) + taylor_ops + div_ops
            elif active_type == "tanh":
                quotation_ops += ar_ops * (10 + 3) + taylor_ops + div_ops
            elif active_type == "exp":
                quotation_ops += ar_ops * 10 + taylor_ops
            elif active_type == "silu":  # sigmoid+mul
                quotation_ops += ar_ops * (12 + 1) + div_ops + taylor_ops
            elif active_type == "swish":  # mul+sigmoid+mul
                quotation_ops += ar_ops * (12 + 2) + div_ops + taylor_ops
            elif active_type == "elu":
                alpha = self.attrs["alpha"]
                quotation_ops += ar_ops * (10 + 1) + taylor_ops + cmp_ops
                if alpha != 1:
                    quotation_ops += ar_ops
            elif active_type == "erf":
                # calculate sign quotation_ops
                dtype_signed = tpu_is_data_type_signed(data_type_t[in_type])
                cur_quotation_ops = cmp_ops
                if dtype_signed:
                    cur_quotation_ops *= 2  # one more equal_select atomic op
                quotation_ops += cur_quotation_ops + ar_ops * (8 + 5 + 10) + taylor_ops * 2
            elif active_type == "gelu":
                dtype_signed = tpu_is_data_type_signed(data_type_t[in_type])
                cur_quotation_ops = cmp_ops
                if dtype_signed:
                    cur_quotation_ops *= 2  # one more equal_select atomic op
                quotation_ops += cur_quotation_ops + ar_ops * (8 + 5 + 10) + taylor_ops * 2
                quotation_ops += ar_ops * 4
            elif active_type == "ln":  # log
                quotation_ops += ar_ops * 8 + normalize_ops * 2 + taylor_ops + cmp_ops
            elif active_type == "log_sigmoid":
                quotation_ops += ar_ops * 20 + taylor_ops + normalize_ops * 2 + taylor_ops + cmp_ops
            elif active_type == "arcsinh" or active_type == "arccosh":
                quotation_ops += ar_ops * 12 + rsqrt_ops + cmp_ops * 2 + + normalize_ops * 2 + taylor_ops + cmp_ops
            elif active_type == "arctanh":
                quotation_ops += ar_ops * 11 + div_ops + normalize_ops * 2 + taylor_ops + cmp_ops
            elif active_type == "soft_plus":
                quotation_ops += ar_ops * 19 + taylor_ops + normalize_ops * 2 + taylor_ops + cmp_ops
            elif active_type == "tan":
                quotation_ops += ar_ops * 11 + taylor_ops + div_ops
            elif active_type == "sin":
                quotation_ops += ar_ops * 5 + taylor_ops
            elif active_type == "cos":
                quotation_ops += ar_ops * 4 + taylor_ops
            elif active_type == "arcsin":
                quotation_ops += ar_ops * 7 + taylor_ops
                quotation_ops += self.active("sqrt", in_shape, in_type, as_quotation=True) * 2
                quotation_ops += self.active("sign", in_shape, in_type, as_quotation=True)
            elif active_type == "arccos":
                quotation_ops += self.active("arcsin", in_shape, in_type, as_quotation=True) + ar_ops

            return quotation_ops

    def scale(self):  # calculation according to the test case
        in_shape, in_type = OpParse.getShapeType(self.op.operands[0])
        in_shape = self.reshape_to_4d(in_shape)
        ops = self.atomic_ar(in_shape, in_type)
        if len(self.op.operands) == 3:
            ops += self.atomic_ar(in_shape, in_type)
        if self.do_relu:
            ops += self.atomic_ar(in_shape, in_type)
            if ast.literal_eval(self.attrs["relu_limit"].split(":")[0]) > 0:
                ops += self.atomic_ar(in_shape, in_type)
        return ops

    def sigmoid(self):
        return self.active("sigmoid")

    def silu(self):
        return self.active("silu")

    def softmax(self):
        in_shape, in_type = OpParse.getShapeType(self.op.operands[0])
        dim = len(in_shape)
        axis = int(self.attrs["axis"].split(":")[0].strip())
        if axis < 0:
            axis = dim + axis

        outer_num, c, inner_num = 1, 1, 1
        for i in range(axis):
            outer_num *= in_shape[i]
        for i in range(axis, axis + 1):
            c *= in_shape[i]
        for i in range(axis + 1, dim):
            inner_num *= in_shape[i]

        if inner_num == 1:
            in_shape = [1, outer_num, 1, c]
            return self.atomic_ar(in_shape, in_type, "arith.sub") * 10 + self.atomic_ar(in_shape,
                                                                                        in_type,
                                                                                        "arith.div") + self.atomic_pord(
                in_shape, in_type, "pord.maxpooling") * 2 + self.atomic_sfu(in_shape, in_type,
                                                                            "sfu.taylor_4x")
        else:
            n, c, h, w = [c, outer_num, inner_num, 1]
            k = 1
            if c < NPU_NUM:
                k_max = int(NPU_NUM / c)
                for i in range(k_max, 0, -1):
                    if h % i == 0:
                        k = i
                        h /= k
                        if h < 16:
                            h *= k
                            k = 1
                            continue
                        else:
                            c *= k
                            break
            in_shape = [n, c, h, w]
            per_dim = [1, c, h, w]

            # pord_dim = [1, c, n, ALIGN(h * w, base_eu_num)]
            # pord_ker=[n,1]
            # pord_pad=[0,0,0,0]
            # pord_str=[1,1]
            # pord_dil=[1,1]
            # after on maxpooling2d operation,we get
            pord_out = [1, c, 1, ALIGN(h * w, base_eu_num)]
            # one avgpooling2d with the same in/out dims is also needed in softmax

            ops = 0
            if in_type != "f32":
                ops += self.atomic_ar(in_shape, in_type) * 2

            if self.attrs["log"] == "true":
                ops += self.atomic_ar(in_shape, in_type) * 2 + self.atomic_ar(per_dim, in_type) + self.active("ln",
                                                                                                              in_shape=per_dim,
                                                                                                              in_type=in_type,
                                                                                                              as_quotation=True)
            else:
                ops += self.atomic_ar(per_dim, in_type, "arith.div") + self.atomic_ar(in_shape, in_type, "arith.mul")

            return self.atomic_pord(pord_out, in_type, "pord.maxpooling") * 2 + self.atomic_ar(
                in_shape, in_type) + self.active("exp", in_shape=in_shape, in_type=in_type, as_quotation=True)

    def slice(self):  # No nodechip file
        return 0

    def squeeze(self):  # a usage of reshape
        return 0

    def sqrt(self):
        return self.active("sqrt")

    def permute(self, in_shape=None, in_type=None, order=None, as_quotation=False):
        if not as_quotation:
            in_shape, in_type = OpParse.getShapeType(self.op.operands[0])
            order = ast.literal_eval(self.attrs["order"])
        else:
            assert in_shape is not None
            assert in_type is not None
            assert order is not None

        type_len = tpu_data_type_size(in_type)
        dims = len(in_shape)
        raw_order = [i for i in range(dims)]
        tmp_order = [0] * FW_MAX_SHAPE_DIMS
        steps = [[0] * 4 for _ in range(FW_MAX_SHAPE_DIMS)]
        step_num = 0

        fixed_dim = 1
        left_dim = 1
        right_dim = 1
        trans_method = [0] * FW_MAX_SHAPE_DIMS
        trans_total_times = [0] * FW_MAX_SHAPE_DIMS

        for i in range(dims):
            if order[i] == raw_order[i]:
                fixed_dim *= in_shape[order[i]]
                continue
            pivot = 0
            for j in range(i + 1, dims):
                if order[i] == raw_order[j]:
                    pivot = j - i
                    break

            left_dim = 1
            right_dim = 1
            for j in range(i, dims):
                if j < pivot + i:
                    left_dim *= in_shape[raw_order[j]]
                else:
                    right_dim *= in_shape[raw_order[j]]

                if j + pivot < dims:
                    tmp_order[j] = raw_order[j + pivot]
                else:
                    tmp_order[j] = raw_order[i + j + pivot - dims]

            for j in range(i, dims):
                raw_order[j] = tmp_order[j]

            if left_dim != 1 and right_dim != 1:
                cur_shape, cur_trans_method, cur_max_trans_counts = get_transpose_info(fixed_dim, left_dim, right_dim,
                                                                                       type_len)
                # print(shape)
                # print(trans_method)
                # print(max_trans_counts)
                if trans_method == 0:
                    steps[step_num][0] = fixed_dim
                    steps[step_num][1] = left_dim
                    steps[step_num][2] = right_dim
                else:
                    steps[step_num][0] = cur_shape[0]
                    steps[step_num][1] = cur_shape[1]
                    steps[step_num][2] = cur_shape[2]
                    steps[step_num][3] = cur_shape[3]
                    trans_total_times[step_num] = cur_max_trans_counts
                    trans_method[step_num] = cur_trans_method
                step_num += 1
            fixed_dim *= in_shape[order[i]]
        for i in range(3, dims):
            if i != order[i]:
                break
            if i == dims - 1:
                step_num = 1
                trans_method[0] = 2  # TRANS_GDMA_NCH

        ops = 0
        for i in range(0, step_num):
            if trans_method[i] == 1:  # TRANS_NPU_N_SWITCH_W
                cur_shape = [steps[i][0], steps[i][1], steps[i][2], steps[i][3]]
                ops += self.atomic_ar(cur_shape, in_type, "arith.copy")
            elif trans_method[i] == 2:  # TRANS_GDMA_NCH
                ops += 0
            elif trans_method[i] == 3:  # TRANS_NPU_H_SWITCH_W
                cur_shape = [steps[i][0], steps[i][1], steps[i][2], steps[i][3]]
                ops += self.atomic_ar(cur_shape, in_type, "arith.copy")
            else:
                x, y, z = steps[i][0], steps[i][1], steps[i][2]
                cur_shape = [x, z, 1, y]
                if y >= z:
                    ops += self.atomic_transbc(in_type, cur_shape, "tsbc.cw_ts")
                else:
                    ops += self.atomic_transbc(in_type, cur_shape, "tsbc.wc_ts")

        if not as_quotation:
            return ops
        else:
            return ops, steps

    def transpose(self):
        return self.permute()

    def upsample(self):
        in_shape, in_type = OpParse.getShapeType(self.op.operands[0])
        scale_h = int(self.attrs["scale_h"].split(":")[0].strip())
        scale_w = int(self.attrs["scale_w"].split(":")[0].strip())
        return self.atomic_ar(in_shape, in_type) * scale_h * scale_w

    def weigh(self):
        return 0

    def input(self):
        return 0

    def scatternd(self):  # calculation according to the test case. No nodechip file
        in_shape, in_type = OpParse.getShapeType(self.op.operands[0])
        return self.atomic_ar(in_shape, in_type) * 6

    def cos(self):
        return self.active("cos")

    def cosh(self):
        return self.active("cosh")

    def sin(self):
        return self.active("sin")

    def sinh(self):
        return self.active("sinh")

    def elu(self):
        return self.active("elu")

    def erf(self):
        return self.active("erf")

    def exp(self):
        return self.active("exp")


def out_layers_details(module_parsered, output_path=None):
    total_types = []
    total_details = []

    for i in range(len(module_parsered.body.operations)):
        op = module_parsered.body.operations[i]
        # print(op)
        type = Operation.type(op)
        if type not in total_types:
            total_types.append(type)
        if type in ['top.None', 'func.return', 'top.Weight']:
            continue

        cur_instance = OpParse(op)

        if not total_details:
            total_details.append("\t".join(cur_instance.keys()) + "\n")
        else:
            total_details.append("\t".join([str(x) for x in cur_instance.values()]) + "\n")

        if output_path:
            with open(os.path.join(output_path, "LayersDetails.csv"), "w+") as f:
                f.writelines(total_details)
                f.write("total types:" + str(total_types))
        else:
            with open(r"./LayersDetails.csv", "w+") as f:
                f.writelines(total_details)
                f.write("total types:" + str(total_types))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlir_file", type=str, required=True, help="mlir model file ready to be parsed")
    parser.add_argument("--output_path", type=str, help="output path of the parsed csv")
    args = parser.parse_args()
    module_parsered = MlirParser(args.mlir_file)
    out_layers_details(module_parsered, args.output_path)
