#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import builtins
import copy

from utils.preprocess import get_preprocess_parser, preprocess
from tools.model_transform import get_model_transform
from utils.mlir_parser import MlirParser, Operation
import argparse, re, ast, math, os

# 1684X
NPU_NUM = 64
__base_eu_num = 16
__eu_num_map = {
    "TOP_F32": __base_eu_num,
    "TOP_I32": __base_eu_num,
    "TOP_SI32": __base_eu_num,
    "TOP_UI32": __base_eu_num,
    "TOP_F16": __base_eu_num * 2,
    "TOP_BF16": __base_eu_num * 2,
    "TOP_I16": __base_eu_num * 2,
    "TOP_UI16": __base_eu_num * 2,
    "TOP_SI16": __base_eu_num * 2,
    "TOP_I8": __base_eu_num * 4,
    "TOP_UI8": __base_eu_num * 4,
    "TOP_SI8": __base_eu_num * 4,
}

simple_op_map = {
    "maxpool": "pool",
    "avgpool": "pool"
}

# constant multiple times of out_shape
complex_op_map = {
    "add": "ar",
    "avgpool": "pool",
    "batchnorm": "ar",
    "clip": "ar",
    "div": "ar",
    "leakyrelu": "ar",
    "log": "ar",
    "maxpool": "pool",
    "relu": "ar",
    "sigmoid": "ar",
    "silu": "ar"
}
hard_op_factor_map = {
    "sigmoid": 4,
    "silu": 5,
    "batchnorm": 2,
    "clip": 2,
    "div": 5,
    "log": 4
}


def ALIGN(x, a):
    return math.ceil(x / a) * a


def EU_NUM(dtype):
    return __eu_num_map[dtype]


class Operation_V2(Operation):
    def __int__(self, op, body, idx):
        super().__init__(op, body, idx)

    @staticmethod
    def name(op):  # rewrite method
        loc = op.location
        if loc == "loc(unknown)":
            return None
        # loc(fused["pool1", "pool1_mask"]) => pool1
        return re.search(r'\"(.*)\"', str(loc)).group(1)


class SimpleOps(object):
    def __init__(self, in_dict, out_shape, dtype, attrs):
        assert type(in_dict) == dict
        assert type(out_shape) == list
        assert type(dtype) == str

        self.in_dict = in_dict
        self.out_shape = out_shape
        self.dtype = dtype
        self.attrs = attrs
        if 'do_relu' in attrs:
            if attrs['do_relu'].lower() == "false":
                self.relu = False
            else:
                self.relu = True
        else:
            self.relu = False

    def getNumElements(self, shape_in_list):
        ele = 1
        for i in range(len(shape_in_list)):
            ele *= shape_in_list[i]
        return ele

    def add(self):
        return self.getNumElements(self.out_shape)

    def pool(self):  # max,avg
        kernel_shape = ast.literal_eval(self.attrs["kernel_shape"])
        kernel_mul = 1
        for i in kernel_shape:
            kernel_mul *= i
        if self.relu:
            return self.getNumElements(self.out_shape) * (kernel_mul + 1)
        else:
            return self.getNumElements(self.out_shape) * kernel_mul

    def batchnorm(self):
        return 2 * self.getNumElements(self.out_shape)

    def clip(self):
        return 2 * self.getNumElements(self.out_shape)

    def concat(self):
        return 0

    def conv(self, is_arch=False):
        group = ast.literal_eval(self.attrs["group"].split(":")[0])
        kernel_shape = ast.literal_eval(self.attrs["kernel_shape"])
        kernel_mul = 1
        for i in kernel_shape:
            kernel_mul *= i

        extra = 0
        in_key = next(iter(self.in_dict))
        in_shape, dtype = self.in_dict[in_key]
        if len(self.in_dict) > 2:
            extra += 1
        if self.relu:
            extra += 1

        ic = in_shape[1]

        return self.getNumElements(self.out_shape) * (2 * kernel_mul * ic / group + extra)

    def deconv(self, is_arch=False):
        group = ast.literal_eval(self.attrs["group"].split(":")[0])
        kernel_shape = ast.literal_eval(self.attrs["kernel_shape"])
        kernel_mul = 1
        for i in kernel_shape:
            kernel_mul *= i

        extra = 0
        in_key = next(iter(self.in_dict))
        in_shape, dtype = self.in_dict[in_key]
        if len(self.in_dict) > 2:
            extra += 1
        if self.relu:
            extra += 1

        oc = self.out_shape[1]

        return self.getNumElements(in_shape) * (2 * kernel_mul * oc / group + extra)

    def depth2space(self):
        return 0

    def div(self):
        return self.getNumElements(self.out_shape)

    def leakyrelu(self):
        return self.getNumElements(self.out_shape)

    def log(self):
        return 4 * self.getNumElements(self.out_shape)

    def matmul(self):
        extra = 0
        if self.relu:
            extra += 1
        if len(self.in_dict) > 2:
            extra += 1

        if self.attrs['right_transpose'].lower() == "false":
            right_transpose = False
        else:
            right_transpose = True

        # assert len(self.in_dict) == 2
        a_key = next(iter(self.in_dict))
        a_shape = self.in_dict[a_key][0]
        self.in_dict.pop(a_key)
        b_key = next(iter(self.in_dict))
        b_shape = self.in_dict[b_key][0]

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

        return batch * (2 * k + extra)

    def mul(self):
        extra = 0
        if self.relu:
            extra += 1
        input_nums = len(self.in_dict)
        return self.getNumElements(self.out_shape) * (input_nums - 1 + extra)

    def mulconst(self):
        extra = 0
        if self.relu:
            extra += 1
        return self.getNumElements(self.out_shape) * (1 + extra)

    def permute(self):
        return 0

    def relu(self):
        return self.getNumElements(self.out_shape)

    def reshape(self):
        return 0

    def scale(self):
        extra = 0
        if self.relu:
            extra += 1
        return self.getNumElements(self.out_shape) * (2 + extra)

    def sigmoid(self):
        return 4 * self.getNumElements(self.out_shape)

    def silu(self):
        return 5 * self.getNumElements(self.out_shape)

    def slice(self):
        return 0

    def softmax(self):
        assert len(self.in_dict) == 1
        layername = next(iter(self.in_dict))
        in_shape = self.in_dict[layername][0]
        return self.getNumElements(in_shape) * 5

    def squeeze(self):
        return 0

    def upsample(self):
        if self.relu:
            return 2 * self.getNumElements(self.out_shape)
        else:
            return self.getNumElements(self.out_shape)

    def weigh(self):
        return 0

    def input(self):
        return 0


class ComplexOps(object):  # currently only support input shape=(n,c,h,w)
    def __init__(self, in_dict, out_shape, dtype, attrs):
        assert type(in_dict) == dict
        assert type(out_shape) == list
        assert type(dtype) == str

        self.in_dict = in_dict
        self.out_shape = out_shape
        self.dtype = dtype
        self.attrs = attrs
        if 'do_relu' in attrs:
            if attrs['do_relu'].lower() == "false":
                self.relu = False
            else:
                self.relu = True
        else:
            self.relu = False

    def ar(self, op_name):  # use the dtype of output, refer to the "Class ar_op(bdc_base)" in opdef_1684x.py
        if op_name in hard_op_factor_map:
            factor = hard_op_factor_map[op_name]
        else:
            factor = 1

        n, c, h, w = self.out_shape
        hw = h * w
        c = ALIGN(c, NPU_NUM)
        hw = ALIGN(hw, EU_NUM(self.dtype))
        return n * c * hw * factor

    def pool(self, op_name):
        n, c, h, w = self.out_shape
        kernel_shape = ast.literal_eval(self.attrs["kernel_shape"])
        kernel_mul = 1
        for i in kernel_shape:
            kernel_mul *= i
        factor = 1
        c = ALIGN(c, NPU_NUM)
        w = ALIGN(h * w, EU_NUM(self.dtype))
        h = 1
        if self.relu:
            return n * c * h * w * (factor * kernel_mul + 1)
        else:
            return n * c * h * w * factor * kernel_mul

    def conv(self, op_name):
        group = ast.literal_eval(self.attrs["group"].split(":")[0])
        kh, kw = ast.literal_eval(self.attrs["kernel_shape"])
        extra = 0
        in_key = next(iter(self.in_dict))
        in_shape, dtype = self.in_dict[in_key]
        if len(self.in_dict) > 2:
            extra += 1
        if self.relu:
            extra += 1

        n, ic, ih, iw = in_shape
        n, oc, oh, ow = self.out_shape
        ic = ALIGN(ic, NPU_NUM)
        ow = ALIGN(oh * ow, EU_NUM(dtype))
        oh = 1
        oc = ALIGN(oc, NPU_NUM)
        return n * oc * oh * ow * (2 * ic * kh * kw / group + extra)

    def deconv(self, is_arch=False):
        group = ast.literal_eval(self.attrs["group"].split(":")[0])
        kh, kw = ast.literal_eval(self.attrs["kernel_shape"])
        extra = 0
        in_key = next(iter(self.in_dict))
        in_shape, dtype = self.in_dict[in_key]
        if len(self.in_dict) > 2:
            extra += 1
        if self.relu:
            extra += 1

        n, ic, ih, iw = in_shape
        n, oc, oh, ow = self.out_shape
        ic = ALIGN(ic, NPU_NUM)
        iw = ALIGN(ih * iw, EU_NUM(dtype))
        ih = 1
        oc = ALIGN(oc, NPU_NUM)
        return n * ic * ih * iw * (2 * oc * kh * kw / group + extra)

    def concat(self, name):
        return 0

    def depth2space(self, name):
        return 0

    def matmul(self, name):
        backup_dict = copy.deepcopy(self.in_dict)
        if len(self.in_dict) > 2:
            extra = 1
        a_key = next(iter(self.in_dict))
        m, lk = self.in_dict[a_key][0]
        self.in_dict.pop(a_key)
        b_key = next(iter(self.in_dict))
        rk, n = self.in_dict[b_key][0]
        if self.attrs["left_transpose"].lower() == "true":
            lk, m = m, lk
        if self.attrs["right_transpose"].lower() == "true":
            n, rk = rk, n

        assert lk == rk
        k = lk
        k = ALIGN(k, EU_NUM(backup_dict[a_key][1]))
        m = ALIGN(m, NPU_NUM)
        return m * n * (2 * k + extra)

    def mul(self, op_name):
        extra = 0
        if self.relu:
            extra += 1
        input_nums = len(self.in_dict)
        n, c, h, w = self.out_shape

        hw = h * w
        c = ALIGN(c, NPU_NUM)
        hw = ALIGN(hw, EU_NUM(self.dtype))
        return n * c * hw * (input_nums - 1 + extra)

    def mulconst(self, op_name):
        extra = 0
        if self.relu:
            extra += 1
        n, c, h, w = self.out_shape

        hw = h * w
        c = ALIGN(c, NPU_NUM)
        hw = ALIGN(hw, EU_NUM(self.dtype))
        return n * c * hw * (1 + extra)

    def permute(self, name):
        return 0

    def reshape(self, name):
        return 0

    def scale(self, op_name):
        extra = 0
        if self.relu:
            extra += 1
        n, c, h, w = self.out_shape

        hw = h * w
        c = ALIGN(c, NPU_NUM)
        hw = ALIGN(hw, EU_NUM(self.dtype))
        return n * c * hw * (2 + extra)

    def slice(self, name):
        return 0

    def softmax(self, op_name):
        extra = 0
        if self.relu:
            extra += 1
        key = next(iter(self.in_dict))
        n, c, h, w = self.in_dict[key][0]

        hw = h * w
        c = ALIGN(c, NPU_NUM)
        hw = ALIGN(hw, EU_NUM(self.in_dict[key][1]))
        return n * c * hw * (5 + extra)

    def squeeze(self, name):
        return 0

    def upsample(self, op_name):
        n, c, h, w = self.out_shape

        hw = h * w
        c = ALIGN(c, NPU_NUM)
        hw = ALIGN(hw, EU_NUM(self.dtype))
        if self.relu:
            return 2 * n * c * hw
        else:
            return n * c * hw

    def weigh(self, name):
        return 0

    def input(self, name):
        return 0


def out_layers_details(module_parsered, output_path=None):
    name2loc = {}
    name2attr = {}
    name2append_attr = {}
    name2out_shape = {}
    name2out_dtype = {}

    total_types = []
    total_layer_ops = 0
    total_tiu_ops = 0

    total_details = []
    total_details.append(
        "layer_name" + "\t" + "layer_type" + "\t" + "loc" + "\t" + "in_shape" + "\t" + "out_shape" + "\t" + "cur_layer_ops" + "\t" + "total_layer_ops""\t" + "cur_tiu_ops" + "\t" + "total_tiu_ops" + "\t" + "cur_tiu_utility" + "\t" + "total_tiu_utility" + "\n")

    for i in range(len(module_parsered.body.operations)):
        cur_layer_ops = 0
        op = module_parsered.body.operations[i]
        type = Operation_V2.type(op)

        if type not in total_types:
            total_types.append(type)

        if type in ['top.None', 'func.return']:
            continue

        name = Operation_V2.name(op)
        loc = Operation_V2.loc(op)
        attrs = Operation_V2.attrs(op)
        append_attr = Operation_V2.append_attr(op, module_parsered.attrs)
        output_shape = Operation_V2.shape(op)
        opds = Operation_V2.operands_v2(op, module_parsered.body, i)

        name2loc[name] = loc
        name2attr[name] = attrs
        name2append_attr[name] = append_attr
        name2out_shape[name] = output_shape
        name2out_dtype[name] = ast.literal_eval(append_attr['module.state'])
        name2in_shape_type = {}
        for op_name in opds:
            name2in_shape_type[op_name] = [name2out_shape[op_name], name2out_dtype[name]]

        if type in ['top.Weight']:
            continue

        name2in_shape_type2 = copy.deepcopy(
            name2in_shape_type)  # deepcopy,avoid KEYERROR when count tiu_ops after layer_ops

        cal_type = re.search(r"(?<=\.)\S+$", type).group(
            0).lower()  # match, for example:Top.Mul, "Mul" will be matched and trans into lower case
        soft_detail = SimpleOps(name2in_shape_type, output_shape, name2out_dtype[name], attrs)
        if cal_type in simple_op_map:
            cur_layer_ops = getattr(soft_detail, simple_op_map[cal_type])()
        else:
            cur_layer_ops = getattr(soft_detail, cal_type)()

        hard_detail = ComplexOps(name2in_shape_type2, output_shape, name2out_dtype[name], attrs)
        if cal_type in complex_op_map:
            cur_tiu_ops = getattr(hard_detail, complex_op_map[cal_type])(cal_type)
        else:
            cur_tiu_ops = getattr(hard_detail, cal_type)(cal_type)

        cur_layer_ops = int(cur_layer_ops)
        cur_tiu_ops = int(cur_tiu_ops)
        total_layer_ops += cur_layer_ops
        total_tiu_ops += cur_tiu_ops
        if cur_layer_ops != 0 and cur_tiu_ops != 0:
            cur_tiu_utility = round(cur_layer_ops / cur_tiu_ops, 2)
        else:
            cur_tiu_utility = "None"
        if total_layer_ops != 0 and total_tiu_ops != 0:
            total_tiu_utility = round(total_layer_ops / total_tiu_ops, 2)
        else:
            total_tiu_utility = "None"

        # print(cur_layer_ops, "\t", total_layer_ops)
        # print(cur_tiu_ops, "\t", total_tiu_ops)
        # print(cur_tiu_utility, total_tiu_utility)
        # print("\n")

        total_details.append(name + "\t" +
                             type + "\t" +
                             loc + "\t" +
                             str(name2in_shape_type) + "\t" +
                             str([output_shape, name2out_dtype[name]]) + "\t" +
                             str(cur_layer_ops) + "\t" +
                             str(total_layer_ops) + "\t" +
                             str(cur_tiu_ops) + "\t" +
                             str(total_tiu_ops) + "\t" +
                             str(cur_tiu_utility) + "\t" +
                             str(total_tiu_utility) +
                             "\n")

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
