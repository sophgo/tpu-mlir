import re
import datetime
from typing import Optional
from dataclasses import dataclass
from utils.mlir_parser import MlirParser
import warnings

FLOAT_MAP = {
    "bm1684x": "F16",
    "bm1684": "F32",
    "cv183x": "BF16",
    "cv182x": "BF16",
    "cv181x": "BF16",
    "cv180x": "BF16",
    "bm1688": "F16",
    "cv186x": "F16",
    "bm1690": "F16",
    "cv184x": "BF16"
}

chip_support_mix_fp_type = {
    "bm1684x": ["F16", "F32"],
    "bm1688": ["F16", "F32"],
    "cv186x": ["F16", "F32"],
    "bm1684": ["F32"],
    "cv183x": ["BF16"],
    "cv182x": ["BF16"],
    "cv181x": ["BF16"],
    "cv180x": ["BF16"],
    "cv184x": ["BF16"]
}

calibration_methods = ['mse', 'max', 'kl', 'percentile9999', 'aciq_gauss', 'aciq_laplace']


@dataclass
class calibration_result:
    max_val: float = 1.0
    min_val: float = -1.0
    abs_max: float = 1.0  # abs max
    mse: Optional[float] = 0.0
    kl: Optional[float] = 0.0
    p99_min: Optional[float] = -1.0  #percentile9999 min
    p99_max: Optional[float] = 1.0  #percentile9999 max
    p99: Optional[float] = 0.0  #percentile9999 threshold in symmetric
    aciq_g: Optional[float] = 0.0
    aciq_l: Optional[float] = 0.0

    def canonicalize(self):
        if self.abs_max > max(abs(self.min_val), abs(self.max_val)):
            self.abs_max = max(abs(self.min_val), abs(self.max_val))
        if self.p99_min < self.min_val:
            self.p99_min = self.min_val
        if self.p99_max > self.max_val:
            self.p99_max = self.max_val
        if self.p99 > self.abs_max:
            self.p99 = self.abs_max
        if self.mse > self.abs_max:
            self.mse = self.abs_max
        if self.kl > self.abs_max:
            self.kl = self.abs_max
        if self.aciq_g > self.abs_max:
            self.aciq_g = self.abs_max
        if self.aciq_l > self.abs_max:
            self.aciq_l = self.abs_max

    def set_unsigned_all(self, v: float):
        self.max_val = v
        self.min_val = 0.0
        self.abs_max = v  # abs max
        self.mse = v
        self.kl = v
        self.p99_min = 0.0  #percentile9999 min
        self.p99_max = v  #percentile9999 max
        self.p99 = v  #percentile9999 threshold in symmetric
        self.aciq_g = v
        self.aciq_l = v

    def set_signed_all(self, v: float):
        self.max_val = v
        self.min_val = -v
        self.abs_max = v  # abs max
        self.mse = v
        self.kl = v
        self.p99_min = -v  #percentile9999 min
        self.p99_max = v  #percentile9999 max
        self.p99 = v  #percentile9999 threshold in symmetric
        self.aciq_g = v
        self.aciq_l = v

    def get_threshold(self, method: str):
        if method == 'max':
            return self.abs_max
        elif method == 'mse':
            return self.mse
        elif method == 'kl':
            return self.kl
        elif method == 'percentile9999':
            return self.p99
        elif method == 'aciq_gauss':
            return self.aciq_g
        elif method == 'aciq_laplace':
            return self.aciq_l
        else:
            raise RuntimeError("not support cali method: {}".format(method))


def parse_calibration_methods(cali_method: list, debug_cmd: dict):
    cali_methods = []
    if len(debug_cmd) == 0:
        methods_d = []
    else:
        methods_d = [k.lower() for k in debug_cmd.keys()]
    methods_d = [k for k in methods_d if k in calibration_methods]
    if len(methods_d) > 0:
        return methods_d
    if 'percentile9999' in debug_cmd or 'percentile9999' in cali_method:
        cali_methods.append('percentile9999')
    if 'kl' in debug_cmd or 'kl' in cali_method:
        cali_methods.append('kl')
    if 'mse' in debug_cmd or 'mse' in cali_method:
        cali_methods.append('mse')
    if 'max' in debug_cmd or 'max' in cali_method:
        cali_methods.append('max')
    if 'aciq_gauss' in debug_cmd or 'aciq_gauss' in cali_method:
        cali_methods.append('aciq_gauss')
    if 'aciq_laplace' in debug_cmd or 'aciq_laplace' in cali_method:
        cali_methods.append('aciq_laplace')
    return cali_methods


def parse_method_list(input_str):
    return input_str.split(',')


def compactable_method_list(method_list: list):
    cali_methods = []
    for method in method_list:
        if method in calibration_methods:
            cali_methods.append(method)
        elif method in ['use_' + m for m in calibration_methods]:
            cali_methods.append(method[4:])
    return cali_methods


def compactable_cmd_method_list(debug_cmd: dict):
    debug_cmds = dict()
    if len(debug_cmd) == 0:
        return debug_cmds
    k_l = [x.lower() for x in debug_cmd.keys()]
    for k, v in debug_cmd.items():
        compact = ['use_' + x for x in calibration_methods]
        if k.lower() in compact and k[4:] not in k_l:
            debug_cmds[k[4:].lower()] = v
        else:
            debug_cmds[k] = v
    return debug_cmds


def is_fuseop(op_name):
    return re.match(r'^fused\[".*?"\]$', op_name)


def split_fuseop(op_name):
    if is_fuseop(op_name):
        new_ops = re.findall(r'"([^"]+)"', op_name)
        return new_ops
    else:
        return [op_name]


def fuseop_list_append(op_name, fuseop_list):
    if is_fuseop(op_name):
        new_ops = re.findall(r'"([^"]+)"', op_name)
        if op_name not in fuseop_list:
            fuseop_list[op_name] = new_ops
            fuseop_list[new_ops[0]] = op_name
    return


def get_no_fused_tensors(parser: MlirParser, all_tensors: list):
    tensor_list = []
    fused_pattern = re.compile(r'^fused\[(.*?)\]$')
    for op in all_tensors:
        match = fused_pattern.match(op)
        if match:
            fused_ops = op.split('["')[1].split('"]')[0].split(', ')
            tensor_list.extend([fused_op.strip('"') for fused_op in fused_ops])
            has_next = False
            for fused_op in fused_ops:
                fused_op = fused_op.strip('"')
                if parser.get_next_op_by_op_name(fused_op):
                    has_next = True
                    break
            if has_next:
                for fused_op in fused_ops:
                    fused_op = fused_op.strip('"')
                    if not parser.get_next_op_by_op_name(fused_op):
                        try:
                            tensor_list.remove(fused_op)
                        except ValueError:
                            warnings.warn(f"无法从 tensor_list 中移除 '{fused_op}'，因为它不存在。")
        else:
            tensor_list.append(op)
    return tensor_list


def gen_shape_pattern_qtable(shape_fp_layers, transformer_fp_layers, args, flag, logs=None):
    chip = args.chip
    cali_table_name = args.calibration_table
    if args.fp_type == 'auto':
        shape_mix_mode = FLOAT_MAP[args.chip]
        pattern_mix_mode = FLOAT_MAP[args.chip]
    else:
        shape_mix_mode = args.fp_type
        pattern_mix_mode = args.fp_type
        if args.fp_type not in chip_support_mix_fp_type[args.chip]:
            print('parameter error, fp_type:{args.fp_type} not support by {args.chip}')
            exit(1)

    if '/' in cali_table_name:
        last_index = cali_table_name.rfind('/')
        quantize_table = cali_table_name[:last_index + 1] + "shape_pattern_qtable"
    else:
        if args.quantize_table:
            quantize_table = args.quantize_table + "_shape_pattern_part"
        else:
            quantize_table = "shape_pattern_qtable"

    with open(quantize_table, "w") as f:
        f.write("# genetated time: {}\n".format(datetime.datetime.now()))
        f.write("# chip: {}  shape_mix_mode: {}  pattern_mix_mode: {}\n".format(
            chip, shape_mix_mode, pattern_mix_mode))
        f.write("# number of {} layer(shape): {}\n".format(shape_mix_mode, len(shape_fp_layers)))
        f.write("# number of {} layer(pattern): {}\n".format(pattern_mix_mode,
                                                             len(transformer_fp_layers)))
        if args.part_quantize and flag == 0:
            f.write("# part_quantize \n")

        if logs:
            f.write("###\n")
            f.write("# Match_Pattern logs:\n")
            for line in logs:
                f.write("#" + str(line) + "\n")

        f.write("###\n")
        f.write("# op_name   quantize_mode\n")
        f.write("# mix_prec layers by op_shape identification\n")
        for layer in shape_fp_layers:
            f.write("{} {}\n".format(layer, shape_mix_mode))
        f.write("# mix_prec layers by transformer_pattern identification\n")
        for layer in transformer_fp_layers:
            f.write("{} {}\n".format(layer, pattern_mix_mode))


class CalibrationTable:

    def __init__(self, table):
        self.headers, self.thresholds_map = self.parse(table)

    def parse(self, table):
        thresholds_map = dict()
        headers = []
        with open(table, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if len(line) == 0:
                    continue
                if line.startswith('#'):
                    if line.startswith('#asym_op') or line.startswith('#int4_th'):
                        break
                    headers.append(line)
                    continue
                # op_name    threshold    min    max
                fields = line.split(' ')
                if len(fields) != 4:
                    print("Table format should be 'op_name, threshold, min, max'")
                    raise RuntimeError("Error with parse {} in {}".format(line, table))

                op_name, threshold, _min, _max = fields
                thresholds_map[op_name] = [float(threshold), float(_min), float(_max)]
        return headers, thresholds_map

    def dump(self, dest_table):
        with open(dest_table, "w") as f:
            for line in self.headers:
                f.write(line + "\n")
            for k, v in self.thresholds_map.items():
                f.write("{} {:.7f} {:.7f} {:.7f}\n".format(k, *v))

    def update_to(self, dest_table, target_op, new_threshold):
        with open(dest_table, "w") as f:
            for line in self.headers:
                f.write(line + "\n")
            for k, v in self.thresholds_map.items():
                if k == target_op:
                    f.write("{} {:.7f} {:.7f} {:.7f}\n".format(k, new_threshold, v[1], v[2]))
                else:
                    f.write("{} {:.7f} {:.7f} {:.7f}\n".format(k, *v))
