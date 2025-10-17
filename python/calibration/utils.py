import re
import datetime

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


def parse_method_list(input_str):
    return input_str.split(',')


def is_fuseop(op_name):
    return re.match(r'^fused\[".*?"\]$', op_name)


def split_fuseop(op_name):
    if is_fuseop(op_name):
        new_ops = re.findall(r'"([^"]+)"', op_name)
        return new_ops
    else:
        return [op_name]


def gen_shape_pattern_qtable(shape_fp_layers, transformer_fp_layers, args, flag):
    chip = args.chip
    cali_table_name = args.calibration_table
    shape_fp_layers_set = set(
        shape_fp_layers)  #use "set" for the efficiency of processing probably complex models
    transformer_fp_layers_for_qtable = [
        item for item in transformer_fp_layers if item not in shape_fp_layers_set
    ]
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
        quantize_table = args.quantize_table + "shape_pattern" or "shape_pattern_qtable"

    with open(quantize_table, "w") as f:
        f.write("# genetated time: {}\n".format(datetime.datetime.now()))
        f.write("# chip: {}  shape_mix_mode: {}  pattern_mix_mode: {}\n".format(
            chip, shape_mix_mode, pattern_mix_mode))
        f.write("# number of {} layer(shape): {}\n".format(shape_mix_mode, len(shape_fp_layers)))
        f.write("# number of {} layer(pattern): {}\n".format(pattern_mix_mode,
                                                             len(transformer_fp_layers_for_qtable)))
        if args.part_quantize and flag == 0:
            f.write("# part_quantize \n")
        f.write("###\n")
        f.write("# op_name   quantize_mode\n")
        f.write("# mix_prec layers by op_shape identification\n")
        for layer in shape_fp_layers:
            f.write("{} {}\n".format(layer, shape_mix_mode))
        f.write("# mix_prec layers by transformer_pattern identification\n")
        for layer in transformer_fp_layers_for_qtable:
            f.write("{} {}\n".format(layer, pattern_mix_mode))
