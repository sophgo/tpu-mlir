import re

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
    "mars3": "BF16"
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
    "mars3": ["BF16"]
}


def parse_method_list(input_str):
    return input_str.split(',')


def is_fuseop(op_name):
    return re.match(r'^fused\[".*?"\]$', op_name)


def split_fuseop(op_name):
    if is_fuseop(op_name):
        new_ops = re.findall(r'"([^"]+)"', op_name)
        return new_ops[0]
    else:
        return op_name
