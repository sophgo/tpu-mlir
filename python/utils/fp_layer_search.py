from utils.mlir_parser import MlirParser
import datetime

FLOAT_MAP = {
    "bm1684x": "F16",
    "bm1684": "F32",
    "bm1688": "F16",
    "cv183x": "BF16",
    "cv182x": "BF16",
    "cv181x": "BF16",
    "cv180x": "BF16",
    "cv180x": "BF16"
}

chip_support_mix_fp_type = {
    "bm1684x": ["F16", "F32"],
    "bm1688": ["F16", "F32"],
    "bm1684": ["F32"],
    "cv183x": ["BF16"],
    "cv182x": ["BF16"],
    "cv181x": ["BF16"],
    "cv180x": ["BF16"],
    "cv186x": ["BF16"]
}

class FpLayerSearcher:
    def __init__(self, args):
        self.args = args
        self.fp32_mlir = args.mlir_file
        self.chip = args.chip
        self.fpfwd_inputs = args.fpfwd_inputs
        self.fpfwd_outputs = args.fpfwd_outputs
        self.fpfwd_blocks = args.fpfwd_blocks
        self.quantize_free_table = args.quantize_free_table
        if args.fp_type == 'auto':
            self.mix_mode = FLOAT_MAP[self.chip]
        else:
            self.mix_mode = args.fp_type
            if args.fp_type not in chip_support_mix_fp_type[self.chip]:
                print('parameter error, fp_type:{args.fp_type} not support by {self.chip}')
                exit(1)
        self.parser = MlirParser(args.mlir_file)

    def parse_inputs(self, input_str):
        ops = [op for op in input_str.split(',')]
        return ops

    def parse_blocks(self,block_list):
        parsed_blocks = []
        if not block_list:
            return []
        for block in block_list:
            block_starts,block_ends = block.split(':')
            block_start = [start for start in block_starts.split(',')]
            block_end = [start for start in block_ends.split(',')]
            block = [block_start, block_end]
            parsed_blocks.append(block)
        return parsed_blocks

    def get_fpfwd_inputs(self, input_names, parser, mode):
        if isinstance(input_names, str):
            input_names = self.parse_inputs(input_names)
        op_names = []
        for op_name in input_names:
            if not parser.get_op_by_op_name(op_name):
                print("warning, op: {} not found in mlir file, please check it again".format(op_name))
        if mode == 0:
            for op_name in input_names:
                op_names.extend(parser.get_all_pre_ops_by_op_name(op_name))
        else:
            for op_name in input_names:
                op_names.extend(parser.get_all_next_ops_by_op_name(op_name))
        op_names = list(set(op_names))
        return op_names

    def get_blocks(self, input_str, parser):
        blocks = self.parse_blocks(input_str)
        blocks_name_set = set()
        for block in blocks:
            pre_lists = set(self.get_fpfwd_inputs(block[1], parser, 0))
            next_lists = set(self.get_fpfwd_inputs(block[0], parser, 1))
            block_name_set = pre_lists & next_lists
            blocks_name_set.update(block_name_set)
        return list(blocks_name_set)

    def print_log_info(self, fp_layer_list):
        with open(self.quantize_free_table, "w") as f:
            f.write("# genetated time: {}\n".format(datetime.datetime.now()))
            f.write("# chip: {}  mix_mode: {}\n".format(self.chip, self.mix_mode))
            f.write("# number of {} layer: {}\n".format(self.mix_mode, len(fp_layer_list)))
            f.write("###\n")
            f.write("# op_name   quantize_mode\n")
            for layer in fp_layer_list:
                f.write("{} {}\n".format(layer, self.mix_mode))
    def run(self):
        fp_layer_set = set()
        if self.fpfwd_inputs:
            fp_layer_set.update(set(self.get_fpfwd_inputs(self.fpfwd_inputs, self.parser, 0)))
        if self.fpfwd_outputs:
            fp_layer_set.update(set(self.get_fpfwd_inputs(self.fpfwd_outputs, self.parser, 1)))
        if self.fpfwd_blocks:
            fp_layer_set.update(set(self.get_blocks(self.fpfwd_blocks, self.parser)))
        self.print_log_info(fp_layer_set)
