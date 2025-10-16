import pymlir
import numpy as np
import mlir.dialects.top as top
from calibration.data_selector import DataSelector
from utils.mlir_parser import MlirParser
from utils.preprocess import preprocess


class ModelModifier:

    def __init__(self, args, data_selector: DataSelector):
        self.args = args
        self.module = pymlir.module()
        self.module.load(args.mlir_file)
        self.parser = MlirParser(args.mlir_file)
        self.batch_size = self.parser.get_batch_size()
        self.input_num = self.parser.get_input_num()
        self.data_selector = data_selector
        self.data_list = data_selector.data_list
        self.args.input_num = len(self.data_list)
        if data_selector.all_image or data_selector.all_yuv:
            n = self.args.input_num % self.batch_size
            if n != 0:
                for i in range(self.batch_size - n):
                    self.data_list.append(self.data_list[i])
                    self.args.input_num += 1
            self.args.input_num = self.args.input_num // self.batch_size
        self.unranked_type = None

    def init_ppa(self):
        self.ppa_list = []
        for i in range(self.input_num):
            tmp = preprocess()
            tmp.load_config(self.parser.get_input_op_by_idx(i))
            self.ppa_list.append(tmp)

    def load_net_weights(self):
        weight_file = self.parser.module_weight_file
        print('load weights from {}'.format(weight_file))
        weights = np.load(weight_file)
        self.module_weights = {k: weights[k] for k in weights}

    def load_net_inputs(self):
        self.input_tensors = {}

        if self.data_selector.all_image or self.data_selector.all_yuv:
            batched_inputs = [[] for i in range(self.input_num)]
        else:
            batched_inputs = {}
        only_one = len(self.module.input_names) == 1
        for data_idx, data in enumerate(self.data_list):
            if self.data_selector.all_npz:
                x = np.load(data)
                batch_idx = len(self.input_tensors)
                if batch_idx not in self.input_tensors:
                    self.input_tensors[batch_idx] = {}
                if only_one:
                    assert (len(x.files) == 1)
                    n0 = self.module.input_names[0]
                    n1 = x.files[0]
                    if n1 in batched_inputs:
                        batched_inputs[n1] = np.concatenate(
                            [batched_inputs[n1], x[n1].astype(np.float32)], axis=0)
                    else:
                        batched_inputs[n1] = x[n1].astype(np.float32)
                    if batched_inputs[n1].shape[0] >= self.batch_size:
                        self.input_tensors[batch_idx][n0] = batched_inputs[n1][:self.batch_size]
                        batched_inputs[n1] = batched_inputs[n1][self.batch_size:]

                else:
                    for input in self.module.input_names:
                        assert (input in x)
                        if input in batched_inputs:
                            batched_inputs[input] = np.concatenate(
                                [batched_inputs[input], x[input].astype(np.float32)], axis=0)
                        else:
                            batched_inputs[input] = x[input].astype(np.float32)
                        real_batch_size = self.parser.get_op_by_op_name(input).shape[0]
                        self.input_tensors[batch_idx][input] = batched_inputs[
                            input][:real_batch_size]
                        batched_inputs[input] = batched_inputs[input][real_batch_size:]

            elif self.data_selector.all_image or self.data_selector.all_yuv:
                inputs = [s.strip() for s in data.split(',')]
                assert (self.input_num == len(inputs))
                for i in range(self.input_num):
                    batched_inputs[i].append(inputs[i])
                if (data_idx + 1) % self.batch_size == 0:
                    batch_idx = (data_idx + 1) // self.batch_size - 1
                    self.input_tensors[batch_idx] = {}
                    for i in range(self.input_num):
                        x = self.ppa_list[i].run(','.join(batched_inputs[i]))
                        name = self.ppa_list[i].input_name
                        self.input_tensors[batch_idx][name] = x
                        batched_inputs = [[] for i in range(self.input_num)]

            else:
                self.input_tensors[data_idx] = {}
                inputs = [s.strip() for s in data.split(',')]
                assert (self.input_num == len(inputs))
                for name, input in zip(self.module.input_names, inputs):
                    x = np.load(input)
                    self.input_tensors[data_idx][name] = x

        self.args.input_num = min(self.args.input_num, len(self.input_tensors))
        print(f"input_num = {self.args.input_num}, ref = {len(self.input_tensors)}")
        print(f"real input_num = {self.args.input_num}")
        assert self.args.input_num > 0

    def run(self):
        raise NotImplementedError

    def insert_origin_op(self, op_type, args, kwargs):
        if op_type == 'top.Add':
            new_out = top.AddOp(
                self.unranked_type,
                args,
                do_relu=kwargs['do_relu'].value,
                relu_limit=kwargs['relu_limit'].value,
                ip=kwargs['ip'],
                loc=kwargs['loc'],
            ).output
        elif op_type == 'top.Concat':
            new_out = top.ConcatOp(
                self.unranked_type,
                args,
                axis=kwargs['axis'].value,
                ip=kwargs['ip'],
                loc=kwargs['loc'],
            ).output
        elif op_type == 'top.SubConst':
            new_out = top.SubConstOp(
                self.unranked_type,
                args[0],
                const_val=kwargs['const_val'],
                is_reverse=kwargs['is_reverse'].value,
                ip=kwargs['ip'],
                loc=kwargs['loc'],
            ).output
        elif op_type == 'top.Gather':
            new_out = top.GatherOp(
                self.unranked_type,
                *args,
                axis=kwargs['axis'].value,
                keepdims=kwargs['keepdims'].value,
                ip=kwargs['ip'],
                loc=kwargs['loc'],
            ).output
        elif op_type == 'top.Unsqueeze':
            new_out = top.UnsqueezeOp(
                self.unranked_type,
                *args,
                axes=[_.value for _ in kwargs['axes']],
                ip=kwargs['ip'],
                loc=kwargs['loc'],
            ).output
        elif op_type == 'top.MulConst':
            new_out = top.MulConstOp(
                self.unranked_type,
                *args,
                const_val=kwargs['const_val'].value,
                ip=kwargs['ip'],
                loc=kwargs['loc'],
            ).output
        elif op_type == 'top.LayerNorm':
            new_out = top.LayerNormOp(
                self.unranked_type,
                *args,
                normalized_shape=[],
                axis=kwargs['axis'].value,
                eps=kwargs['eps'].value,
                ip=kwargs['ip'],
                loc=kwargs['loc'],
            ).output
        elif op_type == 'top.MatMul':
            new_out = top.MatMulOp(
                self.unranked_type,
                *args,
                do_relu=kwargs['do_relu'].value,
                relu_limit=kwargs['relu_limit'].value,
                ip=kwargs['ip'],
                loc=kwargs['loc'],
            ).output
        elif op_type == 'top.Reshape':
            new_out = top.ReshapeOp(
                self.unranked_type,
                *args,
                shape=[_.value for _ in kwargs['shape']],
                ip=kwargs['ip'],
                loc=kwargs['loc'],
            ).output
        elif op_type == 'top.Permute':
            new_out = top.PermuteOp(
                self.unranked_type,
                *args,
                order=[_.value for _ in kwargs['order']],
                ip=kwargs['ip'],
                loc=kwargs['loc'],
            ).output
        elif op_type == 'top.Softmax':
            new_out = top.SoftmaxOp(
                self.unranked_type,
                *args,
                axis=kwargs['axis'].value,
                log=kwargs['log'].value,
                ip=kwargs['ip'],
                loc=kwargs['loc'],
            ).output
        elif op_type == 'top.GELU':
            new_out = top.GELUOp(
                self.unranked_type,
                *args,
                ip=kwargs['ip'],
                loc=kwargs['loc'],
            ).output
        elif op_type == 'top.Slice':
            new_out = top.SliceOp(
                self.unranked_type,
                *args,
                offset=[_.value for _ in kwargs['offset']],
                steps=[_.value for _ in kwargs['steps']],
                ends=[_.value for _ in kwargs['ends']],
                axes=[_.value for _ in kwargs['axes']],
                ip=kwargs['ip'],
                loc=kwargs['loc'],
            ).output
        elif op_type == 'top.Squeeze':
            new_out = top.SqueezeOp(
                self.unranked_type,
                *args,
                axes=[_.value for _ in kwargs['axes']],
                ip=kwargs['ip'],
                loc=kwargs['loc'],
            ).output
        elif op_type == 'top.Conv':
            new_out = top.ConvOp(
                self.unranked_type,
                *args,
                kernel_shape=[_.value for _ in kwargs['kernel_shape']],
                strides=[_.value for _ in kwargs['strides']],
                dilations=[_.value for _ in kwargs['dilations']],
                pads=[_.value for _ in kwargs['pads']],
                group=kwargs['group'].value,
                weight_is_coeff=kwargs['weight_is_coeff'].value,
                do_relu=kwargs['do_relu'].value,
                relu_limit=kwargs['relu_limit'].value,
                ip=kwargs['ip'],
                loc=kwargs['loc'],
            ).output
        elif op_type == 'top.Sigmoid':
            new_out = top.SigmoidOp(
                self.unranked_type,
                args[0],
                scale=kwargs['scale'].value,
                bias=kwargs['bias'].value,
                ip=kwargs['ip'],
                loc=kwargs['loc'],
            ).output
        elif op_type == 'top.Mul':
            new_out = top.MulOp(
                self.unranked_type,
                args,
                ip=kwargs['ip'],
                loc=kwargs['loc'],
            ).output
        elif op_type == 'top.Deconv':
            new_out = top.DeconvOp(
                self.unranked_type,
                *args,
                kernel_shape=[_.value for _ in kwargs['kernel_shape']],
                strides=[_.value for _ in kwargs['strides']],
                dilations=[_.value for _ in kwargs['dilations']],
                pads=[_.value for _ in kwargs['pads']],
                output_padding=[_.value for _ in kwargs['output_padding']],
                group=kwargs['group'].value,
                do_relu=kwargs['do_relu'].value,
                relu_limit=kwargs['relu_limit'].value,
                ip=kwargs['ip'],
                loc=kwargs['loc'],
            ).output
        elif op_type == 'top.Scale':
            new_out = top.ScaleOp(
                self.unranked_type,
                *args,
                ip=kwargs['ip'],
                loc=kwargs['loc'],
            ).output
        elif op_type == 'top.LeakyRelu':
            new_out = top.LeakyReluOp(
                self.unranked_type,
                args[0],
                alpha=kwargs['alpha'].value,
                ip=kwargs['ip'],
                loc=kwargs['loc'],
            ).output
        elif op_type == 'top.Pad':
            new_out = top.PadOp(
                self.unranked_type,
                args[0],
                paddings=[_.value for _ in kwargs['paddings']],
                val=kwargs['val'].value,
                mode=kwargs['mode'],
                ip=kwargs['ip'],
                loc=kwargs['loc'],
            ).output
        elif op_type == 'top.InstanceNorm':
            new_out = top.InstanceNormOp(
                self.unranked_type,
                *args,
                eps=kwargs['eps'].value,
                ip=kwargs['ip'],
                loc=kwargs['loc'],
            ).output
        elif op_type == 'top.Reduce':
            new_out = top.ReduceOp(
                self.unranked_type,
                *args,
                axes=[_.value for _ in kwargs['axes']],
                keepdims=kwargs['keepdims'].value,
                mode=kwargs['mode'],
                ip=kwargs['ip'],
                loc=kwargs['loc'],
            ).output
        elif op_type == 'top.Clip':
            new_out = top.ClipOp(
                self.unranked_type,
                args[0],
                min=kwargs['min'].value,
                max=kwargs['max'].value,
                ip=kwargs['ip'],
                loc=kwargs['loc'],
            ).output
        elif op_type == 'top.Tile':
            new_out = top.TileOp(
                self.unranked_type,
                args[0],
                tile=[_.value for _ in kwargs['tile']],
                ip=kwargs['ip'],
                loc=kwargs['loc'],
            ).output
        elif op_type == 'top.Div':
            new_out = top.DivOp(
                self.unranked_type,
                args,
                ip=kwargs['ip'],
                loc=kwargs['loc'],
            ).output
        elif op_type == 'top.Sub':
            new_out = top.SubOp(
                self.unranked_type,
                args,
                ip=kwargs['ip'],
                loc=kwargs['loc'],
            ).output
        elif op_type == 'top.AddConst':
            new_out = top.AddConstOp(
                self.unranked_type,
                args[0],
                const_val=kwargs['const_val'].value,
                ip=kwargs['ip'],
                loc=kwargs['loc'],
            ).output
        elif op_type == 'top.SiLU':
            new_out = top.SiLUOp(
                self.unranked_type,
                args[0],
                ip=kwargs['ip'],
                loc=kwargs['loc'],
            ).output
        elif op_type == 'top.MaxPool':
            new_out = top.MaxPoolOp(
                self.unranked_type,
                args[0],
                kernel_shape=[_.value for _ in kwargs['kernel_shape']],
                strides=[_.value for _ in kwargs['strides']],
                pads=[_.value for _ in kwargs['pads']],
                count_include_pad=kwargs['count_include_pad'].value,
                do_relu=kwargs['do_relu'].value,
                relu_limit=kwargs['relu_limit'].value,
                ip=kwargs['ip'],
                loc=kwargs['loc'],
            ).output
        elif op_type == 'top.Upsample':
            new_out = top.UpsampleOp(
                self.unranked_type,
                args[0],
                scale_h=kwargs['scale_h'].value,
                scale_w=kwargs['scale_w'].value,
                ip=kwargs['ip'],
                loc=kwargs['loc'],
            ).output
        elif op_type == 'top.Rope':
            new_out = top.RopeOp(
                self.unranked_type,
                *args,
                ip=kwargs['ip'],
                loc=kwargs['loc'],
            ).output
        else:
            # unknown op type, stop inserting mul and return
            print(f"not support inserting mul in models with {op_type}")
            return None  # insert failed
        return new_out  # insert success
