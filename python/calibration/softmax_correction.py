import os
import numpy as np
import mlir.dialects.top as top
from tqdm import tqdm
from calibration.model_modifier import ModelModifier
from utils.mlir_parser import Operation
from utils.mlir_shell import _os_system
from transform.MLIRImporter import MLIRImporter
from mlir.ir import Location


def parse_sm_head_axis(activation):
    shape = activation.shape
    if len(shape) == 3:
        return 0
    elif len(shape) == 4:
        return 1
    else:
        raise ValueError("unsupported softmax output shape: {}".format(shape))


class SoftmaxCorrecter(ModelModifier):

    def get_sm_op_names(self):
        self.sm_op_names = []
        self.all_op_names = self.parser.get_op_name_list()
        for op in self.all_op_names:
            op_type = self.parser.get_op_type_by_op_name(op)
            if op_type == 'top.Softmax':
                next_ops = self.parser.get_next_op_by_op_name(op)
                if len(next_ops) != 1: continue
                next_op = next_ops[0]
                next_op_type = self.parser.get_op_type_by_op_name(next_op)
                if next_op_type != 'top.MatMul': continue
                matmul_pre_ops = self.parser.get_pre_op_by_op_name(next_op)
                if len(matmul_pre_ops) == 2 and matmul_pre_ops[0] == op:
                    self.sm_op_names.append(op)
        for i, op_name in enumerate(self.sm_op_names):
            print(f"correct softmax op {i} {op_name}")

    def calibrate_softmax(self):
        self.softmax_calib_scales = {op_name: 0 for op_name in self.sm_op_names}
        N = self.args.input_num
        pbar = tqdm([i for i in range(N)], total=N, position=0, leave=True)
        for idx in range(N):
            pbar.set_description(f"Softmax Correction for sample {idx}")
            for name in self.module.input_names:
                input_tensor = self.input_tensors[idx][name]
                self.module.set_tensor(name, input_tensor, input_tensor.shape)
            self.module.invoke()
            for op_name in self.sm_op_names:
                output = self.parser.get_outputs_by_op_name(op_name)[0]
                activation = self.module.get_tensor(output).copy()
                self.softmax_calib_scales[op_name] += np.max(activation, axis=-1, keepdims=True) / N
            pbar.update(1)
        pbar.close()
        for op_name in self.sm_op_names:
            activation = self.softmax_calib_scales[op_name]
            head_axis = parse_sm_head_axis(activation)
            reduce_axes = tuple(i for i in range(len(activation.shape)) if i != head_axis)
            self.softmax_calib_scales[op_name] = np.mean(activation,
                                                         axis=reduce_axes,
                                                         keepdims=True)

    def insert_mul_ops(self):
        sm_ops = self.sm_op_names
        aftmm2sm = {self.parser.get_next_op_by_op_name(sm)[0]: sm for sm in sm_ops}
        aftop2aftmm = {}
        for mm in aftmm2sm:
            aft_ops = self.parser.get_next_op_by_op_name(mm)
            for op in aft_ops:
                aftop2aftmm[op] = mm

        input_shapes = [list(self.input_tensors[0][n].shape) for n in self.module.input_names]
        output_shapes = [[] for n in self.module.output_names]
        input_types = []
        for n in self.module.input_names:
            if self.input_tensors[0][n].dtype == np.float32:
                input_types.append('F32')
            elif self.input_tensors[0][n].dtype == np.int32:
                input_types.append('INT32')
            else:
                raise TypeError(
                    f'input_name: {n}, sunknown input data type: {self.input_tensors[0][n]}')

        new_mlir = MLIRImporter(input_shapes,
                                output_shapes,
                                self.parser.module_name,
                                "ONNX",
                                input_types,
                                run_mode='STATIC',
                                weight_file=self.parser.module_weight_file)
        self.unranked_type = new_mlir.get_tensor_type([])

        idx2operand = {0: new_mlir.none_op}
        idx2name = {0: ''}
        # create input ops
        for i, name in enumerate(self.module.input_names):
            op = self.parser.body.operations[i + 1]
            attrs = op.attributes
            name = Operation.name(op)
            kwargs = {}
            for a in attrs:
                if hasattr(a.attr, 'value'):
                    kwargs[a.name] = a.attr.value
                else:
                    kwargs[a.name] = [_.value for _ in a.attr]
            input_ = new_mlir.create_input_op(
                Location.fused([Location.name(name)], context=new_mlir.ctx), i, kwargs)
            idx2operand[i + 1] = input_
            idx2name[i + 1] = name

        sm2mulout = {}  # softmax to mul output mapping, used to replace matmul input
        aftmm2mulout = {}  # matmul to mul output mapping, used to replace following op input
        return_idx = []
        # insert other ops
        for i in range(len(self.module.input_names) + 1, len(self.parser.body.operations) - 1):
            op = self.parser.body.operations[i]
            op_type = Operation.type(op)
            attrs = op.attributes
            name = Operation.name(op)
            kwargs = {a.name: a.attr for a in attrs}
            kwargs['ip'] = new_mlir.insert_point
            kwargs['loc'] = Location.fused([Location.name(name)], context=new_mlir.ctx)
            pre_op_ids = [int(_.get_name().strip('%')) for _ in op.operands]
            args = [idx2operand[j] for j in pre_op_ids]
            if name in aftmm2sm:  # replace matmul input with inserted mul output
                args[0] = sm2mulout[aftmm2sm[name]]
            if name in aftop2aftmm:
                args[0] = aftmm2mulout[aftop2aftmm[name]]
            if op_type == 'top.Weight':
                weight_shape = Operation.shape(op)
                new_out = new_mlir.create_weight_op(name, weight_shape)
                self.module_weights[name] = self.module_weights[name].reshape(weight_shape)
            else:
                new_out = self.insert_origin_op(op_type, args, kwargs)
            if new_out is None: return  # fail to insert, stop inserting mul

            if name in sm_ops:  # insert mul after softmax
                mul_shape = self.softmax_calib_scales[name].shape
                mulW_name = name + "_Smc_Inv_Weight"
                self.module_weights[mulW_name] = 1.0 / self.softmax_calib_scales[name]
                mulW_out = new_mlir.create_weight_op(mulW_name, mul_shape)
                mul_name = name + '_Smc_Inv_Mul'
                mul_out = top.MulOp(self.unranked_type, [new_out, mulW_out],
                                    loc=Location.fused([Location.name(mul_name)],
                                                       context=new_mlir.ctx),
                                    ip=new_mlir.insert_point).output
                sm2mulout[name] = mul_out
            if name in aftmm2sm:  # insert mul after matmul
                sm_name = aftmm2sm[name]
                mul_shape = self.softmax_calib_scales[sm_name].shape
                mulW_name = aftmm2sm[name] + "_Smc_Weight"
                self.module_weights[mulW_name] = self.softmax_calib_scales[sm_name]
                mulW_out = new_mlir.create_weight_op(mulW_name, mul_shape)
                mul_name = name + '_Smc_Mul'
                mul_out = top.MulOp(self.unranked_type, [new_out, mulW_out],
                                    loc=Location.fused([Location.name(mul_name)],
                                                       context=new_mlir.ctx),
                                    ip=new_mlir.insert_point).output
                aftmm2mulout[name] = mul_out

            idx2operand[i] = new_out
            idx2name[i] = name
            if name in self.parser.get_output_op_names_n_shapes():
                return_idx.append(i)
        # create return op
        return_op = []
        for i in return_idx:
            return_op.append(idx2operand[i])
        new_mlir.create_return_op(return_op)
        # update mlir file
        new_mlir_txt = new_mlir.print_module()
        with open('tmp.mlir', 'w') as f:
            f.write(new_mlir_txt)
        np.savez(self.parser.module_weight_file, **self.module_weights)
        cmd = ['tpuc-opt', 'tmp.mlir', '--shape-infer', '-o', self.args.mlir_file]
        _os_system(cmd, log_level="normal")
        os.remove('tmp.mlir')

    def run(self):
        if self.data_selector.all_image:
            self.init_ppa()
        self.load_net_weights()
        self.load_net_inputs()
        self.get_sm_op_names()
        self.calibrate_softmax()
        self.insert_mul_ops()
