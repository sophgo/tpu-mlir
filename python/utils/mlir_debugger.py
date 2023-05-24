#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
import pymlir
from collections import Counter
from pprint import pformat
import numpy as np
from utils.mlir_parser import MlirParser
import os
from numpy_helper.tensor_compare import TensorCompare


class MlirDebugger:
    """
    invoke from middle layer:
    
    mlir = MlirDebugger('./basicvsr-spynet-bdx4.mlir')
    torch_ref = np.load('./basicvsr-spynet-bdx4_ref_outputs.npz')
    ipt = np.load("./basicvsr-spynet-bdx4_in_f32.npz")
    middle_results, failed, counter = mlir.invoke_from(['ref.1','supp.1'],'358',ipt,torch_ref)
    """
    def __init__(self, mlir_file):
        self.mlir_file = mlir_file
        self.parser = MlirParser(mlir_file)

        self.oldcwd = os.getcwd()
        os.chdir(os.path.dirname(mlir_file))

        self.weight_file = self.parser.module_weight_file
        if os.path.exists(self.weight_file):
            self.weights = np.load(self.weight_file)
        else:
            raise KeyError(f"{self.weight_file} not exists in cwd: {os.getcwd()}")

        self.tc = TensorCompare(
            close_order_tol=3,
            cosine_similarity_tol=0.99,
            euclidean_similarity_tol=0.9,
            signal_to_quantization_noise_tol=float("-inf"),
            per_axis_compare=-1,
        )

        self.module = pymlir.module()
        self.module.load(os.path.basename(mlir_file))

    def invoke_at(self, name, input_data_dict: dict):
        op = self.parser.get_op_by_op_name(name)
        missing = [
            i for i in op.opds if i not in input_data_dict and i not in self.weights
        ]
        if len(missing) > 0:
            raise ValueError(
                f"Lacking of value of key {missing} of op {pformat(op.__dict__)}"
            )

        for k in op.opds:
            if k not in input_data_dict:
                continue

            input = input_data_dict[k]
            if input.dtype == np.int8 or input.dtype == np.uint8:
                self.module.set_tensor_from_int(k, input_data_dict[k])
            else:
                self.module.set_tensor(k, input.astype(np.float32))

        outputs = {}
        if op.type == "top.Input":
            outputs[op.name] = input_data_dict[op.name]
        else:
            if op.type == 'tpu.GenericCpu':
                print(f"[run]  {op.type}/{op.attrs['cpu_op_name']}{op.opds} -> {name}")
            else:
                print(f"[run]  {op.type}{op.opds} -> {name}")
            # must define variable even it's not used.
            output = self.module.invoke_at(name)

            outkeys = set(op.outputs)
            for k, v in self.module.get_all_tensor().items():
                if k in outkeys:
                    data = v.copy()
                    # try de-quant
                    outputs[f"{k}_ori"] = {'data':data,'zero_point':float(op.attrs.get('quant_zero_point',0)),'quant':float(op.attrs.get('quant_scale',1)) }
                    data = (data - float(op.attrs.get('quant_zero_point',0))) *float(op.attrs.get('quant_scale',1))
                    outputs[k] = data
        return outputs

    def make_flow(self, name, to):
        """ignore output of unrelavent ops"""
        res = set()
        res.add(to)
        outputs = [to]
        while name not in res:
            inputs = []
            if len(outputs) == 0:
                raise KeyError(f"Can not find name {name} from previous op of {to}")
            for op in outputs:
                pre = self.parser.get_pre_op_by_op_name(op)
                inputs.extend(pre)
                res.update(pre)
            outputs = inputs
        return res

    def value_between(self, name, feed_inputs):
        op = self.parser.get_op_by_op_name(name)
        inputs = {
            i: feed_inputs[i] if i in feed_inputs else self.weights.get(i)
            for i in op.opds
        }
        outputs = {
            i: feed_inputs[i] if i in feed_inputs else self.weights.get(i)
            for i in op.outputs
        }

        return inputs, outputs

    def invoke_from(self, op_names, output_name, feed_dict, reference={}, force_correct=False):
        """
        results -> type -> name -> { op, result, inputs, outputs, reference_outputs}

        Args:
            name (op name): /bert/encoder/layer.0/attention/self/Add_output_0_Add_f32
            kwargs: input of name, {"/bert/encoder/layer.0/attention/self/Add_output_0_Add_f32": ...}
        """
        res = {}

        middle_results = {}
        feed_dict = dict(feed_dict)
        pre_inputs = {i:self.parser.get_pre_op_by_op_name(i) for i in op_names}
        print(
            f"inputs(pre op) of {op_names} : {pre_inputs}"
        )
        print(
            f"output(next op) of {output_name} : {self.parser.get_next_op_by_op_name(output_name)}"
        )


        inputs = []
        for k in op_names:
            if self.parser.get_op_by_op_name(k).type == 'top.Input':
                inputs.append(k)
            else:
                inputs.extend(self.parser.get_pre_op_by_op_name(k))
        outputs = []
        # ensure root input correctness
        memo = {k:reference[k] if k in reference else feed_dict[k] for k in inputs}


        failed = {}
        failed_type_counter = Counter()
        succeed_type_counter = Counter()


        while output_name not in middle_results:
            if len(inputs) == 0:
                print("Early stop")
                break
            for ipt in inputs:  # try find output meet input requirements
                opt_names = self.parser.get_next_op_by_op_name(ipt)
                for opt in opt_names:
                    op = self.parser.get_op_by_op_name(opt)
                    missing = [pre for pre in self.parser.get_pre_op_by_op_name(opt) if pre not in memo]
                    if any(missing):
                        print(f"Skip {opt} for lack of {missing}")
                        continue

                    for pre in self.parser.get_pre_op_by_op_name(opt):
                        if pre in memo:
                            continue
                        if pre not in feed_dict:
                            raise KeyError(f"Missing {pre} in feed_dict or op_names")
                        memo[pre] = feed_dict[pre]
                        print(f"[warn] taken {pre} from feed_dict for {opt}({op.type}) operation")

                    if force_correct:
                        for pre in self.parser.get_pre_op_by_op_name(opt):
                            if pre in reference:
                                print(f'use reference value of {pre}')
                                memo[pre] = reference[pre]

                    res = self.invoke_at(opt, memo)
                    middle_results.update(res)
                    outputs.append(opt)

                    op_input = {k: feed_dict[k] for k in op.opds if k in feed_dict}
                    op_input_weight = {
                        k: self.weights[k] for k in op.opds if k in self.weights
                    }
                    op_input_shape = {k: feed_dict[k].shape for k in op.opds if k in feed_dict}
                    op_input_weight_shape = {
                        k: self.weights[k].shape for k in op.opds if k in self.weights
                    }

                    if opt in reference:
                        cmp_ret = self.tc.compare(reference[opt], res[opt], verbose=2)
                        ret, msg, code, meta, _ = cmp_ret

                        print(cmp_ret)
                        if ret:
                            succeed_type_counter[op.type] += 1
                        else:
                            failed_type_counter[op.type] += 1
                            failed.setdefault(op.type, []).append(
                                {
                                    "op": op,
                                    "name": op.name,
                                    "compare_score": meta,
                                    "input": op_input,
                                    "input_shape": op_input_shape,
                                    "input_weight": op_input_weight,
                                    "input_weight_shape": op_input_weight_shape,
                                    "output": res[opt],
                                    "output_shape": res[opt].shape,
                                    "reference_output": reference[opt],
                                }
                            )

                            print(failed_type_counter)


            inputs = list(set(outputs))
            inputs = [i for i in inputs if i not in memo]
            memo.update(middle_results)

        counter = {"succeed": succeed_type_counter, "failed": failed_type_counter}
        return middle_results, failed, counter


