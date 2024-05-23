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

pymlir.set_mem_mode("value_mem")
from collections import Counter
from pprint import pformat
import numpy as np
from utils.mlir_parser import MlirParser

import os
from numpy_helper.tensor_compare import TensorCompare
from typing import List, Union, Dict
from utils.log_setting import setup_logger

logger = setup_logger("debugger")


# print = logger.info
def print(*args):
    info = ", ".join([str(i) for i in args])
    logger.info(info)


debug = logger.debug


def before_compare(actual, desire):
    actual = actual.astype(np.float32)
    desire = desire.astype(np.float32)
    if actual.size != desire.size:
        if actual.size > desire.size:
            actual = actual.flatten()[: desire.size].reshape(desire.shape)
        else:
            new_actual = np.zeros_like(desire)
            new_actual[: actual.size] = actual.flatten()
            actual = new_actual
    return actual, desire


class MlirDebugger:
    """
    invoke from middle layer:

    mlir = MlirDebugger('./basicvsr-spynet-bdx4.mlir')
    torch_ref = np.load('./basicvsr-spynet-bdx4_ref_outputs.npz')
    ipt = np.load("./basicvsr-spynet-bdx4_in_f32.npz")
    middle_results, failed, counter = mlir.invoke_from(['ref.1','supp.1'],'358',ipt,torch_ref)
    """

    def __init__(self, mlir_file: str, tc: TensorCompare = None):
        self.mlir_file = mlir_file
        self.parser = MlirParser(mlir_file)

        self.oldcwd = os.getcwd()
        os.chdir(os.path.dirname(os.path.realpath(mlir_file)))

        self.weight_file = self.parser.module_weight_file
        if os.path.exists(self.weight_file):
            self.weights = np.load(self.weight_file)
        else:
            raise KeyError(f"{self.weight_file} not exists in cwd: {os.getcwd()}")

        if tc is None:
            tc = TensorCompare(
                close_order_tol=3,
                cosine_similarity_tol=0.99,
                euclidean_similarity_tol=0.9,
                signal_to_quantization_noise_tol=float("-inf"),
                per_axis_compare=-1,
            )
        self.tc = tc

        self.module = pymlir.module()
        self.module.load(os.path.basename(mlir_file))

    def invoke_at(self, name: str, input_data_dict: Dict[str, np.ndarray]) -> dict:
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
            if op.type == "tpu.GenericCpu":
                print(f"[run]  {op.type}/{op.attrs['cpu_op_name']}{op.opds} -> {name}")
            else:
                print(f"[run]  {op.type}{op.opds} -> {name}")
            # must define variable even it's not used, nor the memory of this value would be cleared
            _ = self.module.invoke_at(name)

            outkeys = set(op.outputs)
            dic = self.module.get_all_tensor()
            for k in outkeys:
                if k not in dic:
                    print(f"Miss {k}")
                    continue
                v = dic[k]
                print(k, outkeys)
                data = v.copy()  # type: np.ndarray
                # try de-quant

                zero_point = float(op.attrs.get("quant_zero_point", 0))
                scale = float(op.attrs.get("quant_scale", 1))
                if "quant_scale" in op.attrs:
                    dtype_str = "int8"
                else:
                    dtype_str = "f32"

                outputs[f"{k}_context"] = {
                    "ori_dtype": dtype_str,
                    "zero_point": zero_point,
                    "quant": scale,
                }
                outputs[k + f"_{dtype_str}"] = data
                data = (data.astype(np.float32) - zero_point) * scale

                outputs[k] = data
        return outputs

    def invoke_from(
        self,
        op_names: List[str],
        output_names: Union[List[str], str],
        feed_dict: Dict[str, np.ndarray] = {},
        reference: Dict[str, np.ndarray] = {},
        force_correct: bool = False,
        fast_fail=False,
        excepts=[],
    ):
        """
        results -> type -> name -> { op, result, inputs, outputs, reference_outputs}

        Args:
            op_names: ["/bert/encoder/layer.0/attention/self/Add_output_0_Add_f32", ]
            output_names: list or string that with output name(s)
                ["/bert/encoder/layer.0/attention/self/Add_output_0_Add_f32", ]

        middle_results, failed, counter = mlir.invoke_from(['ref.1','supp.1'],'358',ipt,torch_ref)
        """
        res = {}
        if isinstance(output_names, str):
            output_names = [output_names]

        output_keyset = set(output_names)
        middle_results = {}

        inputs = []
        for k in op_names:
            if self.parser.get_op_by_op_name(k).type == "top.Input":
                inputs.append(k)
            else:
                inputs.extend(self.parser.get_pre_op_by_op_name(k))
        outputs = []
        # ensure root input correctness
        memo = {k: reference[k] if k in reference else feed_dict[k] for k in inputs}
        failed = {}
        failed_type_counter = Counter()
        succeed_type_counter = Counter()

        while len(output_keyset) > 0:
            if len(inputs) == 0:
                print("Early stop")
                break
            if len(failed) > 0 and fast_fail:
                print("fast fail, stop inference")
                break
            # print("inputs", inputs)
            for ipt in inputs:  # try find output meet input requirements
                opt_names = self.parser.get_next_op_by_op_name(ipt)
                # get_next_op may skip some operation, like squeeze -> topk[0, 1],
                # the second output may be skiped.
                # print("opt_names", opt_names)
                for opt in opt_names:
                    op = self.parser.get_op_by_op_name(opt)
                    missing = [
                        pre
                        for pre in self.parser.get_pre_op_by_op_name(opt)
                        if pre not in memo
                    ]

                    if any(missing):
                        print(f"Temporarily skip inference {opt} for lack of {missing}")
                        continue

                    for pre in self.parser.get_pre_op_by_op_name(opt):
                        if pre in memo:
                            continue
                        if pre not in feed_dict:
                            raise KeyError(f"Missing {pre} in feed_dict or op_names")
                        memo[pre] = feed_dict[pre]
                        print(
                            f"[warn] taken {pre} from feed_dict for {opt}({op.type}) operation"
                        )

                    if force_correct:
                        for pre in self.parser.get_pre_op_by_op_name(opt):
                            if pre in reference:
                                print(f"use reference value of {pre}")
                                if memo[pre].shape != reference[pre].shape:
                                    print(
                                        f"inplace because shape not equal memo {memo[pre].shape} but ref {reference[pre].shape}"
                                    )
                                    memo[pre].flatten()[: reference[pre].size] = (
                                        reference[pre].flatten()
                                    )
                                else:
                                    memo[pre] = reference[pre]

                    res = self.invoke_at(opt, memo)
                    middle_results.update(res)
                    outputs.extend(res.keys())
                    output_keyset = (
                        output_keyset - res.keys()
                    )  # update output_keyset, when len(output_keyset) == 0, all required output_name are calculated.

                    op_input = {k: memo[k] for k in op.opds if k in memo}
                    op_input_weight = {
                        k: self.weights[k] for k in op.opds if k in self.weights
                    }
                    op_input_shape = {k: memo[k].shape for k in op.opds if k in memo}
                    op_input_weight_shape = {
                        k: self.weights[k].shape for k in op.opds if k in self.weights
                    }

                    if opt in reference and opt not in excepts:
                        actual, desire = before_compare(res[opt], reference[opt])

                        cmp_ret = self.tc.compare(
                            actual,
                            desire,
                            verbose=2,
                        )
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
                            if fast_fail:
                                break

            inputs = list(set(outputs))
            inputs = [i for i in inputs if i not in memo]
            memo.update(middle_results)

        counter = {"succeed": succeed_type_counter, "failed": failed_type_counter}
        return middle_results, failed, counter

    def invoke_all(
        self,
        feed_dict: Dict[str, np.ndarray] = {},
        reference: Dict[str, np.ndarray] = {},
        force_correct: bool = False,
        fast_fail=False,
        excepts=[],
    ):
        op_names = [i.name for i in self.parser.inputs]
        output_names = list(self.parser.get_output_op_names_n_shapes().keys())
        print(op_names)
        print(output_names)
        middle_results, failed, counter = self.invoke_from(
            op_names,
            output_names,
            feed_dict,
            reference,
            force_correct,
            fast_fail=fast_fail,
            excepts=excepts,
        )
        return middle_results, failed, counter
