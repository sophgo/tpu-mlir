#!/usr/bin/env python3
# ==============================================================================
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
import argparse
import json
from typing import Dict, List, Tuple, Union, Any
from utils.mlir_parser import MlirParser
import os
import ast
from types import MappingProxyType

# Define parameter types
ParamType = Union[int, float, bool, List[int], List[float], List[List[int]], str]

# Operator sets with their parameters and types
train_op_sets = {
    "BatchNormTrain": {
        "do_relu": bool,
        "epsilon": float,
        "momentum": float
    },
    "BatchNormBwd": {
        "epsilon": float,
    },
    "Conv": {
        "strides": List[int],
        "pads": List[int],
        "dilations": List[int],
        "group": int,
    },
    "Convbwd": {
        "input_shapes": List[int],
        "stride": List[int],
        "padding": List[int],
        "dilations": List[int],
        "groups": int,
        "grad_input_enable": bool,
        "grad_weight_enable": bool,
        "grad_bias_enable": bool,
        "grad_out_shapes": List[int]
    },
    "MaxPoolWithMask": {
        "ceil_mode": bool,
        "do_relu": bool,
        "kernel_shape": List[int],
        "pads": List[int],
        "strides": List[int],
        "dilations": List[int]  # not support in tpu-mlir for now
    },
    "MaxPoolingIndicesBwd": {
        "dilations": List[int],
        "input_shape": List[int],
        "kernel_shape": List[int],
        "pads": List[int],
        "strides": List[int]
    },
}

# Inference operator sets
infer_op_sets = {
    "Abs": {},
    "Add": {},
    "Relu": {},
    "Cast": {
        "to": str,
    },
    "Concat": {
        "axis": int,
    },
    "Compare": {
        "mode": str,
    },
    "Conv": {
        "strides": List[int],
        "pads": List[int],
        "dilations": List[int],
        "group": int,
    },
    "Conv1d": {
        "strides": List[int],
        "pads": List[int],
        "dilations": List[int],
        "group": int,
    },
    "Conv2d": {
        "strides": List[int],
        "pads": List[int],
        "dilations": List[int],
        "group": int,
    },
    "Conv3d": {
        "strides": List[int],
        "pads": List[int],
        "dilations": List[int],
        "group": int,
    },
    "BatchNorm": {
        "epsilon": float,
        "momentum": float,
        "do_relu": bool,
    },
    "MaxPool": {
        "kernel_shape": List[int],
        "strides": List[int],
        "pads": List[int],
        "ceil_mode": bool,
        "dilations": List[int],
    },
    "AvgPool": {
        "kernel_shape": List[int],
        "strides": List[int],
        "pads": List[int],
        "ceil_mode": bool,
        "dilations": List[int],
    },
}

all_op_sets = MappingProxyType({**train_op_sets, **infer_op_sets})


def save_operator_params(output_path: str, op_type: str, params: List, kind: str = None) -> None:
    """Save operator parameters with operator type validation key"""
    data = {"operator": op_type, "cases": params}
    if kind:
        data["type"] = kind
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    return data["cases"]


def convert_value(value: str, target_type: type) -> Any:
    if ":" in value:
        value = value.split(":")[0].strip()

    if target_type == bool:
        if value.lower() in ("true", "1"):
            return True
        elif value.lower() in ("false", "0"):
            return False
        else:
            raise ValueError(f"Cannot convert '{value}' to bool")
    elif target_type == str:
        return value
    elif target_type in (int, float):
        try:
            return target_type(value)
        except ValueError:
            try:
                return target_type(ast.literal_eval(value))
            except (ValueError, SyntaxError):
                if target_type == int and "." in value:
                    return int(float(value))
                raise ValueError(f"Cannot convert '{value}' to {target_type.__name__}")
    elif hasattr(target_type, "__origin__") and target_type.__origin__ == list:
        element_type = target_type.__args__[0]
        if value.startswith("[") and value.endswith("]"):
            items = [x.strip() for x in value[1:-1].split(",")]
            return [
                convert_value(item.split(":")[0].strip() if ":" in item else item, element_type)
                for item in items
            ]
        else:
            return [
                convert_value(value.split(":")[0].strip() if ":" in value else value, element_type)
            ]
    else:
        raise ValueError(f"Unsupported target type: {target_type}")


def extract_op_params(op, target_params: Dict[str, type], idx: int = 0) -> Dict[str, ParamType]:
    params = {"id": idx, "input_shapes": None, "model_params": None}
    params["input_shapes"] = convert_value(op.in_shapes, str)
    model_params = {}
    for param_name, param_type in target_params.items():
        if param_name in op.attrs:
            raw_value = op.attrs[param_name]
            try:
                model_params[param_name] = convert_value(raw_value, param_type)
            except (ValueError, SyntaxError) as e:
                print(
                    f"Warning: Failed to convert parameter '{param_name}' with value '{raw_value}' "
                    f"to type {param_type.__name__}: {e}")
                model_params[param_name] = raw_value
    params["model_params"] = model_params
    return params


def parse_mlir_and_extract_ops(mlir_file: str, target_op: str, kind: str = None) -> Dict[str, Any]:
    """Parse MLIR file and extract parameters for target operator"""
    parser = MlirParser(mlir_file)

    assert parser.module_state.split("_")[0] == "TOP", "MLIR file must be a TOP dialect file"

    if target_op not in all_op_sets:
        raise ValueError(
            f"Unsupported operator type: {target_op}. Available operators: {sorted(list(all_op_sets.keys()))}"
        )

    resolved_kind = kind
    if resolved_kind is None:
        in_train = target_op in train_op_sets
        in_infer = target_op in infer_op_sets
        if in_train and in_infer:
            raise ValueError(f"Operator '{target_op}' exists in both train and inference sets. "
                             f"Please specify --kind train|inference explicitly to disambiguate.")
        elif in_train:
            resolved_kind = "train"
        elif in_infer:
            resolved_kind = "inference"

    if resolved_kind not in ["train", "inference"]:
        raise ValueError(f"Invalid kind: {resolved_kind}. Use 'train' or 'inference'.")

    target_params = train_op_sets[target_op] if resolved_kind == "train" else infer_op_sets[
        target_op]
    cases = []
    idx = 0
    for op in parser.ops:
        op_base_type = op.type.split(".")[-1]
        if op_base_type == target_op:
            params = extract_op_params(op, target_params, idx)
            cases.append(params)
            idx += 1

    return {"operator": target_op, "type": resolved_kind, "cases": cases}


def main():
    parser = argparse.ArgumentParser(
        description="Extract operator parameters from MLIR file and save to JSON.")
    parser.add_argument("mlir_file", type=str, help="Path to the input MLIR file")
    parser.add_argument("output_json", type=str, help="Path to the output JSON file")
    parser.add_argument(
        "--op",
        type=str,
        required=True,
        choices=list(all_op_sets.keys()),
        help=f"Operator to be extracted. Available options: {list(all_op_sets.keys())}")
    parser.add_argument("--kind",
                        type=str,
                        default=None,
                        choices=["train", "inference"],
                        help="Operator kind. If omitted, inferred from --op.")
    parser.add_argument("--validate",
                        action="store_true",
                        help="Validate the output file contains the correct operator type")

    args = parser.parse_args()

    if not os.path.isfile(args.mlir_file):
        raise FileNotFoundError(f"MLIR file {args.mlir_file} does not exist")

    try:
        # Extract and save parameters
        op_data = parse_mlir_and_extract_ops(args.mlir_file, args.op, args.kind)
        save_operator_params(args.output_json, args.op, op_data["cases"], op_data.get("type"))

        print(f"Successfully extracted parameters for {len(op_data['cases'])} {args.op} operators")
        print(f"Results saved to {args.output_json}")

    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
