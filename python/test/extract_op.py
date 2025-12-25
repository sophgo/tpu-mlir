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
onnx_op_sets = {
    "Abs": {},
    "Add": {},
    "Sub": {},
    "Mul": {},
    "Div": {},
    "AddConst": {
        "const_val": float
    },
    "And": {},
    "Relu": {
        "relu_limit": float
    },
    "Concat": {
        "axis": int
    },
    "Compare": {
        "mode": str
    },
    "Softmax": {
        "axis": int,
        "beta": float,
        "log": bool
    },
}

torch_op_sets = {}

framework_op_sets = {
    "onnx": onnx_op_sets,
    "torch": torch_op_sets,
}


def save_operator_params(output_path: str, op_type: str, params: List) -> None:
    """Save operator parameters with operator type validation key"""
    data = {"operator": op_type, "cases": params}
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
        if isinstance(value, str):
            value = value.strip()
            if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
                return value[1:-1]
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


def parse_mlir_and_extract_ops(mlir_file: str, target_op: str,
                               op_sets: Dict[str, Any]) -> Dict[str, Any]:
    """Parse MLIR file and extract parameters for target operator"""
    parser = MlirParser(mlir_file)

    assert parser.module_state.split("_")[0] == "TOP", "MLIR file must be a TOP dialect file"

    if target_op not in op_sets:
        raise ValueError(
            f"Unsupported operator type: {target_op}. Available operators: {list(op_sets.keys())}")

    target_params = op_sets[target_op]
    cases = []
    idx = 0
    for op in parser.ops:
        op_base_type = op.type.split(".")[-1]
        if op_base_type == target_op:
            params = extract_op_params(op, target_params, idx)
            cases.append(params)
            idx += 1

    return {"operator": target_op, "cases": cases}


def main():
    parser = argparse.ArgumentParser(
        description="Extract operator parameters from MLIR file and save to JSON.")
    parser.add_argument("mlir_file", type=str, help="Path to the input MLIR file")
    parser.add_argument("output_json", type=str, help="Path to the output JSON file")
    parser.add_argument(
        "--framework",
        type=str,
        required=True,
        choices=list(framework_op_sets.keys()),
        help=f"Framework to be processed. Available options: {list(framework_op_sets.keys())}")
    temp_args, _ = parser.parse_known_args()
    framework = getattr(temp_args, "framework", None)
    op_choices = list(framework_op_sets[framework].keys()) if framework else []
    parser.add_argument("--op",
                        type=str,
                        required=True,
                        choices=op_choices,
                        help=f"Operator to be extracted. Available options: {op_choices}")

    args = parser.parse_args()

    if not os.path.isfile(args.mlir_file):
        raise FileNotFoundError(f"MLIR file {args.mlir_file} does not exist")
    try:
        # Extract and save parameters
        op_data = parse_mlir_and_extract_ops(args.mlir_file, args.op,
                                             framework_op_sets[args.framework])
        save_operator_params(args.output_json, args.op, op_data["cases"])

        print(f"Successfully extracted parameters for {len(op_data['cases'])} {args.op} operators")
        print(f"Results saved to {args.output_json}")

    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
