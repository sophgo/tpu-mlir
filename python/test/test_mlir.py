#!/usr/bin/env python3
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import json
import numpy as np
import os, sys
import argparse
from utils.timer import Timer
from typing import List, Dict, Any, Optional, Callable, Tuple, Union
from utils.regression_logger import run_in_log_wrapper
from contextlib import contextmanager
import multiprocessing
from utils.misc import collect_process
from mlir.ir import *
import mlir.dialects.top as top
from transform.MLIRImporter import MLIRImporter, Platform

# Constants
SUPPORTED_CHIPS = ["bm1684x", "bm1688", "bm1690", "bm1690e"]
SUPPORTED_MODES = ["f32", "f16", "bf16"]  # Extend as needed


def deploy_case_bmodel(case_name: str,
                       chip: str,
                       mode: str,
                       tolerance: Tuple[float, float] = (0.98, 0.95),
                       test_reference: Optional[str] = None,
                       num_core: int = 1,
                       debug: bool = False,
                       dynamic: bool = False,
                       rvti: bool = False,
                       disable_lg: bool = False,
                       no_check: bool = False) -> None:
    """
    Run `model_deploy.py` for a single case/chip/mode.

    Args:
        case_name: base name for generated mlir/npz
        chip: chip name, e.g. "bm1684x"
        mode: quant mode, e.g. "f32" / "f16"
        tolerance: (cos_tol, euclidean_tol) pair passed to --tolerance
    """
    chip = chip.lower()
    mode = mode.lower()

    bmodel_name = f"{case_name}_{chip}_{mode}.bmodel"
    cos_tol, euclidean_tol = tolerance
    test_reference_arg = (f"--test_reference {test_reference} "
                          if test_reference is not None else "")
    test_args = ""
    if not no_check:
        test_args = f"--test_input {case_name}_input.npz --tolerance {cos_tol},{euclidean_tol} {test_reference_arg}"
    deploy_cmd = [
        f"model_deploy.py --mlir {case_name}.mlir", f"--chip {chip}", f"--model {bmodel_name}",
        test_args, f"--quantize {mode.upper()}", f"--num_core {num_core}", "--high_precision"
    ]
    if debug:
        deploy_cmd.append("--debug")
    if dynamic:
        deploy_cmd.append("--dynamic")
    if rvti:
        deploy_cmd.append("--rvti")
    if disable_lg:
        deploy_cmd.append("--disable_layer_group")
    deploy_cmd.append("--disable_gdma_check")
    deploy_cmd = " ".join(deploy_cmd)
    print(deploy_cmd)
    assert os.system(deploy_cmd) == 0


def rand_data(shape: List[int],
              dtype: str,
              min_val: float = -10,
              max_val: float = 10,
              seed: Optional[int] = None,
              int_satu: bool = False) -> np.ndarray:
    """
    Generate random data with specified shape and dtype.

    Args:
        shape: Shape of the output array
        dtype: Data type ('float32', 'float16', 'int32', etc.)
        min_val: Minimum value for clipping
        max_val: Maximum value for clipping
        seed: Random seed for reproducibility
        int_satu: Whether to apply saturation for integer types

    Returns:
        np.ndarray: Generated random data array.
    """
    if seed is not None:
        np.random.seed(seed)

    if dtype in ['float32', 'float16']:
        data = np.random.randn(*shape).astype(dtype)
        return np.clip(data, min_val, max_val)

    int_dtypes = ['int32', 'uint32', 'int16', 'uint16', 'int8', 'uint8']
    if dtype in int_dtypes:
        if int_satu:
            data = np.random.randint(0, 127, size=shape).astype(dtype)
            return np.clip(data, min_val, max_val)
        else:
            return np.random.randint(0, 127, size=shape).astype(dtype)

    raise ValueError(f"Unsupported data type: {dtype}")


@contextmanager
def change_directory(path: str):
    """
    Context manager for temporarily changing the working directory.

    Args:
        path: Directory to change to
    """
    original_dir = os.getcwd()
    try:
        os.makedirs(path, exist_ok=True)
        os.chdir(path)
        yield
    finally:
        os.chdir(original_dir)


Failed_Cases = []


class MLIR_IR_TESTER(object):
    _id = 0  # Class variable for generating unique names

    # This class is built for testing single operator transform.
    def __init__(self, args):
        Y, N = True, False
        # Test function registry with chip support
        self._test_functions = {
            #############################
            # MLIR Test Case, Alphabetically
            #############################
            # case:  (test_function,      bm1684x_support, bm1688_support)
            "error0": (self.test_error0, Y, Y),
            "insert": (self.test_insert, Y, Y),
            "fattention": (self.test_fattention, Y, Y),
            "slice": (self.test_slice, Y, Y),
            "a16matmul": (self.test_a16matmul, Y, Y),
            "chunk_gated_delta_rule": (self.test_chunk_gated_delta_rule, Y, Y),
            "recurrent_gated_delta_rule": (self.test_recurrent_gated_delta_rule, Y, Y),
            "concat_slice": (self.test_concat_slice, Y, Y),
        }
        # currently test_mlir.py only supports fp quant mode
        self.support_quant_modes = ["f32", "f16"]  # no need "bf16" for now
        self.mode = args.mode.lower()
        self.simple = args.simple
        self.chip = args.chip.lower()
        self.concise_log = args.concise_log  # use when run regression/main_entry.py
        self.num_core = args.num_core
        self.debug = args.debug
        self.dynamic = args.dynamic
        self.rvti = args.rvti
        self.disable_lg = args.disable_lg
        self.no_check = args.no_check
        if self.chip not in SUPPORTED_CHIPS:
            raise ValueError(f"Unsupported chip: {self.chip}. Supported: {SUPPORTED_CHIPS}")

        # Set quantization modes
        self.support_quant_modes = SUPPORTED_MODES
        if self.simple:
            self.support_quant_modes = ["f16"]

        if self.mode == "" or self.mode == "all":
            self.quant_modes = self.support_quant_modes
        else:
            if self.mode not in self.support_quant_modes:
                raise ValueError(f"Chip {self.chip} does not support mode: {self.mode}")
            self.quant_modes = [self.mode]

    class Desc:
        """Descriptor for data generation parameters."""

        def __init__(self, dtype: str, min: float = -10, max: float = 10) -> None:
            """
            Initialize data descriptor.

            Args:
                dtype: Data type string (e.g., 'float32', 'float16')
                min: Minimum value for data generation
                max: Maximum value for data generation
            """
            self.dtype = dtype
            self.min = min
            self.max = max

    @property
    def test_function(self) -> Dict[str, Tuple[Callable, bool, bool]]:
        """Get registered test functions."""
        return self._test_functions

    @classmethod
    def unique_name(cls, name: str) -> str:
        """Generate a unique name with ID suffix."""
        unique = f"{name}_{cls._id}"
        cls._id += 1
        return unique

    @run_in_log_wrapper
    def test_single(self, case: str):
        """Run a single test case."""
        np.random.seed(0)
        MLIR_IR_TESTER._id = 0

        print(f"Test: {case}")

        if case not in self._test_functions:
            raise RuntimeError(f"Test case '{case}' does not exist")

        func, _, _ = self._test_functions[case]

        with change_directory(case):
            func(case)

        print(f"====== TEST {case} Success ======")

    def check_support(self, case: str) -> bool:
        """Check if a test case is supported by the current chip."""
        if case not in self._test_functions:
            return False

        _, bm1684x_support, bm1688_support = self._test_functions[case]

        if self.chip == "bm1684x" and bm1684x_support:
            return True
        if self.chip == "bm1688" and bm1688_support:
            return True

        return False

    def _L(self, block_mlir: MLIRImporter, names: Union[str, List[str]]) -> Location:
        """
        Create MLIR location.

        Args:
            block_mlir: MLIR importer instance
            names: Location name(s) as string or list of strings

        Returns:
            MLIR Location object
        """
        if isinstance(names, str):
            return Location.fused([Location.name(names)], context=block_mlir.ctx)
        elif isinstance(names, list):
            return Location.fused([Location.name(n) for n in names], context=block_mlir.ctx)
        else:
            raise TypeError(f"Unsupported type for names: {type(names)}")

    def _T(self, block_mlir: MLIRImporter, shape: List[int]):
        """
        Get MLIR tensor type for shape.

        Args:
            block_mlir: MLIR importer instance
            shape: Tensor shape

        Returns:
            MLIR tensor type
        """
        return block_mlir.get_tensor_type(shape)

    def _create_input_ops(self, block_mlir: MLIRImporter, input_shapes: List[List[int]]) -> List:
        """
        Create input operations for all input shapes.

        Args:
            block_mlir: MLIR importer instance
            input_shapes: List of input tensor shapes

        Returns:
            List of input operation outputs
        """
        inputs = []
        for i, shape in enumerate(input_shapes):
            loc = self._L(block_mlir, f"in{i}")
            input_op = block_mlir.create_input_op(loc, i)
            inputs.append(input_op)
        return inputs

    def _create_mlir_importer(
            self,
            case_name: str,
            input_shapes: List[List[int]],
            weight_shapes: List[List[int]],
            output_shapes: List[List[int]],
            input_types: Optional[List[str]] = None) -> Tuple[MLIRImporter, List, List, Any]:
        """
        Create MLIR importer and return it along with operations.

        Args:
            case_name: Test case name
            input_shapes: List of input tensor shapes
            weight_shapes: List of weight tensor shapes
            output_shapes: List of output tensor shapes
            input_types: List of input tensor types (default: all F32)

        Returns:
            Tuple of (MLIRImporter instance, input ops, weight ops, insert_point)
        """
        if input_types is None:
            input_types = ["F32" for _ in input_shapes]
        elif len(input_shapes) != len(input_types):
            raise ValueError(f"input_types length ({len(input_types)}) must match "
                             f"input_shapes length ({len(input_shapes)})")

        block_mlir = MLIRImporter(input_shapes, output_shapes, case_name, Platform.LLM, input_types)
        # Create input operations
        input_ops = self._create_input_ops(block_mlir, input_shapes)
        # Create weight operations
        weight_ops = [
            block_mlir.create_weight_op(f"weight{i}", shape)
            for i, shape in enumerate(weight_shapes)
        ]

        ip = block_mlir.insert_point
        return block_mlir, input_ops, weight_ops, ip

    def _save_mlir_and_data(self,
                            case_name: str,
                            block_mlir: MLIRImporter,
                            input_shapes: List[List[int]],
                            weight_shapes: List[List[int]],
                            input_descs: Optional[List[Desc]] = None,
                            weight_descs: Optional[List[Desc]] = None):
        """
        Save MLIR text, weights, and input data to files.

        Args:
            case_name: Test case name
            block_mlir: MLIR importer instance
            input_shapes: List of input tensor shapes
            weight_shapes: List of weight tensor shapes
            input_descs: List of Desc objects for input data generation (default: float32, -10 to 10)
            weight_descs: List of Desc objects for weight data generation (default: float32, -10 to 10)
        """
        # Use default Desc if not provided
        if input_descs is None:
            input_descs = [self.Desc('float32') for _ in input_shapes]
        if weight_descs is None:
            weight_descs = [self.Desc('float32') for _ in weight_shapes]

        # Ensure descs match shapes length
        if len(input_descs) != len(input_shapes):
            raise ValueError(
                f"input_descs length ({len(input_descs)}) must match input_shapes length ({len(input_shapes)})"
            )
        if len(weight_descs) != len(weight_shapes):
            raise ValueError(
                f"weight_descs length ({len(weight_descs)}) must match weight_shapes length ({len(weight_shapes)})"
            )

        # Generate weights using dictionary comprehension
        weights = {
            f"weight{i}": rand_data(shape, desc.dtype, desc.min, desc.max)
            for i, (shape, desc) in enumerate(zip(weight_shapes, weight_descs))
        }

        # Save MLIR text
        mlir_txt = block_mlir.print_module()
        with open(f"{case_name}.mlir", "w") as f:
            f.write(mlir_txt)

        # Save weights and inputs
        weight_file = f"{case_name}_top_f32_all_origin_weight.npz"
        np.savez(weight_file, **weights)
        if not self.no_check:
            # Generate inputs using dictionary comprehension
            inputs = {
                f"in{i}": rand_data(shape, desc.dtype, desc.min, desc.max)
                for i, (shape, desc) in enumerate(zip(input_shapes, input_descs))
            }
            np.savez(f"{case_name}_input.npz", **inputs)

    def _deploy_test_case(self, case_name: str, tolerance: Tuple[float,
                                                                 float] = (0.98, 0.95)) -> None:
        """
        Deploy test case for each quantization mode.

        Args:
            case_name: Test case name
            tolerance: Tolerance tuple (cos_tol, euclidean_tol)
        """
        for mode in self.quant_modes:
            try:
                deploy_case_bmodel(case_name=case_name,
                                   chip=self.chip,
                                   mode=mode,
                                   tolerance=tolerance,
                                   debug=self.debug,
                                   dynamic=self.dynamic,
                                   num_core=self.num_core,
                                   rvti=self.rvti,
                                   disable_lg=self.disable_lg,
                                   no_check=self.no_check)
            except Exception as e:
                # print(f"[Error] Mode {mode} failed for {case_name}: {e}")
                raise RuntimeError(
                    f"Deployment failed for case '{case_name}' in mode '{mode}'") from e

    def test_error0(self, case_name):
        """Test case error0: Complex RMSNorm + Rope operations with Reshape."""
        # Define input and output shapes
        input_shapes = [
            [1, 1024, 8, 128],  # in0
            [1, 1024, 1, 128],  # in1
            [1, 1024, 1, 128],  # in2
            [1, 1024, 2048],  # in3
        ]
        weight_shapes = [
            [1, 1, 1, 128],  # weight0
            [1, 1, 1, 128]  # weight1
        ]
        output_shapes = [
            [1, 1024, 8, 128],  # out0
            [1, 1024, 16, 128],  # out1
        ]

        # Create MLIR importer
        block_mlir, input_ops, weight_ops, ip = self._create_mlir_importer(
            case_name, input_shapes, weight_shapes, output_shapes, ["F32", "F32", "F32", "F32"])

        in0, in1, in2, in3 = input_ops
        # First RMSNorm + Rope
        rmsnorm0 = top.RMSNormOp(self._T(block_mlir, input_shapes[0]),
                                 in0,
                                 weight_ops[0],
                                 eps=1e-6,
                                 loc=self._L(block_mlir, "rmsnorm0"),
                                 ip=ip).output

        rope0 = top.RopeOp(self._T(block_mlir, input_shapes[0]),
                           rmsnorm0,
                           in1,
                           in2,
                           rope_mode=StringAttr.get("contiguous_halves"),
                           loc=self._L(block_mlir, "rope0"),
                           ip=ip).output

        # Reshape
        reshape = top.ReshapeOp(self._T(block_mlir, output_shapes[1]),
                                in3,
                                shape=[1, -1, 16, 128],
                                loc=self._L(block_mlir, "reshape"),
                                ip=ip).output

        # Second RMSNorm + Rope
        rmsnorm1 = top.RMSNormOp(self._T(block_mlir, output_shapes[1]),
                                 reshape,
                                 weight_ops[1],
                                 eps=1e-6,
                                 loc=self._L(block_mlir, "rmsnorm1"),
                                 ip=ip).output

        rope1 = top.RopeOp(self._T(block_mlir, output_shapes[1]),
                           rmsnorm1,
                           in1,
                           in2,
                           rope_mode=StringAttr.get("contiguous_halves"),
                           loc=self._L(block_mlir, "rope1"),
                           ip=ip).output

        # Create return operation
        block_mlir.create_return_op([rope0, rope1])

        # Save MLIR text, weights, and inputs
        self._save_mlir_and_data(case_name, block_mlir, input_shapes, weight_shapes)

        # Deploy for each quantization mode
        self._deploy_test_case(case_name, tolerance=(0.1, 0.1))

    def test_insert(self, case_name):
        """Test case1: Simple RMSNorm operation."""
        input_shapes = [
            [16, 64, 8, 128],  # in0
            [16, 4, 8, 128],  # in1
        ]
        weight_shapes = [
            [1, 1, 1, 128],  # weight0
        ]
        output_shapes = [
            [16, 64, 8, 128],  # out0
        ]

        # Create MLIR importer
        block_mlir, input_ops, weight_ops, ip = self._create_mlir_importer(
            case_name, input_shapes, weight_shapes, output_shapes, ["F32", "F32"])
        in0_op, in1_op = input_ops
        # Create RMSNorm operation
        op0 = top.RMSNormOp(self._T(block_mlir, input_shapes[0]),
                            in0_op,
                            weight_ops[0],
                            eps=1e-6,
                            loc=self._L(block_mlir, "rmsnorm0"),
                            ip=ip).output

        op1 = top.InsertOp(self._T(block_mlir, input_shapes[0]),
                           op0,
                           in1_op,
                           axis=1,
                           offset=32,
                           loc=self._L(block_mlir, "insert0"),
                           ip=ip).output

        # Create return operation
        block_mlir.create_return_op([op1])

        # Save MLIR text, weights, and inputs
        self._save_mlir_and_data(case_name,
                                 block_mlir,
                                 input_shapes,
                                 weight_shapes,
                                 input_descs=[self.Desc('float32', -5, 5) for _ in input_shapes],
                                 weight_descs=None)

        # Deploy for each quantization mode
        self._deploy_test_case(case_name)

    def test_slice(self, case_name):
        """Test case slice: RMSNorm followed by two Slice ops to produce 2 outputs."""
        input_shapes = [
            [1, 32, 128],  # in0
        ]
        weight_shapes = [
            [1, 32, 1],  # weight0 for rmsnorm
        ]
        output_shapes = [
            [1, 32, 50],  # out0: slice first half along axis 1
            [1, 32, 50],  # out1: slice second half along axis 1
        ]

        # Create MLIR importer
        block_mlir, input_ops, weight_ops, ip = self._create_mlir_importer(
            case_name, input_shapes, weight_shapes, output_shapes, ["F32"])

        in0_op = input_ops[0]

        # RMSNorm
        rmsnorm_out = top.AddOp(self._T(block_mlir, input_shapes[0]), [in0_op, weight_ops[0]],
                                loc=self._L(block_mlir, "rmsnorm0"),
                                ip=ip).output

        # Slice 0: first half [0:32] along axis 1
        slice0 = top.SliceOp(self._T(block_mlir, output_shapes[0]),
                             rmsnorm_out,
                             block_mlir.none_op,
                             block_mlir.none_op,
                             block_mlir.none_op,
                             offset=[0, 0, 46],
                             steps=[1, 1, 1],
                             ends=[1, 32, 96],
                             loc=self._L(block_mlir, "slice0"),
                             ip=ip).output

        # Slice 1: second half [32:64] along axis 1
        slice1 = top.SliceOp(self._T(block_mlir, output_shapes[1]),
                             rmsnorm_out,
                             block_mlir.none_op,
                             block_mlir.none_op,
                             block_mlir.none_op,
                             offset=[0, 0, -50],
                             steps=[1, 1, 1],
                             ends=[1, 32, 128],
                             loc=self._L(block_mlir, "slice1"),
                             ip=ip).output

        # Create return operation
        block_mlir.create_return_op([slice0, slice1])

        # Save MLIR text, weights, and inputs
        self._save_mlir_and_data(case_name, block_mlir, input_shapes, weight_shapes)

        # Deploy for each quantization mode
        self._deploy_test_case(case_name)

    def test_fattention(self, case_name):
        """Test case fattention: Fused attention with multiple inputs/outputs."""
        S = 64
        D = 128
        input_shapes = [
            [1, S, 16, D],  # Q
            [1, S, 2, D],  # K
            [1, S, 2, D],  # V
            [1, 1, S, S],
        ]
        weight_shapes = [
            [1, 1, 1, D],  # weight0
            [1, 1, 1, D],  # weight1
            [1, 1, 1, D],  # weight2
        ]
        output_shapes = [
            [1, S, 16, D],  # out0
        ]

        # Create MLIR importer
        block_mlir, input_ops, weight_ops, ip = self._create_mlir_importer(
            case_name, input_shapes, weight_shapes, output_shapes, ["F32", "F32", "F32", "F32"])

        in0_op, in1_op, in2_op, in3_op = input_ops
        q_op = top.RMSNormOp(self._T(block_mlir, input_shapes[0]),
                             in0_op,
                             weight_ops[0],
                             eps=1e-6,
                             loc=self._L(block_mlir, "rmsnorm0"),
                             ip=ip).output
        k_op = top.RMSNormOp(self._T(block_mlir, input_shapes[0]),
                             in1_op,
                             weight_ops[1],
                             eps=1e-6,
                             loc=self._L(block_mlir, "rmsnorm1"),
                             ip=ip).output
        v_op = top.RMSNormOp(self._T(block_mlir, input_shapes[0]),
                             in2_op,
                             weight_ops[2],
                             eps=1e-6,
                             loc=self._L(block_mlir, "rmsnorm2"),
                             ip=ip).output

        op = top.FAttentionOp(self._T(block_mlir, output_shapes[0]),
                              q_op,
                              k_op,
                              v_op,
                              in3_op,
                              block_mlir.none_op,
                              batch=1,
                              q_head=16,
                              kv_head=2,
                              dim=D,
                              scale=1 / (D**0.5),
                              mq=S,
                              mk=S,
                              keep_dims=True,
                              loc=self._L(block_mlir, "fattention"),
                              ip=ip).output

        # Create return operation
        block_mlir.create_return_op([op])

        # Save MLIR text, weights, and inputs
        self._save_mlir_and_data(case_name,
                                 block_mlir,
                                 input_shapes,
                                 weight_shapes,
                                 input_descs=[self.Desc('float32', -5, 5) for _ in input_shapes],
                                 weight_descs=None)

        # Deploy for each quantization mode
        self._deploy_test_case(case_name)

    def test_a16matmul(self, case_name):
        """Test case A16MatMul: Simple A16MatMul operation."""
        B = 1
        S = 1024
        K = 2560
        N = 32
        q_group_size = 128
        weight_bits = 4

        # For weight_bits=4, weight is packed: [N, K // (8 // weight_bits)] = [N, K // 2]
        weight_packed_dim = K // (8 // weight_bits)
        # Scale/zp shape: [N, K // q_group_size]
        scale_dim = K // q_group_size

        input_shapes = [
            [B, S, K],  # input activation
        ]
        output_shapes = [
            [B, S, N],  # matmul output
        ]

        weight_shape = [N, weight_packed_dim]
        scale_shape = [N, scale_dim]
        zp_shape = [N, scale_dim]

        input_types = ["F32"]
        block_mlir = MLIRImporter(input_shapes, output_shapes, case_name, Platform.LLM, input_types)
        input_ops = self._create_input_ops(block_mlir, input_shapes)
        ip = block_mlir.insert_point

        in0_op = input_ops[0]

        # Create weight ops with appropriate types
        qweight_op = block_mlir.create_weight_op("qweight", weight_shape, "UINT8")
        scale_op = block_mlir.create_weight_op("scales", scale_shape, "F32")
        zp_op = block_mlir.create_weight_op("qzeros", zp_shape, "UINT8")

        # A16MatMul operation
        out = top.A16MatMulOp(self._T(block_mlir, output_shapes[0]),
                              in0_op,
                              qweight_op,
                              scale_op,
                              zp_op,
                              block_mlir.none_op,
                              right_transpose=True,
                              q_group_size=q_group_size,
                              weight_bits=weight_bits,
                              loc=self._L(block_mlir, "a16matmul0"),
                              ip=ip).output

        # Create return operation
        block_mlir.create_return_op([out])

        # Generate data
        inputs = {
            "in0": rand_data(input_shapes[0], 'float32', -1, 1),
        }
        weights = {
            "qweight": rand_data(weight_shape, 'uint8', 0, 255),
            "scales": rand_data(scale_shape, 'float32', -1, 1),
            "qzeros": rand_data(zp_shape, 'uint8', 0, 255),
        }

        # Save MLIR text
        mlir_txt = block_mlir.print_module()
        with open(f"{case_name}.mlir", "w") as f:
            f.write(mlir_txt)

        # Save weights and inputs
        np.savez(f"{case_name}_top_f32_all_origin_weight.npz", **weights)
        if not self.no_check:
            np.savez(f"{case_name}_input.npz", **inputs)

        # Deploy for each quantization mode (A16MatMul only supports F16/BF16 lowering)
        saved_modes = self.quant_modes
        self.quant_modes = [m for m in self.quant_modes if m in ["f16", "bf16"]]
        self._deploy_test_case(case_name)
        self.quant_modes = saved_modes

    def test_chunk_gated_delta_rule(self, case_name):
        """Test case: ChunkGatedDeltaRule operator."""
        B = 1
        S = 128
        num_qk_heads = 16
        num_v_heads = 64
        D = 128
        chunk_size = 64
        scale = 1.0 / (D**0.5)

        input_shapes = [
            [B, S, num_qk_heads, D],  # query
            [B, S, num_qk_heads, D],  # key
            [B, S, num_v_heads, D],  # value
            [B, S, num_v_heads],  # g
            [B, S, num_v_heads],  # beta
            [B, num_v_heads, D, D],  # recurrent_state
        ]
        weight_shapes = [
            [chunk_size, chunk_size],  # triu_mask
            [chunk_size, chunk_size],  # strict_triu_mask
            [chunk_size, chunk_size],  # tril_mask
            [chunk_size, chunk_size],  # eye
        ]
        output_shapes = [
            [B, S, num_v_heads, D],  # attn_out
        ]

        # Create MLIR importer
        block_mlir, input_ops, weight_ops, ip = self._create_mlir_importer(
            case_name, input_shapes, weight_shapes, output_shapes, ["F32"] * len(input_shapes))

        q_op, k_op, v_op, g_op, beta_op, state_op = input_ops
        triu_mask_op, strict_triu_mask_op, tril_mask_op, eye_op = weight_ops

        op = top.ChunkGatedDeltaRuleOp(self._T(block_mlir, output_shapes[0]),
                                       q_op,
                                       k_op,
                                       v_op,
                                       g_op,
                                       beta_op,
                                       state_op,
                                       triu_mask_op,
                                       strict_triu_mask_op,
                                       tril_mask_op,
                                       eye_op,
                                       num_k_heads=num_qk_heads,
                                       num_v_heads=num_v_heads,
                                       d=D,
                                       chunk_size=chunk_size,
                                       use_qk_l2norm=True,
                                       scale=scale,
                                       loc=self._L(block_mlir, "chunk_gated_delta_rule"),
                                       ip=ip).attn_out

        # Create return operation
        block_mlir.create_return_op([op])

        # Generate mask weights with correct values
        cs = chunk_size
        triu_mask = np.triu(np.ones((cs, cs), dtype=np.float32), k=0)
        strict_triu_mask = np.triu(np.ones((cs, cs), dtype=np.float32), k=1)
        tril_mask = np.tril(np.ones((cs, cs), dtype=np.float32), k=0)
        eye_mask = np.eye(cs, dtype=np.float32)

        # Generate input data with appropriate ranges
        inputs = {
            "in0": rand_data(input_shapes[0], 'float32', -1, 1),  # query
            "in1": rand_data(input_shapes[1], 'float32', -1, 1),  # key
            "in2": rand_data(input_shapes[2], 'float32', -1, 1),  # value
            "in3": rand_data(input_shapes[3], 'float32', -0.5, 0),  # g (small negative for decay)
            "in4": rand_data(input_shapes[4], 'float32', 0, 1),  # beta (0 to 1)
            "in5": rand_data(input_shapes[5], 'float32', -0.1, 0.1),  # recurrent_state
        }
        weights = {
            "weight0": triu_mask,
            "weight1": strict_triu_mask,
            "weight2": tril_mask,
            "weight3": eye_mask,
        }

        # Save MLIR text
        mlir_txt = block_mlir.print_module()
        with open(f"{case_name}.mlir", "w") as f:
            f.write(mlir_txt)

        # Save weights and inputs
        np.savez(f"{case_name}_top_f32_all_origin_weight.npz", **weights)
        np.savez(f"{case_name}_input.npz", **inputs)

        # Deploy for each quantization mode
        self._deploy_test_case(case_name, tolerance=(0.9, 0.8))

    def test_recurrent_gated_delta_rule(self, case_name):
        """Test case: RecurrentGatedDeltaRule operator."""
        B = 1
        num_qk_heads = 16
        num_v_heads = 16
        D = 128
        scale = 1.0 / (D**0.5)

        input_shapes = [
            [B, 1, num_qk_heads, D],  # query
            [B, 1, num_qk_heads, D],  # key
            [B, 1, num_v_heads, D],  # value
            [B, 1, num_v_heads],  # g
            [B, 1, num_v_heads],  # beta
            [B, num_v_heads, D, D],  # recurrent_state
        ]
        weight_shapes = []
        output_shapes = [
            [B, 1, num_v_heads, D],  # attn_out
        ]

        # Create MLIR importer
        block_mlir, input_ops, weight_ops, ip = self._create_mlir_importer(
            case_name, input_shapes, weight_shapes, output_shapes, ["F32"] * len(input_shapes))

        q_op, k_op, v_op, g_op, beta_op, state_op = input_ops

        op = top.RecurrentGatedDeltaRuleOp(self._T(block_mlir, output_shapes[0]),
                                           q_op,
                                           k_op,
                                           v_op,
                                           g_op,
                                           beta_op,
                                           state_op,
                                           num_k_heads=num_qk_heads,
                                           num_v_heads=num_v_heads,
                                           d=D,
                                           use_qk_l2norm=True,
                                           scale=scale,
                                           loc=self._L(block_mlir, "recurrent_gated_delta_rule"),
                                           ip=ip).attn_out

        # Create return operation
        block_mlir.create_return_op([op])

        # Generate input data with appropriate ranges
        inputs = {
            "in0": rand_data(input_shapes[0], 'float32', -1, 1),  # query
            "in1": rand_data(input_shapes[1], 'float32', -1, 1),  # key
            "in2": rand_data(input_shapes[2], 'float32', -1, 1),  # value
            "in3": rand_data(input_shapes[3], 'float32', -0.5, 0),  # g (small negative for decay)
            "in4": rand_data(input_shapes[4], 'float32', 0, 1),  # beta (0 to 1)
            "in5": rand_data(input_shapes[5], 'float32', -1.0, 1.0),  # recurrent_state
        }

        # Save MLIR text
        mlir_txt = block_mlir.print_module()
        with open(f"{case_name}.mlir", "w") as f:
            f.write(mlir_txt)

        # Save inputs
        np.savez(f"{case_name}_input.npz", **inputs)

        # Deploy for each quantization mode
        self._deploy_test_case(case_name, tolerance=(0.9, 0.8))

    def test_concat_slice(self, case_name):
        """Test case: ConcatSlice operator - concat along axis then slice to keep original shape."""
        N, C, H, W = 1, 2048, 32, 1
        axis = 2
        H1 = 8  # size of in1 along the concat axis

        input_shapes = [
            [N, C, H, W],  # in0
            [N, C, H1, W],  # in1
        ]
        weight_shapes = []
        output_shapes = [
            [N, C, H, W],  # output (same as in0)
        ]

        # Create MLIR importer
        block_mlir, input_ops, weight_ops, ip = self._create_mlir_importer(
            case_name, input_shapes, weight_shapes, output_shapes, ["F32"] * len(input_shapes))

        in0_op, in1_op = input_ops

        op = top.ConcatSliceOp(self._T(block_mlir, output_shapes[0]),
                               in0_op,
                               in1_op,
                               axis=axis,
                               loc=self._L(block_mlir, "concat_slice"),
                               ip=ip).output

        # Create return operation
        block_mlir.create_return_op([op])

        # Save MLIR text, weights, and inputs
        self._save_mlir_and_data(case_name, block_mlir, input_shapes, weight_shapes)

        # Deploy for each quantization mode
        self._deploy_test_case(case_name)


def test_one_case_in_all(tester: MLIR_IR_TESTER, case: str, error_cases: List,
                         success_cases: List) -> None:
    """Run a single test case and record results."""
    t = Timer()
    try:
        tester.test_single(case)
        success_cases.append(f"{case}:{int(t.elapsed_time())}s")
    except Exception as e:
        import traceback
        error_cases.append(f"{case}:{int(t.elapsed_time())}s")
        print(f"[Error] Test case '{case}' failed: {e}")
        traceback.print_exc()


def test_all_base(tester: 'MLIR_IR_TESTER') -> Tuple[List[str], List[str]]:
    """Run all test cases with multiprocessing."""
    process_number = multiprocessing.cpu_count() // 2 + 1

    with multiprocessing.Manager() as manager:
        error_cases_mp = manager.list()
        success_cases_mp = manager.list()
        processes = []

        for case in tester.test_function:
            if not tester.check_support(case):
                continue
            p = multiprocessing.Process(target=test_one_case_in_all,
                                        name=case,
                                        args=(tester, case, error_cases_mp, success_cases_mp))
            processes.append(p)
            if len(processes) >= process_number:
                collect_process(processes, error_cases_mp)
                processes = []
        # Collect remaining processes
        if processes:
            collect_process(processes, error_cases_mp)
            processes = []

        # Copy results to regular lists before Manager is closed
        error_cases = list(error_cases_mp)
        success_cases = list(success_cases_mp)

    print(f"Success: {success_cases}")
    print(f"Failure: {error_cases}")

    status = 'Failed' if error_cases else 'Success'
    print(f"====== test_mlir.py --chip {tester.chip} TEST {status} ======")

    return error_cases, success_cases


def test_all(tester: MLIR_IR_TESTER) -> Tuple[List[str], List[str]]:
    """Run all test cases and return results."""
    return test_all_base(tester)


def main():
    """Main entry point."""

    parser = argparse.ArgumentParser()
    # yapf: disable
    parser.add_argument("--chip", default="bm1684x", type=str,
                        choices=SUPPORTED_CHIPS, help="chip platform name")
    parser.add_argument("--mode", default="all", type=str, choices=['all', 'f32', 'f16', 'bf16'], help="quantize modes, only supports fp for now")
    parser.add_argument("--simple", action="store_true", help='do simple test for commit test')
    parser.add_argument("--case", default="all", type=str, help="test one case, if all, then test all cases")
    parser.add_argument("--show_all", action="store_true", help='show all cases')
    parser.add_argument("--debug", action="store_true", help='whether to save intermediate file')
    parser.add_argument("--report", default="", type=str, help="report file name")
    parser.add_argument("--concise_log", action="store_true", help="use concise log")
    parser.add_argument("--num_core", default=1, type=int, help="number of cores to use")
    parser.add_argument("--dynamic", action="store_true", help='enable dynamic compile and inference')
    parser.add_argument("--rvti", action="store_true", help='enable rvti, only for bm1684x2 and bm1690e')
    parser.add_argument("--no_check", action="store_true", help='do not check result, only run deploy')
    parser.add_argument("--disable_lg", action="store_true", help='disable layergroup')
    # yapf: enable
    args = parser.parse_args()
    tester = MLIR_IR_TESTER(args)
    # Handle show_all flag
    if args.show_all:
        print("====== Show All Cases ============")
        for case in tester.test_function:
            print(case)
        return

    output_dir = f"mlir_test_{args.chip}"
    os.makedirs(output_dir, exist_ok=True)

    # Run tests
    with change_directory(output_dir):
        if args.case.lower() == "all":
            error_cases, success_cases = test_all_base(tester) if args.report else test_all(tester)
        else:
            tester.test_single(args.case)
            error_cases, success_cases = [], []

    # Save report if requested
    if args.report and (error_cases or success_cases):
        result = {'success': list(success_cases), 'failure': list(error_cases)}
        with open(args.report, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
