#!/usr/bin/env python3
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
"""MLIR-level regression tester.

Drives small hand-built Top-dialect MLIR cases (attention kernels, slice,
softplus-mul, …) end-to-end through ``model_deploy.py``. Common helpers
(random data, working-directory context manager, chip-support resolver) come
from :mod:`_test_base`.
"""

import argparse
import json
import multiprocessing
import os
import shlex
import subprocess
import sys
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from utils.tpu_info import get_tpu_info

import numpy as np

from mlir.ir import *
import mlir.dialects.top as top
from transform.MLIRImporter import MLIRImporter, Platform

from utils.misc import collect_process
from utils.regression_logger import run_in_log_wrapper
from utils.timer import Timer

from _test_base import (
    Y,
    N,
    change_directory,
    make_chip_resolver,
    rand_data,
)

# Constants
SUPPORTED_CHIPS = ["bm1684x", "bm1688", "bm1690", "bm1690e", "bm1684x2"]
SUPPORTED_MODES = ["f32", "f16", "bf16"]  # Extend as needed

# Chip columns of the per-row support tuples in ``_test_functions``.
_CHIP_COLUMNS = ("bm1684x", "bm1688")
# ``bm1684x2`` shares the ``bm1684x`` support column.
_CHIP_ALIASES = {"bm1684x2": "bm1684x"}
_resolve_chip_support = make_chip_resolver(_CHIP_COLUMNS, _CHIP_ALIASES)


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
                       disable_hp: bool = False,
                       no_check: bool = False,
                       ip: str = "",
                       pwd: str = "") -> None:
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
        test_args, f"--quantize {mode.upper()}", f"--num_core {num_core}"
    ]
    if debug:
        deploy_cmd.append("--debug")
    if dynamic:
        deploy_cmd.append("--dynamic")
        deploy_cmd.append("--addr_mode basic")
    if rvti:
        deploy_cmd.append("--rvti")
    if disable_lg:
        deploy_cmd.append("--disable_layer_group")
    if not disable_hp:
        deploy_cmd.append("--high_precision")
    deploy_cmd.append("--disable_gdma_check")
    if ip:
        deploy_cmd.append(f"--ip {ip}")
        if pwd:
            deploy_cmd.append(f"--pwd {shlex.quote(pwd)}")
    deploy_cmd = " ".join(deploy_cmd)
    print(deploy_cmd)
    assert os.system(deploy_cmd) == 0


Failed_Cases = []


class MLIR_IR_TESTER(object):
    _id = 0  # Class variable for generating unique names

    # This class is built for testing single operator transform.
    def __init__(self, args):
        # Test function registry with chip support
        self._test_functions = {
            #############################
            # MLIR Test Case, Alphabetically
            #############################
            # case:  (test_function,      bm1684x_support, bm1688_support)
            "error0": (self.test_error0, Y, Y),
            "layernorm_dynamic_dc": (self.test_layernorm_dynamic_dc, Y, Y),
            "layernorm_dynamic_dc_workaround": (self.test_layernorm_dynamic_dc_workaround, Y, Y),
            "insert": (self.test_insert, Y, Y),
            "fattention": (self.test_fattention, Y, Y),
            "fattention_prefill": (self.test_fattention_prefill, Y, Y),
            "fattention_decode": (self.test_fattention_decode, Y, Y),
            "fattn_o_proj": (self.test_fattn_o_proj, Y, Y),
            "fp8matmul": (self.test_fp8matmul, Y, Y),
            "slice": (self.test_slice, Y, Y),
            "a16matmul": (self.test_a16matmul, Y, Y),
            "a16gather": (self.test_a16gather, Y, Y),
            "chunk_gated_delta_rule": (self.test_chunk_gated_delta_rule, Y, Y),
            "recurrent_gated_delta_rule": (self.test_recurrent_gated_delta_rule, Y, Y),
            "concat_slice": (self.test_concat_slice, Y, Y),
            "softplus_mul": (self.test_softplus_mul, Y, Y),
            "softmax_topk": (self.test_softmax_topk, Y, Y),
            "conv2d_non_overlapping": (self.test_conv2d_non_overlapping, Y, Y),
            "matmul_reshape_permute": (self.test_matmul_reshape_permute, Y, Y),
            "matmul_dynamic": (self.test_matmul_dynamic, Y, Y),
            "reshape_dynamic": (self.test_reshape_dynamic, Y, Y),
            "permute_dynamic": (self.test_permute_dynamic, Y, Y),
        }
        # currently test_mlir.py only supports fp quant mode
        self.support_quant_modes = ["f32", "f16", "bf16"]
        self.mode = args.mode.lower()
        self.simple = args.simple
        self.chip = args.chip.lower()
        self.concise_log = args.concise_log  # use when run regression/main_entry.py
        self.num_core = args.num_core
        self.debug = args.debug
        self.dynamic = args.dynamic
        self.rvti = args.rvti
        self.disable_lg = args.disable_lg
        self.disable_hp = args.disable_hp
        self.no_check = args.no_check
        self.ip = getattr(args, "ip", "")
        self.pwd = getattr(args, "pwd", "")
        self.weights_path = getattr(args, "weights_path", "").strip()
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

        func = self._test_functions[case][0]

        with change_directory(case):
            func(case)

        print(f"====== TEST {case} Success ======")

    def check_support(self, case: str) -> bool:
        """Check if a test case is supported by the current chip."""
        if case not in self._test_functions:
            return False
        flags = self._test_functions[case][1:]
        return _resolve_chip_support(self.chip, flags)

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
        Run shape-infer sanity check, then deploy test case for each quantization mode.

        Args:
            case_name: Test case name
            tolerance: Tolerance tuple (cos_tol, euclidean_tol)
        """
        # Run shape-infer as a sanity check on the generated mlir
        mlir_path = f"{case_name}.mlir"
        try:
            subprocess.run(
                [
                    "tpuc-opt", "--shape-infer", f"{case_name}.mlir", "-o",
                    f"{case_name}_shape_infer.mlir"
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"shape-infer failed for {mlir_path}:\n"
                               f"stdout: {e.stdout}\nstderr: {e.stderr}") from e

        for mode in self.quant_modes:
            try:
                deploy_case_bmodel(case_name=case_name,
                                   chip=self.chip,
                                   mode=mode,
                                   tolerance=tolerance,
                                   test_reference=None,
                                   debug=self.debug,
                                   dynamic=self.dynamic,
                                   num_core=self.num_core,
                                   rvti=self.rvti,
                                   disable_lg=self.disable_lg,
                                   disable_hp=self.disable_hp,
                                   no_check=self.no_check,
                                   ip=self.ip,
                                   pwd=self.pwd)
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

    def test_layernorm_dynamic_dc(self, case_name):
        """Test: MoonViT norm0 LayerNorm (bf16, dynamic).

        Mirrors vision_block norm0: input [N,1152], weight/bias [1,1152],
        top.LayerNormOp(axis=1, eps=1e-5). Uses the real dumped pos_emb_add
        as input to reproduce the LocateAnything precision issue.
        With --dynamic: compile max_N=4096, test N=1656.
        """
        D = 1152
        max_N = 4096
        test_N = 1656 if self.dynamic else max_N

        input_shapes = [[max_N, D]]
        weight_shapes = [[1, D], [1, D]]  # weight, bias
        output_shapes = [[max_N, D]]

        block_mlir, input_ops, weight_ops, ip = self._create_mlir_importer(
            case_name, input_shapes, weight_shapes, output_shapes, ["F32"])

        out = top.LayerNormOp(self._T(block_mlir, [max_N, D]),
                              input_ops[0],
                              weight_ops[0],
                              weight_ops[1],
                              normalized_shape=[D],
                              axis=1,
                              eps=1e-5,
                              loc=self._L(block_mlir, "norm0"),
                              ip=ip).output

        block_mlir.create_return_op([out])

        actual_input_shapes = [[test_N, D]]
        # use real pos_emb_add if available, else random
        try:
            real = np.load('/tmp/la_dump_tf/vit.npz')['pos_emb_add'][:test_N]
            if real.shape[0] < test_N:  # not enough, fall back
                raise KeyError
        except Exception:
            real = (np.random.randn(test_N, D) * 1.0).astype(np.float32)
        input_descs = [self.Desc('float32', -10, 10)]
        weight_descs = [self.Desc('float32', -10, 10), self.Desc('float32', -10, 10)]
        self._save_mlir_and_data(case_name,
                                 block_mlir,
                                 actual_input_shapes,
                                 weight_shapes,
                                 input_descs=input_descs,
                                 weight_descs=weight_descs)
        np.savez(f"{case_name}_input.npz", in0=real)
        # override weights with the REAL norm0 weight/bias from the model
        try:
            import glob
            from safetensors import safe_open
            Wn = Bn = None
            for sf in glob.glob(
                    '/workspace/llm/locateanything/LocateAnything-3B-AutoRound-W4A16/*.safetensors'
            ):
                with safe_open(sf, 'pt') as m:
                    for k in m.keys():
                        if k.endswith('encoder.blocks.0.norm0.weight'):
                            Wn = m.get_tensor(k).cpu().float().numpy().reshape(1, D)
                        if k.endswith('encoder.blocks.0.norm0.bias'):
                            Bn = m.get_tensor(k).cpu().float().numpy().reshape(1, D)
            if Wn is not None and Bn is not None:
                np.savez(f"{case_name}_top_f32_all_origin_weight.npz", weight0=Wn, weight1=Bn)
        except Exception as e:
            print(f"[warn] could not load real norm0 weights: {e}")

        self._deploy_test_case(case_name, tolerance=(0.99, 0.98))

    def test_layernorm_dynamic_dc_workaround(self, case_name):
        """Test: norm0 LayerNorm with per-position DC subtracted BEFORE LN.

        Workaround for the bf16-dynamic LayerNorm bug (naive var=E[x^2]-E[x]^2
        has catastrophic cancellation when per-position mean is non-zero).
        LN(x - mean(x)) == LN(x) mathematically, but feeding a zero-DC input
        makes the kernel's var computation stable.
        """
        D = 1152
        max_N = 4096
        test_N = 1656 if self.dynamic else max_N

        input_shapes = [[max_N, D]]
        weight_shapes = [[1, D], [1, D]]
        output_shapes = [[max_N, D]]

        block_mlir, input_ops, weight_ops, ip = self._create_mlir_importer(
            case_name, input_shapes, weight_shapes, output_shapes, ["F32"])
        in_op = input_ops[0]

        # per-position mean [N, 1] then subtract -> zero-DC input
        mean_op = top.ReduceOp(self._T(block_mlir, [max_N, 1]),
                               in_op,
                               axes=[1],
                               keepdims=True,
                               mode=StringAttr.get("ReduceMean"),
                               loc=self._L(block_mlir, "norm0.dc_mean"),
                               ip=ip).output
        x0 = top.SubOp(self._T(block_mlir, [max_N, D]), [in_op, mean_op],
                       loc=self._L(block_mlir, "norm0.dc_sub"),
                       ip=ip).output

        out = top.LayerNormOp(self._T(block_mlir, [max_N, D]),
                              x0,
                              weight_ops[0],
                              weight_ops[1],
                              normalized_shape=[D],
                              axis=1,
                              eps=1e-5,
                              loc=self._L(block_mlir, "norm0"),
                              ip=ip).output

        block_mlir.create_return_op([out])

        actual_input_shapes = [[test_N, D]]
        try:
            real = np.load('/tmp/la_dump_tf/vit.npz')['pos_emb_add'][:test_N]
            if real.shape[0] < test_N:
                raise KeyError
        except Exception:
            real = (np.random.randn(test_N, D) * 1.0).astype(np.float32)
        input_descs = [self.Desc('float32', -10, 10)]
        weight_descs = [self.Desc('float32', -10, 10), self.Desc('float32', -10, 10)]
        self._save_mlir_and_data(case_name,
                                 block_mlir,
                                 actual_input_shapes,
                                 weight_shapes,
                                 input_descs=input_descs,
                                 weight_descs=weight_descs)
        np.savez(f"{case_name}_input.npz", in0=real)
        try:
            import glob
            from safetensors import safe_open
            Wn = Bn = None
            for sf in glob.glob(
                    '/workspace/llm/locateanything/LocateAnything-3B-AutoRound-W4A16/*.safetensors'
            ):
                with safe_open(sf, 'pt') as m:
                    for k in m.keys():
                        if k.endswith('encoder.blocks.0.norm0.weight'):
                            Wn = m.get_tensor(k).cpu().float().numpy().reshape(1, D)
                        if k.endswith('encoder.blocks.0.norm0.bias'):
                            Bn = m.get_tensor(k).cpu().float().numpy().reshape(1, D)
            if Wn is not None and Bn is not None:
                np.savez(f"{case_name}_top_f32_all_origin_weight.npz", weight0=Wn, weight1=Bn)
        except Exception as e:
            print(f"[warn] real norm0 weights: {e}")

        self._deploy_test_case(case_name, tolerance=(0.99, 0.98))

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
        S = 1024
        D = 128
        Q_HEAD = 16
        KV_HEAD = 8
        input_shapes = [
            [1, S, Q_HEAD, D],  # Q
            [1, S, KV_HEAD, D],  # K
            [1, S, KV_HEAD, D],  # V
            [1, 1, S, S],
        ]
        weight_shapes = [
            [1, 1, 1, D],  # weight0
            [1, 1, 1, D],  # weight1
            [1, 1, 1, D],  # weight2
        ]
        output_shapes = [
            [1, S, Q_HEAD, D],  # out0
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
        k_op = top.RMSNormOp(self._T(block_mlir, input_shapes[1]),
                             in1_op,
                             weight_ops[1],
                             eps=1e-6,
                             loc=self._L(block_mlir, "rmsnorm1"),
                             ip=ip).output
        v_op = top.RMSNormOp(self._T(block_mlir, input_shapes[2]),
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
                              q_head=Q_HEAD,
                              kv_head=KV_HEAD,
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

    def test_fattention_prefill(self, case_name):
        """Test case fattention prefill: Fused attention with multiple inputs/outputs."""
        QS = 1024
        KS = 1024
        D = 256
        MASK_SIZE = get_tpu_info(self.chip).npu_num * 4
        Q_HEAD = 32
        KV_HEAD = 8
        input_shapes = [
            [1, QS, Q_HEAD, D],  # Q
            [1, KS, KV_HEAD, D],  # K
            [1, KS, KV_HEAD, D],  # V
            [MASK_SIZE, MASK_SIZE],
        ]
        weight_shapes = [
            [1, 1, 1, D],  # weight0
            [1, 1, 1, D],  # weight1
            [1, 1, 1, D],  # weight2
        ]
        output_shapes = [
            [1, QS, Q_HEAD, D],  # out0
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
        k_op = top.RMSNormOp(self._T(block_mlir, input_shapes[1]),
                             in1_op,
                             weight_ops[1],
                             eps=1e-6,
                             loc=self._L(block_mlir, "rmsnorm1"),
                             ip=ip).output
        v_op = top.RMSNormOp(self._T(block_mlir, input_shapes[2]),
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
                              q_head=Q_HEAD,
                              kv_head=KV_HEAD,
                              dim=D,
                              scale=1 / (D**0.5),
                              mq=QS,
                              mk=KS,
                              keep_dims=True,
                              mask_size=MASK_SIZE,
                              loc=self._L(block_mlir, "fattention"),
                              ip=ip).output

        # Create return operation
        block_mlir.create_return_op([op])

        # Generate input data with appropriate ranges
        tril_mask = np.triu(np.ones((MASK_SIZE, MASK_SIZE), dtype=np.float32), k=1)
        inputs = {
            "in0": rand_data(input_shapes[0], 'float32', -1, 1),  # query
            "in1": rand_data(input_shapes[1], 'float32', -1, 1),  # key
            "in2": rand_data(input_shapes[2], 'float32', -1, 1),  # value
            "in3": tril_mask * (-1.0e9),  # mask
        }
        weights = {
            "weight0": rand_data(weight_shapes[0], 'float32', -1, 1),
            "weight1": rand_data(weight_shapes[1], 'float32', -1, 1),
            "weight2": rand_data(weight_shapes[2], 'float32', -1, 1),
        }

        np.savez(f"{case_name}_top_f32_all_origin_weight.npz", **weights)
        if not self.no_check:
            np.savez(f"{case_name}_input.npz", **inputs)

        mlir_txt = block_mlir.print_module()
        with open(f"{case_name}.mlir", "w") as f:
            f.write(mlir_txt)

        # Deploy for each quantization mode
        self._deploy_test_case(case_name)

    def test_fattention_decode(self, case_name):
        """Test case fattention decode: Fused attention with multiple inputs/outputs."""
        QS = 1
        KS = 8192
        D = 128
        Q_HEAD = 16
        KV_HEAD = 8
        input_shapes = [
            [1, QS, Q_HEAD, D],  # Q
            [1, KS, KV_HEAD, D],  # K
            [1, KS, KV_HEAD, D],  # V
            [QS, KS],
        ]
        weight_shapes = [
            [1, 1, 1, D],  # weight0
            [1, 1, 1, D],  # weight1
            [1, 1, 1, D],  # weight2
        ]
        output_shapes = [
            [1, QS, Q_HEAD, D],  # out0
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
        k_op = top.RMSNormOp(self._T(block_mlir, input_shapes[1]),
                             in1_op,
                             weight_ops[1],
                             eps=1e-6,
                             loc=self._L(block_mlir, "rmsnorm1"),
                             ip=ip).output
        v_op = top.RMSNormOp(self._T(block_mlir, input_shapes[2]),
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
                              q_head=Q_HEAD,
                              kv_head=KV_HEAD,
                              dim=D,
                              scale=1 / (D**0.5),
                              mq=QS,
                              mk=KS,
                              keep_dims=True,
                              loc=self._L(block_mlir, "fattention"),
                              ip=ip).output

        # Create return operation
        block_mlir.create_return_op([op])

        # Generate input data with appropriate ranges
        tril_mask = np.zeros((QS, KS), dtype=np.float32)
        inputs = {
            "in0": rand_data(input_shapes[0], 'float32', -1, 1),  # query
            "in1": rand_data(input_shapes[1], 'float32', -1, 1),  # key
            "in2": rand_data(input_shapes[2], 'float32', -1, 1),  # value
            "in3": tril_mask,  # mask
        }
        weights = {
            "weight0": rand_data(weight_shapes[0], 'float32', -1, 1),
            "weight1": rand_data(weight_shapes[1], 'float32', -1, 1),
            "weight2": rand_data(weight_shapes[2], 'float32', -1, 1),
        }

        np.savez(f"{case_name}_top_f32_all_origin_weight.npz", **weights)
        if not self.no_check:
            np.savez(f"{case_name}_input.npz", **inputs)

        mlir_txt = block_mlir.print_module()
        with open(f"{case_name}.mlir", "w") as f:
            f.write(mlir_txt)

        # Deploy for each quantization mode
        self._deploy_test_case(case_name)

    def test_fattn_o_proj(self, case_name):
        """FAttention (keep_dims=false) -> Fp8MatMul o_proj, Qwen3 block_0-like shapes."""
        S, D = 512, 128
        Q_HEAD, KV_HEAD = 16, 8
        block_size = 128
        W_N, W_K = 1024, 2048

        input_shapes = [
            [1, S, Q_HEAD, D],
            [1, S, KV_HEAD, D],
            [1, S, KV_HEAD, D],
            [1, 1, S, S],
        ]
        output_shapes = [
            [1, S, 2048],
            [1, S, 1024],
        ]

        block_mlir, input_ops, _, ip = self._create_mlir_importer(case_name, input_shapes, [],
                                                                  output_shapes)
        q, k, v, mask = input_ops
        w_op = block_mlir.create_weight_op("fp8_weight", [W_N, W_K], "F8E4M3")
        s_op = block_mlir.create_weight_op("fp8_scale", [W_N // block_size, W_K // block_size],
                                           "F32")

        fattn = top.FAttentionOp(self._T(block_mlir, output_shapes[0]),
                                 q,
                                 k,
                                 v,
                                 mask,
                                 block_mlir.none_op,
                                 batch=1,
                                 q_head=Q_HEAD,
                                 kv_head=KV_HEAD,
                                 dim=D,
                                 scale=1.0 / (D**0.5),
                                 mq=S,
                                 mk=S,
                                 keep_dims=False,
                                 loc=self._L(block_mlir, "fattention"),
                                 ip=ip).output
        o_proj = top.Fp8MatMulOp(self._T(block_mlir, output_shapes[1]),
                                 fattn,
                                 w_op,
                                 s_op,
                                 block_mlir.none_op,
                                 weight_transpose=True,
                                 block_size=block_size,
                                 loc=self._L(block_mlir, "o_proj"),
                                 ip=ip).output
        block_mlir.create_return_op([fattn, o_proj])

        inputs = {
            "in0": rand_data(input_shapes[0], "float32", -1, 1),
            "in1": rand_data(input_shapes[1], "float32", -1, 1),
            "in2": rand_data(input_shapes[2], "float32", -1, 1),
            "in3": self._causal_attention_mask(S),
        }
        weights = self._default_fp8_matmul_weights(W_N, W_K, block_size)
        # if self.weights_path:
        #     weights.update(self._load_fp8_matmul_weights(self.weights_path, W_N, W_K, block_size))
        #     print(f"[fattn_o_proj] fp8 weights from {self.weights_path}")

        with open(f"{case_name}.mlir", "w") as f:
            f.write(block_mlir.print_module())
        np.savez(f"{case_name}_top_f32_all_origin_weight.npz", **weights)
        if not self.no_check:
            np.savez(f"{case_name}_input.npz", **inputs)

        saved_modes = self.quant_modes
        self.quant_modes = [m for m in saved_modes if m in ("f16", "bf16")]
        self._deploy_test_case(case_name, tolerance=(0.99, 0.90))
        self.quant_modes = saved_modes

    @staticmethod
    def _causal_attention_mask(seq_len: int) -> np.ndarray:
        mask = np.triu(np.full((seq_len, seq_len), -1.0e9, dtype=np.float32), k=1)
        return mask.reshape(1, 1, seq_len, seq_len)

    @staticmethod
    def _default_fp8_matmul_weights(w_n: int, w_k: int, block_size: int) -> dict:
        return {
            "fp8_weight": np.random.randint(0, 256, (w_n, w_k), dtype=np.uint8),
            "fp8_scale": rand_data([w_n // block_size, w_k // block_size], "float32", 0, 0.1),
        }

    # def _load_fp8_matmul_weights(
    #         self,
    #         npz_path: str,
    #         w_n: int,
    #         w_k: int,
    #         block_size: int,
    #         weight_key: str = "model.layers.0.self_attn.o_proj.weight",
    #         scale_key: str = "model.layers.0.self_attn.o_proj.weight_scale_inv") -> dict:
    #     if not os.path.isfile(npz_path):
    #         raise FileNotFoundError(f"weights npz not found: {npz_path}")
    #     wnpz = np.load(npz_path)
    #     try:
    #         if weight_key not in wnpz.files:
    #             raise KeyError(f"{weight_key} not in {npz_path}, keys={wnpz.files[:8]}...")
    #         if scale_key not in wnpz.files:
    #             raise KeyError(f"{scale_key} not in {npz_path}")
    #         weight = wnpz[weight_key]
    #         scale = wnpz[scale_key]
    #     finally:
    #         wnpz.close()
    #     if tuple(weight.shape) != (w_n, w_k):
    #         raise ValueError(f"weight shape {weight.shape} != ({w_n}, {w_k})")
    #     scale_shape = (w_n // block_size, w_k // block_size)
    #     if tuple(scale.shape) != scale_shape:
    #         raise ValueError(f"scale shape {scale.shape} != {scale_shape}")
    #     return {"fp8_weight": weight, "fp8_scale": scale}

    def test_fp8matmul(self, case_name):
        """Standalone Fp8MatMul (Qwen3 o_proj-like: 1x512x2048 -> 1x512x1024)."""
        S = 512
        W_N, W_K = 1024, 2048
        block_size = 128

        input_shapes = [[1, S, W_K]]
        output_shapes = [[1, S, W_N]]

        block_mlir, input_ops, _, ip = self._create_mlir_importer(case_name, input_shapes, [],
                                                                  output_shapes)
        in0 = input_ops[0]
        w_op = block_mlir.create_weight_op("fp8_weight", [W_N, W_K], "F8E4M3")
        s_op = block_mlir.create_weight_op("fp8_scale", [W_N // block_size, W_K // block_size],
                                           "F32")
        out = top.Fp8MatMulOp(self._T(block_mlir, output_shapes[0]),
                              in0,
                              w_op,
                              s_op,
                              block_mlir.none_op,
                              weight_transpose=True,
                              block_size=block_size,
                              loc=self._L(block_mlir, "fp8matmul"),
                              ip=ip).output
        block_mlir.create_return_op([out])

        weights = self._default_fp8_matmul_weights(W_N, W_K, block_size)
        # if self.weights_path:
        #     weights.update(self._load_fp8_matmul_weights(self.weights_path, W_N, W_K, block_size))
        #     print(f"[fp8matmul] fp8 weights from {self.weights_path}")

        act = rand_data(input_shapes[0], "float32", -1, 1)
        if self.weights_path:
            act = np.load(
                "/workspace/tpu-mlir/mlir_test_bm1684x2/fattn_o_proj_bm1684x2_bf16_model_outputs.npz"
            )["fattention_f32"]
        inputs = {"in0": act}

        with open(f"{case_name}.mlir", "w") as f:
            f.write(block_mlir.print_module())
        np.savez(f"{case_name}_top_f32_all_origin_weight.npz", **weights)
        if not self.no_check:
            np.savez(f"{case_name}_input.npz", **inputs)

        saved_modes = self.quant_modes
        self.quant_modes = [m for m in saved_modes if m in ("f16", "bf16")]
        self._deploy_test_case(case_name, tolerance=(0.99, 0.90))
        self.quant_modes = saved_modes

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

    def test_a16gather(self, case_name):
        """Test case A16Gather: Simple A16Gather operation for embedding lookup."""
        vocab_size = 262144
        dim = 256
        batch = 1
        seq_len = 512
        q_group_size = 32
        weight_bits = 8
        keepdims = False

        # For weight_bits=4, weight is packed: [vocab_size, dim // (8 // weight_bits)] = [vocab_size, dim // 2]
        weight_packed_dim = dim // (8 // weight_bits)
        # Scale/zp shape: [vocab_size, dim // q_group_size]
        n_groups = dim // q_group_size

        # Output shape depends on keepdims:
        #   keepdims=False: [batch, seq_len, dim]
        #   keepdims=True:  [batch, seq_len, 1, dim]
        if keepdims:
            gather_out_shape = [batch, seq_len, 1, dim]
        else:
            gather_out_shape = [batch, seq_len, dim]

        input_shapes = [
            [batch, seq_len],  # indices
        ]
        output_shapes = [
            gather_out_shape,
        ]

        weight_shape = [vocab_size, weight_packed_dim]
        scale_shape = [vocab_size, n_groups]
        zp_shape = [vocab_size, n_groups]

        input_types = ["F32"]
        block_mlir = MLIRImporter(input_shapes, output_shapes, case_name, Platform.LLM, input_types)
        input_ops = self._create_input_ops(block_mlir, input_shapes)
        ip = block_mlir.insert_point

        indices_op = input_ops[0]

        # Create weight ops with appropriate types
        qweight_op = block_mlir.create_weight_op("qweight", weight_shape, "UINT8")
        scale_op = block_mlir.create_weight_op("scales", scale_shape, "F32")
        zp_op = block_mlir.create_weight_op("qzeros", zp_shape, "UINT8")

        # A16Gather operation
        out = top.A16GatherOp(self._T(block_mlir, gather_out_shape),
                              qweight_op,
                              indices_op,
                              scale_op,
                              zp_op,
                              axis=0,
                              keepdims=keepdims,
                              q_group_size=q_group_size,
                              weight_bits=weight_bits,
                              loc=self._L(block_mlir, "a16gather0"),
                              ip=ip).output

        # Create return operation
        block_mlir.create_return_op([out])

        # Generate data
        # Indices should be in range [0, vocab_size)
        indices_data = np.random.randint(0, vocab_size, size=input_shapes[0], dtype=np.int32)
        inputs = {
            "in0": indices_data,
        }
        weights = {
            "qweight": rand_data(weight_shape, 'uint8', 0, 255, int_satu=True),
            "scales": rand_data(scale_shape, 'float32', 0.5, 2.0),
            "qzeros": rand_data(zp_shape, 'uint8', 0, 15,
                                int_satu=True),  # For 4-bit, zeros should be in [0, 15]
        }

        # Save MLIR text
        mlir_txt = block_mlir.print_module()
        with open(f"{case_name}.mlir", "w") as f:
            f.write(mlir_txt)

        # Save weights and inputs
        np.savez(f"{case_name}_top_f32_all_origin_weight.npz", **weights)
        if not self.no_check:
            np.savez(f"{case_name}_input.npz", **inputs)

        # Deploy for each quantization mode (A16Gather only supports F16/BF16 lowering)
        saved_modes = self.quant_modes
        self.quant_modes = [m for m in self.quant_modes if m in ["f16", "bf16"]]
        self._deploy_test_case(case_name, tolerance=(0.95, 0.90))
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

    def test_softplus_mul(self, case_name):
        """Test case: Softplus followed by Mul with a weight tensor.

        Reproduces the pattern from block_0.mlir:
          %33 = top.Softplus(%18)
          %34 = top.Weight() -> tensor<1x1x16xf32>
          %35 = top.Mul(%33, %34)
        """
        input_shapes = [
            [1, 65536, 16],  # input activation (matches in_proj_a output)
        ]
        weight_shapes = [
            [1, 1, 16],  # A_log weight (broadcast multiplier)
        ]
        output_shapes = [
            [1, 65536, 16],  # output after softplus+mul
        ]

        # Create MLIR importer
        block_mlir, input_ops, weight_ops, ip = self._create_mlir_importer(
            case_name, input_shapes, weight_shapes, output_shapes, ["F32"])

        in0_op = input_ops[0]

        # Softplus
        softplus_out = top.SoftplusOp(self._T(block_mlir, input_shapes[0]),
                                      in0_op,
                                      loc=self._L(block_mlir, "softplus"),
                                      ip=ip).output

        # Mul with weight
        mul_out = top.MulOp(self._T(block_mlir, output_shapes[0]), [softplus_out, weight_ops[0]],
                            loc=self._L(block_mlir, "mul"),
                            ip=ip).output

        # Create return operation
        block_mlir.create_return_op([mul_out])

        # Save MLIR text, weights, and inputs
        self._save_mlir_and_data(case_name, block_mlir, input_shapes, weight_shapes)

        # Deploy for each quantization mode
        self._deploy_test_case(case_name)

    def test_softmax_topk(self, case_name):
        """Test case: SoftMax followed by TopK with use_hau=False."""
        input_shapes = [
            [1, 1, 256],
        ]
        weight_shapes = []
        output_shapes = [
            [1, 1, 8],  # TopK values
            [1, 1, 8],  # TopK indices
        ]

        # Create MLIR importer
        block_mlir, input_ops, weight_ops, ip = self._create_mlir_importer(
            case_name, input_shapes, weight_shapes, output_shapes, ["F32"])

        in0_op = input_ops[0]

        # SoftMax
        softmax_out = top.SoftmaxOp(self._T(block_mlir, input_shapes[0]),
                                    in0_op,
                                    axis=2,
                                    loc=self._L(block_mlir, "softmax"),
                                    ip=ip).output

        # TopK with use_hau=False
        topk_op = top.TopKOp(self._T(block_mlir, output_shapes[0]),
                             self._T(block_mlir, output_shapes[0]),
                             softmax_out,
                             axis=2,
                             K=8,
                             use_hau=False,
                             loc=self._L(block_mlir, ["topk_values", "topk_indices"]),
                             ip=ip)

        # Create return operation
        block_mlir.create_return_op([topk_op.values, topk_op.indices])

        # Save MLIR text and inputs
        self._save_mlir_and_data(case_name,
                                 block_mlir,
                                 input_shapes,
                                 weight_shapes,
                                 input_descs=[self.Desc('float32', -5, 5)])

        # Deploy for each quantization mode
        self._deploy_test_case(case_name)

    def test_conv2d_non_overlapping(self, case_name):
        """Test case: Conv2d with kernel_size == stride (non-overlapping patches).

        Reproduces the patch embedding Conv2d:
          Input:  [1, 3, 14, num_patches*14]  (NaViT pixel values)
          Weight: [1152, 3, 14, 14]           (Conv2d weight)
          Bias:   [1152]                       (Conv2d bias)
          Output: [1, 1152, 1, num_patches]    (patch embeddings)

        kernel=[14,14], stride=[14,14], group=1, pads=[0,0,0,0]

        With --dynamic:
          - MLIR compiled with max_patches=4624 (max shape)
          - Test input uses test_patches=1008 (smaller shape)
          - This tests if dynamic shape kernel handles smaller inputs correctly
        """
        in_channels = 3
        out_channels = 1152
        patch_size = 14

        if self.dynamic:
            # Dynamic test: compile with max shape, test with smaller shape
            max_patches = 4624  # max shape for compilation
            test_patches = 1008  # smaller shape for testing
        else:
            # Static test: compile and test with same shape
            max_patches = 1008
            test_patches = 1008

        # MLIR shapes (max shape for compilation)
        input_shapes = [
            [1, in_channels, patch_size, max_patches * patch_size],
        ]
        weight_shapes = [
            [out_channels, in_channels, patch_size, patch_size],
            [out_channels],
        ]
        output_shapes = [
            [1, out_channels, 1, max_patches],
        ]

        block_mlir, input_ops, weight_ops, ip = self._create_mlir_importer(
            case_name, input_shapes, weight_shapes, output_shapes, ["F32"])

        in0_op = input_ops[0]

        # Conv2d patch embedding
        conv_out = top.ConvOp(self._T(block_mlir, output_shapes[0]),
                              in0_op,
                              weight_ops[0],
                              weight_ops[1],
                              kernel_shape=[patch_size, patch_size],
                              strides=[patch_size, patch_size],
                              pads=[0, 0, 0, 0],
                              dilations=[1, 1],
                              loc=self._L(block_mlir, "patch_embedding"),
                              ip=ip).output

        block_mlir.create_return_op([conv_out])

        # Generate test data with test_patches (may be smaller than max_patches)
        actual_input_shapes = [
            [1, in_channels, patch_size, test_patches * patch_size],
        ]
        self._save_mlir_and_data(
            case_name,
            block_mlir,
            actual_input_shapes,  # use smaller shape for input data
            weight_shapes,
            input_descs=[self.Desc('float32', -1, 1)])

        self._deploy_test_case(case_name, tolerance=(0.99, 0.98))

    def test_matmul_reshape_permute(self, case_name):
        """Test case: Reshape + Permute + MatMul equivalence to Conv2d (kernel==stride, no padding).

        When kernel==stride and no padding, Conv2d is equivalent to:
          1. Reshape to extract non-overlapping patches
          2. Permute patches to first dim
          3. Flatten each patch to a vector
          4. MatMul with weight matrix
          5. Add bias

        This tests if MatMul-based approach supports dynamic num_patches.

        Conv2d params: in=3, out=1152, kernel=14, stride=14, pad=0, group=1
        """
        in_channels = 3
        out_channels = 1152
        patch_size = 14

        if self.dynamic:
            max_patches = 4624
            test_patches = 1008
        else:
            max_patches = 1008
            test_patches = 1008

        patch_flat = in_channels * patch_size * patch_size  # 3*14*14 = 588

        # MLIR shapes (max shape)
        input_shapes = [
            [1, in_channels, patch_size, max_patches * patch_size],
        ]
        weight_shapes = [
            [patch_flat, out_channels],  # [588, 1152]
            [1, out_channels],  # bias [1, 1152]
        ]
        output_shapes = [
            [1, out_channels, 1, max_patches],
        ]

        block_mlir, input_ops, weight_ops, ip = self._create_mlir_importer(
            case_name, input_shapes, weight_shapes, output_shapes, ["F32"])

        in0_op = input_ops[0]

        # 1. Reshape [1,3,14,P*14] -> [1,3,14,P,14]
        reshape1 = top.ReshapeOp(self._T(block_mlir,
                                         [1, in_channels, patch_size, max_patches, patch_size]),
                                 in0_op,
                                 shape=[1, in_channels, patch_size, -1, patch_size],
                                 loc=self._L(block_mlir, "reshape_split"),
                                 ip=ip).output

        # 2. Permute [1,3,14,P,14] -> [1,P,3,14,14]
        permute1 = top.PermuteOp(self._T(block_mlir,
                                         [1, max_patches, in_channels, patch_size, patch_size]),
                                 reshape1,
                                 order=[0, 3, 1, 2, 4],
                                 loc=self._L(block_mlir, "permute"),
                                 ip=ip).output

        # 3. Reshape [1,P,3,14,14] -> [-1,588]
        reshape2 = top.ReshapeOp(self._T(block_mlir, [max_patches, patch_flat]),
                                 permute1,
                                 shape=[-1, patch_flat],
                                 loc=self._L(block_mlir, "reshape_flatten"),
                                 ip=ip).output

        # 4. MatMul [P,588] @ [588,1152] + bias [1,1152] -> [P,1152]
        matmul_out = top.MatMulOp(self._T(block_mlir, [max_patches, out_channels]),
                                  reshape2,
                                  weight_ops[0],
                                  weight_ops[1],
                                  loc=self._L(block_mlir, "matmul_embed"),
                                  ip=ip).output

        # 5. Reshape [P,1152] -> [1,1152,1,-1]
        reshape3 = top.ReshapeOp(self._T(block_mlir, output_shapes[0]),
                                 matmul_out,
                                 shape=[1, out_channels, 1, -1],
                                 loc=self._L(block_mlir, "reshape_output"),
                                 ip=ip).output

        block_mlir.create_return_op([reshape3])

        # Generate test data with test_patches
        actual_input_shapes = [
            [1, in_channels, patch_size, test_patches * patch_size],
        ]
        self._save_mlir_and_data(
            case_name,
            block_mlir,
            actual_input_shapes,
            weight_shapes,
            input_descs=[self.Desc('float32', -1, 1)],
            weight_descs=[self.Desc('float32', -0.1, 0.1),
                          self.Desc('float32', -0.1, 0.1)])

        self._deploy_test_case(case_name, tolerance=(0.99, 0.98))

    def test_matmul_dynamic(self, case_name):
        """Test: bare MatMul with dynamic batch dim.

        [P, 588] @ [588, 1152] + bias [1, 1152] -> [P, 1152]
        With --dynamic: compile max_P=4624, test P=1008.
        """
        K = 588
        N = 1152
        max_P = 4624
        test_P = 1008 if self.dynamic else max_P

        input_shapes = [[max_P, K]]
        weight_shapes = [[K, N], [1, N]]
        output_shapes = [[max_P, N]]

        block_mlir, input_ops, weight_ops, ip = self._create_mlir_importer(
            case_name, input_shapes, weight_shapes, output_shapes, ["F32"])

        out = top.MatMulOp(self._T(block_mlir, [max_P, N]),
                           input_ops[0],
                           weight_ops[0],
                           weight_ops[1],
                           loc=self._L(block_mlir, "matmul"),
                           ip=ip).output

        block_mlir.create_return_op([out])

        actual_input_shapes = [[test_P, K]]
        self._save_mlir_and_data(
            case_name,
            block_mlir,
            actual_input_shapes,
            weight_shapes,
            input_descs=[self.Desc('float32', -1, 1)],
            weight_descs=[self.Desc('float32', -0.1, 0.1),
                          self.Desc('float32', -0.1, 0.1)])

        self._deploy_test_case(case_name, tolerance=(0.99, 0.98))

    def test_reshape_dynamic(self, case_name):
        """Test: single Reshape with dynamic last dim.

        [1, 3, 14, P*14] -> [1, -1, 588]  (flatten patches)
        With --dynamic: compile max_P=4624, test P=1008.
        """
        C, H, PS = 3, 14, 14
        max_P = 4624
        test_P = 1008 if self.dynamic else max_P

        input_shapes = [[1, C, H, max_P * PS]]
        output_shapes = [[1, max_P, C * H * PS]]

        block_mlir, input_ops, weight_ops, ip = self._create_mlir_importer(
            case_name, input_shapes, [], output_shapes, ["F32"])

        # Reshape [1,3,14,P*14] -> [1,-1,588]
        r1 = top.ReshapeOp(self._T(block_mlir, output_shapes[0]),
                           input_ops[0],
                           shape=[1, -1, C * H * PS],
                           loc=self._L(block_mlir, "reshape"),
                           ip=ip).output

        block_mlir.create_return_op([r1])

        actual_input_shapes = [[1, C, H, test_P * PS]]
        self._save_mlir_and_data(case_name,
                                 block_mlir,
                                 actual_input_shapes, [],
                                 input_descs=[self.Desc('float32', -1, 1)])

        self._deploy_test_case(case_name, tolerance=(0.99, 0.98))

    def test_permute_dynamic(self, case_name):
        """Test: 4D Permute with dynamic dim.

        [1, 16, 128, S] -> [1, S, 16, 128]  order=[0,3,1,2]
        With --dynamic: compile max_S=2048, test S=512.
        (Parameters from working model's Permute)
        """
        max_S = 10
        test_S = 5 if self.dynamic else max_S

        input_shapes = [[1, 8, 4, max_S]]
        output_shapes = [[1, max_S, 8, 4]]

        block_mlir, input_ops, weight_ops, ip = self._create_mlir_importer(
            case_name, input_shapes, [], output_shapes, ["F32"])

        p1 = top.PermuteOp(self._T(block_mlir, output_shapes[0]),
                           input_ops[0],
                           order=[0, 3, 1, 2],
                           loc=self._L(block_mlir, "permute"),
                           ip=ip).output

        block_mlir.create_return_op([p1])

        actual_input_shapes = [[1, 8, 4, test_S]]
        self._save_mlir_and_data(case_name,
                                 block_mlir,
                                 actual_input_shapes, [],
                                 input_descs=[self.Desc('float32', -1, 1)])

        self._deploy_test_case(case_name, tolerance=(0.99, 0.98))


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
    parser.add_argument("--mode", default="bf16", type=str, choices=['all', 'f32', 'f16', 'bf16'], help="quantize modes, only supports fp for now")
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
    parser.add_argument("--disable_hp", action="store_true", help='disable high precision')
    parser.add_argument("--ip", default="", type=str,
                        help="remote server as 'username@ip' to run bmodel inference")
    parser.add_argument("--pwd", default="", type=str,
                        help="remote server password for ssh/scp")
    parser.add_argument(
        "--weights_path",
        default="",
        type=str,
        help="Optional npz for fp8matmul/fattn_o_proj: o_proj weight/scale; "
        "optional activation keys fattention_in / model.layers.0.fattention",
    )
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
