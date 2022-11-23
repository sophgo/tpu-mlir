//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/CV18xx/CV18xx.h"
#include "tpu_mlir/Backend/CV18xx/CV18xx_global_api.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/Helper/Quant.h"

using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace tpu_mlir::backend;

// =========================================
// GlobalGenInterface
// =========================================
void tpu::ReduceOp::codegen_global_cv18xx(int64_t layer_id) {
  gaddr_t ga_input = Module::getAddress(input());
  gaddr_t ga_output = Module::getAddress(output());
  std::vector<int64_t> input_shape;
  std::vector<int32_t> axes_v;
  auto mode = type().str();
  auto axes_val = Module::getI64Array(axes());
  axes_v.assign(axes_val->begin(), axes_val->end());
  Module::getShapeVec(input(), input_shape);
  if (mode == "ReduceL2") {
    gaddr_t ga_table = Module::getAddress(buffer());
    gaddr_t ga_mantissa_table = Module::getAddress(reciprocal_mantissa_table());
    cvi_backend_tg_bf16_reduce_l2_kernel(layer_id, ga_input, ga_output,
                                         ga_table, ga_mantissa_table,
                                         input_shape, axes_v);
    return;
  }

  if (Quant::isUniformQuantized(output())) {
    int32_t shift =
        static_cast<int32_t>(Module::getI64Array(rshift().value())->at(0));
    int32_t multi =
        static_cast<int32_t>(Module::getI64Array(multiplier().value())->at(0));
    if (mode == "ReduceMean") {
      cvi_backend_tg_fixed_reduce_mean_kernel(
          layer_id, ga_input, ga_output, input_shape, axes_v, multi, shift);
    } else if (mode == "ReduceSum") {
      cvi_backend_tg_fixed_reduce_sum_kernel(layer_id, ga_input, ga_output,
                                             input_shape, axes_v, multi, shift);
    } else if (mode == "ReduceMax") {
      cvi_backend_tg_fixed_reduce_max_kernel(layer_id, ga_input, ga_output,
                                             input_shape, axes_v);
    } else if (mode == "ReduceMin") {
      cvi_backend_tg_fixed_reduce_min_kernel(layer_id, ga_input, ga_output,
                                             input_shape, axes_v, multi, shift);
    } else {
      llvm_unreachable("unsupport reduce type.");
    }

  } else {
    if (mode == "ReduceMean") {
      cvi_backend_tg_bf16_reduce_mean_kernel(layer_id, ga_input, ga_output,
                                             input_shape, axes_v);
    } else if (mode == "ReduceSum") {
      cvi_backend_tg_bf16_reduce_sum_kernel(layer_id, ga_input, ga_output,
                                            input_shape, axes_v);
    } else if (mode == "ReduceMax") {
      cvi_backend_tg_bf16_reduce_max_kernel(layer_id, ga_input, ga_output,
                                            input_shape, axes_v);
    } else if (mode == "ReduceMin") {
      cvi_backend_tg_bf16_reduce_min_kernel(layer_id, ga_input, ga_output,
                                            input_shape, axes_v);
    } else {
      llvm_unreachable("unsupport reduce type.");
    }
  }
}
