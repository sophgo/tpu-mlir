//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace mlir;

LogicalResult tpu::MulShiftOp::init(InferenceParameter &p) { return success(); }
void tpu::MulShiftOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::MulShiftOp::inference(InferenceParameter &p) {
  auto num_elem = Module::getNumElements(output());
  auto sType = Module::getStorageType(output());
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
  for (int64_t i = 0; i < num_elem; i++) {
    auto v = applyMultiplierAndRShift(p.inputs[0][i], multiplier(), rshift());
    p.outputs[0][i] =
        sType.isUnsignedInteger(8) ? Quant::to_uint8(v) : Quant::to_int8(v);
  }
  return success();
}
