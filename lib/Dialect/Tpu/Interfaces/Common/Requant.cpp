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
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace mlir;

LogicalResult tpu::RequantOp::init(InferenceParameter &p) { return success(); }
void tpu::RequantOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::RequantOp::inference(InferenceParameter &p) {
  auto o_sType = Module::getStorageType(output());
  auto o_qtype = Quant::getUniformQuantizedType(output());
  int64_t num_elem = Module::getNumElements(input());
  int64_t zp_x = 0;
  if (Quant::isUniformQuantized(input())) {
    auto i_qtype = Quant::getUniformQuantizedType(input());
    zp_x = i_qtype.getZeroPoint();
  }
  int64_t rshift_val = rshift();
  int64_t mul_val = multiplier();
  int64_t zp_y = o_qtype.getZeroPoint();
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
  for (int64_t idx = 0; idx < num_elem; idx++) {
    int32_t tmp = (int32_t)p.inputs[0][idx] - zp_x;
    auto v = applyMultiplierAndRShift(tmp, mul_val, rshift_val) + zp_y;
    p.outputs[0][idx] =
        o_sType.isUnsignedInteger(8) ? Quant::to_uint8(v) : Quant::to_int8(v);
  }
  return success();
}
