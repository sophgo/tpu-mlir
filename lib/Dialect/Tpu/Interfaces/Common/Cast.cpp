//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "sophgo/Dialect/Tpu/IR/TpuOps.h"
#include "sophgo/Support/Dnnl/Dnnl.h"
#include "sophgo/Support/Helper/Quant.h"
#include "sophgo/Support/Helper/Module.h"

using namespace sophgo;
using namespace sophgo::helper;
using namespace mlir;

LogicalResult tpu::CastOp::init(InferenceParameter &p) { return success(); }
void tpu::CastOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::CastOp::inference(InferenceParameter &p) {
  auto num_elem = Module::getNumElements(output());
  auto in_type = Module::getStorageType(input());
  auto out_type = Module::getStorageType(output());
  if (in_type.isF32() && out_type.isF32()) {
    llvm_unreachable("shouldn't be exist");
  } else {
    if (Quant::isUniformQuantized(output())) {
      auto qtype = Quant::getUniformQuantizedType(output());
  #pragma omp parallel for schedule(static, omp_schedule(num_elem))
      for (size_t i = 0; i < num_elem; i++) {
        auto v = p.inputs[0][i] / qtype.getScale() + qtype.getZeroPoint();
        p.outputs[0][i] = Quant::to_int8(v);
      }
    } else if (Quant::isUniformQuantized(input())) {
      auto qtype = Quant::getUniformQuantizedType(input());
  #pragma omp parallel for schedule(static, omp_schedule(num_elem))
      for (size_t i = 0; i < num_elem; i++) {
        p.outputs[0][i] =
            qtype.getScale() * (p.inputs[0][i] - qtype.getZeroPoint());
      }
    }
  }

  return success();
}
