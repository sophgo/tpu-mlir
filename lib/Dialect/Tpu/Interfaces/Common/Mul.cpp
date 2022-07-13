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
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace mlir;

LogicalResult tpu::MulOp::init(InferenceParameter &p) {
  return success();
}

void tpu::MulOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::MulOp::inference(InferenceParameter &p) {
  int nInputs = inputs().size();
  auto num_elem = Module::getNumElements(output());
  auto out_type = Module::getStorageType(output());
  std::fill(p.outputs[0], p.outputs[0] + num_elem, 1.0f);
  if (out_type.isa<FloatType>()) {
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
    for (int64_t j = 0; j < num_elem; j++) {
      for (int i = 0; i < nInputs; i++) {
          p.outputs[0][j] *= p.inputs[i][j];
      }
      if (do_relu()) {
        p.outputs[0][j] = std::max(0.0f, p.outputs[0][j]);
      }
    }
    if (out_type.isBF16()) {
      f32_to_bf16(p.outputs[0], p.outputs[0], num_elem);
    } else if (out_type.isF16()) {
      f32_to_f16(p.outputs[0], p.outputs[0], num_elem);
    }
  } else {
    llvm_unreachable("Codegen to be supported");
  }
  return success();
}
