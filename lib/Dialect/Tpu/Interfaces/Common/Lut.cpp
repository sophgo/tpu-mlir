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

LogicalResult tpu::LutOp::init(InferenceParameter &p) { return success(); }
void tpu::LutOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::LutOp::inference(InferenceParameter &p) {
  auto num_element = Module::getNumElements(input());
  if (Quant::isUniformQuantized(input(), output())) {
    auto stype = Module::getStorageType(input());
#pragma omp parallel for schedule(static, omp_schedule(num_element))
    for (int i = 0; i < num_element; ++i) {
      int offset = p.inputs[0][i];
      if (offset < 0) {
        offset += 256;
      }
      assert(offset >= 0 && offset <= 255);
      p.outputs[0][i] = p.inputs[1][offset];
    }
  } else {
    llvm_unreachable("not support");
  }
  return success();
}
