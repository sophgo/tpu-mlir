//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace mlir;

int64_t top::DivOp::getFLOPs() {
  return Module::getNumElements(output());
}

LogicalResult top::DivOp::init(InferenceParameter &p) { return success(); }
void top::DivOp::deinit(InferenceParameter &p) {}

LogicalResult top::DivOp::inference(InferenceParameter &p) {
  auto num_elem = Module::getNumElements(output());
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
  for (int64_t i = 0; i < num_elem; i++) {
    try {
      if (p.inputs[1][i] == 0) {
        throw "division by zero";
      }
      p.outputs[0][i] = p.inputs[0][i] / p.inputs[1][i];
    }catch (const char* msg) {
     std::cerr << msg << std::endl;
    }
  }
  if (do_relu()) {
    function_relu(p.outputs[0], p.outputs[0], num_elem);
  }
  return success();
}
