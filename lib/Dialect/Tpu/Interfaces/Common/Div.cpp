//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace mlir;

LogicalResult tpu::DivOp::init(InferenceParameter &p) {
  return success();
}

void tpu::DivOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::DivOp::inference(InferenceParameter &p) {
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
    auto limit = relu_limit().convertToDouble();
    function_relu(p.outputs[0], p.outputs[0], num_elem, limit);
  }
  return success();
}
