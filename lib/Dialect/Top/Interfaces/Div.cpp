//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Support/MathUtils.h"



int64_t top::DivOp::getFLOPs() {
  return module::getNumElements(output());
}

LogicalResult top::DivOp::init(InferenceParameter &p) { return success(); }
void top::DivOp::deinit(InferenceParameter &p) {}

LogicalResult top::DivOp::inference(InferenceParameter &p) {
  auto num_elem = module::getNumElements(output());
  int lhs_num_elem = module::getNumElements(inputs()[0]);
  int rhs_num_elem = module::getNumElements(inputs()[1]);
  int c_loop = lhs_num_elem / rhs_num_elem;
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
  for (int64_t i = 0; i < num_elem; i++) {
    try {
      //broadcast
      if (p.inputs[1][(c_loop == 1) ? i : i % rhs_num_elem] == 0) {
        throw "division by zero";
      }
      p.outputs[0][i] = p.inputs[0][i] / p.inputs[1][(c_loop == 1) ? i : i % rhs_num_elem];
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
