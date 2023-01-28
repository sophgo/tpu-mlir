//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Module.h"

int64_t top::PermuteOp::getFLOPs() { return 0; }

struct order_info {
  std::vector<int64_t> shape;
  std::vector<int64_t> order;
};

LogicalResult top::PermuteOp::init(InferenceParameter &p) {
  auto info = new order_info();
  std::vector<int64_t> in_shape = module::getShape(getInput());
  i64_array_t in_order = module::getI64Array(getOrder());
  auto ret = permute_reorder(in_shape, *in_order, info->shape, info->order, 5);
  if (ret == false) {
    dump();
    llvm_unreachable("Not Implemented");
  }
  p.handle = (void *)info;
  return success();
}

void top::PermuteOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto info = (order_info *)p.handle;
    delete info;
    p.handle = nullptr;
  }
}

LogicalResult top::PermuteOp::inference(InferenceParameter &p) {
  auto p_info = (order_info *)p.handle;
  function_permute(p.inputs[0], p.outputs[0], p_info->shape, p_info->order);
  return success();
}
