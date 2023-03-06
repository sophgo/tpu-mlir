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

permute_attr_t top::PermuteOp::parseParam() {
  permute_attr_t attr;
  std::vector<int64_t> in_shape = module::getShape(getInput());
  i64_array_t in_order = module::getI64Array(getOrder());
  auto ret =
      permute_reset(in_shape, *in_order, attr.in_shape_fix, attr.order_fix, 4);
  if (ret == false) {
    ret = permute_reset(in_shape, *in_order, attr.in_shape_fix, attr.order_fix,
                        5);
  }
  if (ret == false) {
    dump();
    llvm_unreachable("Not Implemented");
  }
  for (auto o : attr.order_fix) {
    attr.out_shape_fix.push_back(attr.in_shape_fix[o]);
  }
  return attr;
}

LogicalResult top::PermuteOp::init(InferenceParameter &p) {
  permute_attr_t *attr = new permute_attr_t;
  auto param = parseParam();
  attr->in_shape_fix = param.in_shape_fix;
  attr->order_fix = param.order_fix;
  attr->out_shape_fix = param.out_shape_fix;
  if (attr->order_fix.size() == 4) {
    // to 5 dim
    attr->in_shape_fix.push_back(1);
    attr->out_shape_fix.push_back(1);
    attr->order_fix.push_back(4);
  }
  p.handle = (void *)attr;
  return success();
}

void top::PermuteOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto info = (permute_attr_t *)p.handle;
    delete info;
    p.handle = nullptr;
  }
}

LogicalResult top::PermuteOp::inference(InferenceParameter &p) {
  auto p_info = (permute_attr_t *)p.handle;
  function_permute(p.inputs[0], p.outputs[0], p_info->in_shape_fix,
                   p_info->order_fix);
  return success();
}

void top::PermuteOp::shape_inference() {}
