//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

permute_attr_t tpu::ShapeTransposeOp::parseParam() {
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
    ret = permute_reset(in_shape, *in_order, attr.in_shape_fix, attr.order_fix,
                        6);
  }
  if (ret == false) {
    UNREACHABLE_THIS("Not Implemented");
  }
  for (auto o : attr.order_fix) {
    attr.out_shape_fix.push_back(attr.in_shape_fix[o]);
  }
  return attr;
}
LogicalResult tpu::ShapeTransposeOp::init(InferenceParameter &p) {
  permute_attr_t *attr = new permute_attr_t;
  auto param = parseParam();
  attr->in_shape_fix = param.in_shape_fix;
  attr->order_fix = param.order_fix;
  attr->out_shape_fix = param.out_shape_fix;
  p.handle = (void *)attr;
  return success();
}
void tpu::ShapeTransposeOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::ShapeTransposeOp::inference(InferenceParameter &p) {
  auto p_info = (permute_attr_t *)p.handle;
  function_permute(p.inputs[0], p.outputs[0], p_info->in_shape_fix,
                   p_info->order_fix);
  return success();
}

bool tpu::ShapeTransposeOp::support_multi_core() { return false; }