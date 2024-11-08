//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

int64_t top::PermuteOp::getFLOPs() { return 0; }

permute_attr_t top::PermuteOp::parseParam() {
  permute_attr_t attr;
  std::vector<int64_t> in_shape = module::getShape(getInput());
  i64_array_t in_order = module::getI64Array(getOrder());
  if (in_order->size() == 0) {
    // default revert it, eg: shape (2, 3, 4)->(4, 3, 2), per=[2, 1, 0]
    std::vector<int64_t> order;
    for (uint32_t i = in_shape.size() - 1; i >= 0; i--) {
      order.push_back(i);
    }
    auto builder = OpBuilder(getContext());
    setOrderAttr(builder.getI64ArrayAttr(order));
    in_order = module::getI64Array(getOrder());
  }
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
    dump();
    UNREACHABLE_THIS("Not Implemented");
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
  i64_array_t in_order = module::getI64Array(getOrder());
  auto in_shape = module::getShape(getInput());
  std::vector<int64_t> out_shape;
  for (int64_t i = 0; i < in_shape.size(); ++i) {
    out_shape.push_back(in_shape[(*in_order)[i]]);
  }
  module::setShape(getOutput(), out_shape);
  return success();
}

void top::PermuteOp::shape_inference() {
  i64_array_t in_order = module::getI64Array(getOrder());
  auto in_shape = module::getShape(getInput());
  std::vector<int64_t> out_shape;
  for (int64_t i = 0; i < in_shape.size(); ++i) {
    out_shape.push_back(in_shape[(*in_order)[i]]);
  }
  module::setShapeOrVerify(getOutput(), out_shape);
  if (module::isShape(getInput())) {
    auto shape_v = module::getShapeTensorValue(getInput());
    std::vector<std::vector<int64_t>> input_shape_v;
    input_shape_v.push_back(shape_v);
    auto out_shape_v =
        module::commonShapeValInfer(getOperation(), input_shape_v, out_shape);
    module::bindShapeTensorValue(getOutput(), out_shape_v);
  }
}
