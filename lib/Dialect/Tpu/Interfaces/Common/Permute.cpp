//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

permute_attr_t tpu::PermuteOp::parseParam() {
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

LogicalResult tpu::PermuteOp::init(InferenceParameter &p) {
  permute_attr_t *attr = new permute_attr_t;
  auto param = parseParam();
  attr->in_shape_fix = param.in_shape_fix;
  attr->order_fix = param.order_fix;
  attr->out_shape_fix = param.out_shape_fix;
  p.handle = (void *)attr;
  return success();
}

void tpu::PermuteOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto info = (permute_attr_t *)p.handle;
    delete info;
    p.handle = nullptr;
  }
}

LogicalResult tpu::PermuteOp::inference(InferenceParameter &p) {
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

ArrayAttr tpu::PermuteOp::getIndexingMaps() {

  auto order = module::getI64Array(getOrder());
  int no_exchange_dim = 0;
  for (int i = 0, n = order->size(); i < n; i++) {
    if (i == order->at(i))
      no_exchange_dim++;
    else
      break;
  };
  if (no_exchange_dim > 0) {
    MLIRContext *context = getContext();
    AffineMap identityMap =
        AffineMap::getMultiDimIdentityMap(no_exchange_dim, context);
    AffineMap emptyMap = AffineMap::get(no_exchange_dim, 0, context);
    SmallVector<AffineMap> indexingMaps{identityMap, emptyMap, identityMap};
    return Builder(getContext()).getAffineMapArrayAttr(indexingMaps);
  }
  return Builder(getContext()).getAffineMapArrayAttr({});
};

bool tpu::PermuteOp::support_multi_core() {
  auto order = *(module::getI64Array(getOrder()));
  auto in_shape = module::getShape(getInput());
  if (in_shape.size() == 4 && order[0] == 0 && order[1] == 3 && order[2] == 1 &&
      order[3] == 2 && in_shape[3] == 3) {
    return true;
  }
  return false;
}
