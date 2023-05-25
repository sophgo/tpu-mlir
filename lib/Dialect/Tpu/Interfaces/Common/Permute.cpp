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
#include "tpu_mlir/Support/Module.h"

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
    dump();
    llvm_unreachable("Not Implemented");
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
  return success();
}

// Permute can convert to Reshape in some situations.
// For example:
// [4,3,28,1] => [4,3,1,28]
// [4,3,1,28] => [4,1,3,28]
LogicalResult tpu::PermuteOp::canonicalize(tpu::PermuteOp op,
                                           PatternRewriter &rewriter) {
  std::vector<int64_t> shape = module::getShape(op.getInput());
  int dim_size = shape.size();
  int start = 0, end = dim_size - 1;
  auto order = module::getI64Array(op.getOrder());
  while (start < dim_size && start == order->at(start)) {
    start++;
  }
  while (end > start && end == order->at(end)) {
    end--;
  }
  bool do_reshape = true;
  int64_t sum = 1;
  for (int index = start; index <= end; index++) {
    sum *= shape[index];
    if (shape[index] != 1 && sum != shape[index]) {
      do_reshape = false;
      break;
    }
  }
  if (do_reshape == false) {
    return failure();
  }
  std::vector<Value> operands;
  operands.emplace_back(op.getInput());
  rewriter.replaceOpWithNewOp<tpu::ReshapeOp>(op, op.getResult().getType(),
                                              operands);
  return success();
};
