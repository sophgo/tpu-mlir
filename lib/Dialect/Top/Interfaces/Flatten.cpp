//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Module.h"

int64_t top::FlattenOp::getFLOPs() { return 0; }

LogicalResult top::FlattenOp::init(InferenceParameter &p) { return success(); }
void top::FlattenOp::deinit(InferenceParameter &p) {}

LogicalResult top::FlattenOp::inference(InferenceParameter &p) {
  UNREACHABLE_THIS("Not Implemented");
  return success();
}

void top::FlattenOp::shape_inference() {
  auto input_shape = module::getShape(getInput());
  int rank = input_shape.size();
  int64_t start_dim = getStartDim();
  int64_t end_dim = getEndDim();
  if (start_dim < 0) {
    start_dim += rank;
  }

  if (end_dim < 0) {
    end_dim += rank;
  }
  std::vector<int64_t> shape;
  if (module::isPlatform(module::Platform::ONNX)) {
    auto outer_dims =
        std::accumulate(input_shape.begin(), input_shape.begin() + start_dim, 1,
                        std::multiplies<int64_t>());
    auto inner_dims =
        std::accumulate(input_shape.begin() + start_dim, input_shape.end(), 1,
                        std::multiplies<int64_t>());
    shape = {outer_dims, inner_dims};
  } else {
    int64_t flatten_dim = -1;

    for (int i = 0; i < start_dim; i++) {
      shape.emplace_back(input_shape[i]);
    }
    shape.emplace_back(flatten_dim);
    for (int i = end_dim + 1; i < rank; i++) {
      shape.emplace_back(input_shape[i]);
    }
  }

  auto op = getOperation();
  OpBuilder builder(module::getCtx());
  builder.setInsertionPointAfter(op);
  auto out = getOutput();
  std::vector<NamedAttribute> attrs;
  attrs.emplace_back(
      builder.getNamedAttr("shape", builder.getI64ArrayAttr(shape)));
  attrs.emplace_back(builder.getNamedAttr(
      "flatten_start_dim", builder.getI64IntegerAttr(start_dim)));
  auto new_op = builder.create<top::ReshapeOp>(
      getLoc(), out.getType(), ArrayRef<Value>{getInput()}, attrs);
  out.replaceAllUsesWith(new_op.getOutput());
  new_op.shape_inference();
}
