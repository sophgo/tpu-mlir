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

int64_t top::TileExOp::getFLOPs() { return 0; }

LogicalResult top::TileExOp::init(InferenceParameter &p) { return success(); }
void top::TileExOp::deinit(InferenceParameter &p) {}

LogicalResult top::TileExOp::inference(InferenceParameter &p) {
  llvm_unreachable("Not Implemented");
}

void top::TileExOp::shape_inference() {
  auto repeats_ = module::getI64Array(getRepeatsAttr());
  auto in_shape = module::getShape(getInput());
  int64_t dim = std::max(in_shape.size(), (*repeats_).size());
  auto in_shape_ = shape_expand_dim(in_shape, dim);
  auto repeats__ = shape_expand_dim(*repeats_, dim);
  auto out_shape = llvm::SmallVector<int64_t>();
  for (int i = 0; i < dim; ++i) {
    out_shape.push_back(in_shape_[i] * repeats__[i]);
  }
  module::setShapeOrVerify(getOutput(), out_shape);
}
