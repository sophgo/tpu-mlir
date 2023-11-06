//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Interfaces/IndexingMapsInterface.h"
#include "tpu_mlir/Interfaces/IndexingMapsInterface.cpp.inc"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"

using namespace mlir;

namespace tpu_mlir {

mlir::ArrayAttr getBinaryIndexingMaps(mlir::Operation *op) {
  auto in0_shape = module::getShape(op->getOperand(0));
  auto in1_shape = module::getShape(op->getOperand(1));
  auto out_shape = module::getShape(op->getResult(0));
  auto in0_dim_diff = out_shape.size() - in0_shape.size();
  auto in1_dim_diff = out_shape.size() - in1_shape.size();
  MLIRContext *ctx = op->getContext();

  SmallVector<AffineExpr> in0_index, in1_index;
  AffineExpr d0, d1, d2, d3, d4, d5, d6, d7;
  bindDims(ctx, d0, d1, d2, d3, d4, d5, d6, d7);
  SmallVector<AffineExpr> out_index = {d0, d1, d2, d3, d4, d5, d6, d7};
  auto c1 = mlir::getAffineConstantExpr(1, ctx);

  for (int i = 0; i < out_shape.size(); ++i) {
    if (i >= in0_dim_diff) {
      in0_index.push_back(out_shape[i] != in0_shape[i - in0_dim_diff] ? c1 : out_index[i]);
    }
    if (i >= in1_dim_diff) {
      in1_index.push_back(out_shape[i] != in1_shape[i - in1_dim_diff] ? c1 : out_index[i]);
    }
  }

  AffineMap in0_map = AffineMap::get(out_shape.size(), 0, in0_index, ctx);
  AffineMap in1_map = AffineMap::get(out_shape.size(), 0, in1_index, ctx);
  AffineMap identity_map = AffineMap::getMultiDimIdentityMap(out_shape.size(), ctx);
  SmallVector<AffineMap> indexingMaps{in0_map, in1_map, identity_map};
  return Builder(ctx).getAffineMapArrayAttr(indexingMaps);
};

}; // namespace tpu_mlir
