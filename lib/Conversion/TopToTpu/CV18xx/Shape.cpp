//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringCV18xx.h"

namespace tpu_mlir {
namespace cv18xx {

void ShapeLowering::LoweringINT8(PatternRewriter &rewriter, top::ShapeOp op,
                                 bool asymmetric) const {
  lowering_common_int8<tpu::ShapeOp>(rewriter, op, asymmetric);
}

void ShapeLowering::LoweringBF16(PatternRewriter &rewriter,
                                 top::ShapeOp op) const {
  auto v = op.getResult();
  auto shape = module::getShape(v);
  auto ctx = v.getContext();
  Type new_type = RankedTensorType::get(shape, IntegerType::get(ctx, 32));
  rewriter.replaceOpWithNewOp<tpu::ShapeOp>(op, new_type, op.getInput());
}

} // namespace cv18xx
} // namespace tpu_mlir
