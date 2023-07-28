//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringBM1684.h"

namespace tpu_mlir {
namespace bm1684 {

void ClipLowering::LoweringF32(PatternRewriter &rewriter,
                               top::ClipOp op) const {
  auto type = op.getOutput().getType();
  rewriter.setInsertionPointAfter(op);
  auto max_loc = module::getLocLike(op.getOutput(), "max");
  std::vector<NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("const_val", op.getMinAttr()));
  auto max_op = rewriter.create<tpu::MaxConstOp>(
      max_loc, type, ValueRange{op.getInputs()}, attrs);

  attrs.clear();
  attrs.push_back(rewriter.getNamedAttr("const_val", op.getMaxAttr()));
  auto min_op = rewriter.create<tpu::MinConstOp>(
      op.getLoc(), type, ValueRange{max_op.getOutput()}, attrs);

  op.replaceAllUsesWith(min_op.getOperation());

  rewriter.eraseOp(op);
}

void ClipLowering::LoweringINT8(PatternRewriter &rewriter, top::ClipOp op,
                                bool asymmetric) const {
  auto type = op.getOutput().getType();
  rewriter.setInsertionPointAfter(op);
  auto max_loc = module::getLocLike(op.getOutput(), "max");
  std::vector<NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("const_val", op.getMinAttr()));
  auto max_op = rewriter.create<tpu::MaxConstOp>(
      max_loc, type, ValueRange{op.getInputs()}, attrs);

  attrs.clear();
  attrs.push_back(rewriter.getNamedAttr("const_val", op.getMaxAttr()));
  auto min_op = rewriter.create<tpu::MinConstOp>(
      op.getLoc(), type, ValueRange{max_op.getOutput()}, attrs);

  op.replaceAllUsesWith(min_op.getOperation());

  rewriter.eraseOp(op);
}

} // namespace bm1684
} // namespace tpu_mlir
