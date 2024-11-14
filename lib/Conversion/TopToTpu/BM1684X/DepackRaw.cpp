//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringBM1684X.h"

namespace tpu_mlir {
namespace bm1684x {

static void LoweringDepackRaw(PatternRewriter &rewriter, top::DepackRawOp op) {
  std::vector<Value> operands;
  operands.emplace_back(op.getOperand());
  mlir::Type new_type = op.getOutput().getType();
  rewriter.replaceOpWithNewOp<tpu::DepackRawOp>(op, new_type, operands,
                                                op.getOperation()->getAttrs());
  return;
}

void DepackRawLowering::LoweringINT8(PatternRewriter &rewriter,
                                     top::DepackRawOp op,
                                     bool asymmetric) const {
  LoweringDepackRaw(rewriter, op);
}

void DepackRawLowering::LoweringINT4(PatternRewriter &rewriter,
                                     top::DepackRawOp op,
                                     bool asymmetric) const {
  LoweringDepackRaw(rewriter, op);
}

void DepackRawLowering::LoweringF32(PatternRewriter &rewriter,
                                    top::DepackRawOp op) const {
  LoweringDepackRaw(rewriter, op);
}

void DepackRawLowering::LoweringBF16(PatternRewriter &rewriter,
                                     top::DepackRawOp op) const {
  LoweringDepackRaw(rewriter, op);
}

void DepackRawLowering::LoweringF16(PatternRewriter &rewriter,
                                    top::DepackRawOp op) const {
  LoweringDepackRaw(rewriter, op);
}

void DepackRawLowering::LoweringF8(PatternRewriter &rewriter,
                                   top::DepackRawOp op) const {
  llvm_unreachable("FIXME: not implement");
}

void DepackRawLowering::LoweringQuantized(PatternRewriter &rewriter,
                                          top::DepackRawOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

} // namespace bm1684x
} // namespace tpu_mlir
