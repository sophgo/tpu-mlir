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

static void LoweringDetectionOutput(PatternRewriter &rewriter, top::DetectionOutputOp op, Type type) {
  rewriter.setInsertionPointAfter(op);
  std::vector<Value> operands;
  for (auto&& in: op.getOperands())
    operands.emplace_back(in);
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  mlir::Type new_type = getQuantFloatType(op.getOutput());
  rewriter.replaceOpWithNewOp<tpu::DetectionOutputOp>(op, new_type, operands, attrs);
  return;
}

void DetectionOutputLowering::LoweringF32(PatternRewriter &rewriter,
                               top::DetectionOutputOp op) const {
  LoweringDetectionOutput(rewriter, op, rewriter.getF32Type());
}
void DetectionOutputLowering::LoweringINT4(PatternRewriter &rewriter, top::DetectionOutputOp op,
                                bool asymmetric) const {
  LoweringF32(rewriter, op);
}
void DetectionOutputLowering::LoweringINT8(PatternRewriter &rewriter, top::DetectionOutputOp op,
                                bool asymmetric) const {
  LoweringF32(rewriter, op);
}

void DetectionOutputLowering::LoweringBF16(PatternRewriter &rewriter,
                                top::DetectionOutputOp op) const {
  LoweringF32(rewriter, op);
}

void DetectionOutputLowering::LoweringF16(PatternRewriter &rewriter,
                               top::DetectionOutputOp op) const {
  LoweringF32(rewriter, op);
}

void DetectionOutputLowering::LoweringQuantized(PatternRewriter &rewriter,
                                     top::DetectionOutputOp op) const {
  llvm_unreachable("Not Implemented");
}

} // namespace bm1684x
} // namespace tpu_mlir
