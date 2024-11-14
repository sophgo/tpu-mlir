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

void YoloDetectionLowering::LoweringF32(PatternRewriter &rewriter,
                                        top::YoloDetectionOp op) const {
  rewriter.setInsertionPointAfter(op);
  std::vector<Value> operands;
  for (auto &&in : op.getOperands())
    operands.emplace_back(in);
  auto noneOp = module::getNoneOp(op);
  operands.push_back(noneOp);
  mlir::Type new_type = getQuantFloatType(op.getOutput());
  rewriter.replaceOpWithNewOp<tpu::YoloDetectionOp>(
      op, new_type, operands, op.getOperation()->getAttrs());
  return;
}

void YoloDetectionLowering::LoweringINT4(PatternRewriter &rewriter,
                                         top::YoloDetectionOp op,
                                         bool asymmetric) const {
  LoweringF32(rewriter, op);
}

void YoloDetectionLowering::LoweringINT8(PatternRewriter &rewriter,
                                         top::YoloDetectionOp op,
                                         bool asymmetric) const {
  LoweringF32(rewriter, op);
}

void YoloDetectionLowering::LoweringBF16(PatternRewriter &rewriter,
                                         top::YoloDetectionOp op) const {
  LoweringF32(rewriter, op);
}

void YoloDetectionLowering::LoweringF16(PatternRewriter &rewriter,
                                        top::YoloDetectionOp op) const {
  LoweringF32(rewriter, op);
}

void YoloDetectionLowering::LoweringF8(PatternRewriter &rewriter,
                                       top::YoloDetectionOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void YoloDetectionLowering::LoweringQuantized(PatternRewriter &rewriter,
                                              top::YoloDetectionOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

} // namespace bm1684x
} // namespace tpu_mlir
