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

void ConstantFillLowering::LoweringF32(PatternRewriter &rewriter,
                                       top::ConstantFillOp op) const {
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  Type new_type = getQuantFloatType(op->getResult(0));
  rewriter.replaceOpWithNewOp<tpu::ConstantFillOp>(op, new_type,
                                                   op->getOperands(), attrs);
}
void ConstantFillLowering::LoweringINT4(PatternRewriter &rewriter,
                                        top::ConstantFillOp op,
                                        bool asymmetric) const {
  UNREACHABLE_OP("Not Implemented", op);
}
void ConstantFillLowering::LoweringINT8(PatternRewriter &rewriter,
                                        top::ConstantFillOp op,
                                        bool asymmetric) const {
  LoweringF32(rewriter, op);
}

void ConstantFillLowering::LoweringBF16(PatternRewriter &rewriter,
                                        top::ConstantFillOp op) const {
  LoweringF32(rewriter, op);
}

void ConstantFillLowering::LoweringF16(PatternRewriter &rewriter,
                                       top::ConstantFillOp op) const {
  LoweringF32(rewriter, op);
}

void ConstantFillLowering::LoweringF8(PatternRewriter &rewriter,
                                      top::ConstantFillOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void ConstantFillLowering::LoweringQuantized(PatternRewriter &rewriter,
                                             top::ConstantFillOp op) const {
  lowering_common<tpu::ConstantFillOp>(rewriter, op.getOperation(),
                                       op.getOutput().getType());
}

} // namespace bm1684x
} // namespace tpu_mlir
