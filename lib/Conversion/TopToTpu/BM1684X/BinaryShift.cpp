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

void BinaryShiftLowering::LoweringF32(PatternRewriter &rewriter,
                                      top::BinaryShiftOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void BinaryShiftLowering::LoweringINT8(PatternRewriter &rewriter,
                                       top::BinaryShiftOp op,
                                       bool asymmetric) const {
  UNREACHABLE_OP("Not Implemented", op);
}
void BinaryShiftLowering::LoweringINT4(PatternRewriter &rewriter,
                                       top::BinaryShiftOp op,
                                       bool asymmetric) const {
  UNREACHABLE_OP("Not Implemented", op);
}
void BinaryShiftLowering::LoweringBF16(PatternRewriter &rewriter,
                                       top::BinaryShiftOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void BinaryShiftLowering::LoweringF16(PatternRewriter &rewriter,
                                      top::BinaryShiftOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void BinaryShiftLowering::LoweringF8(PatternRewriter &rewriter,
                                     top::BinaryShiftOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void BinaryShiftLowering::LoweringQuantized(PatternRewriter &rewriter,
                                            top::BinaryShiftOp op) const {
  auto round_mode = get_round_mode(op.getRoundModeAttr().str());
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    if (attr.getName() == "round_mode") {
      attrs.push_back(rewriter.getNamedAttr(
          "round_mode", tpu::RoundModeAttr::get(op.getContext(), round_mode)));
    } else {
      attrs.push_back(attr);
    }
  }
  op->setAttrs(attrs);
  lowering_common<tpu::BinaryShiftOp>(rewriter, op.getOperation(),
                                      op.getOutput().getType());
}

} // namespace bm1684x
} // namespace tpu_mlir
