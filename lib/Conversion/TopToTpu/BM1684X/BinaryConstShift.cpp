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

void BinaryConstShiftLowering::LoweringF32(PatternRewriter &rewriter,
                                           top::BinaryConstShiftOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void BinaryConstShiftLowering::LoweringINT8(PatternRewriter &rewriter,
                                            top::BinaryConstShiftOp op,
                                            bool asymmetric) const {
  UNREACHABLE_OP("Not Implemented", op);
}
void BinaryConstShiftLowering::LoweringINT4(PatternRewriter &rewriter,
                                            top::BinaryConstShiftOp op,
                                            bool asymmetric) const {
  UNREACHABLE_OP("Not Implemented", op);
}
void BinaryConstShiftLowering::LoweringBF16(PatternRewriter &rewriter,
                                            top::BinaryConstShiftOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void BinaryConstShiftLowering::LoweringF16(PatternRewriter &rewriter,
                                           top::BinaryConstShiftOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void BinaryConstShiftLowering::LoweringF8(PatternRewriter &rewriter,
                                          top::BinaryConstShiftOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void BinaryConstShiftLowering::LoweringQuantized(
    PatternRewriter &rewriter, top::BinaryConstShiftOp op) const {
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
  lowering_common<tpu::BinaryConstShiftOp>(rewriter, op.getOperation(),
                                           op.getOutput().getType());
}

} // namespace bm1684x
} // namespace tpu_mlir
