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

void CastIntLowering::Lowering(PatternRewriter &rewriter,
                               top::CastOp op) const {
  auto to = op.getTo();
  if (to == "INT32") {
    auto round_mode = op.getRoundModeAttr().str();
    auto new_type = RankedTensorType::get(module::getShape(op.getOutput()),
                                          rewriter.getIntegerType(32, true));
    auto newOp =
        lowering_common<tpu::CastOp>(rewriter, op.getOperation(), new_type);
    newOp.setRoundModeAttr(
        tpu::RoundModeAttr::get(op.getContext(), get_round_mode(round_mode)));
  }
}

void CastLowering::LoweringF32(PatternRewriter &rewriter,
                               top::CastOp op) const {
  auto round_mode = op.getRoundModeAttr().str();
  auto newOp = lowering_common<tpu::CastOp>(rewriter, op.getOperation(),
                                            op.getOutput().getType());
  newOp.setRoundModeAttr(
      tpu::RoundModeAttr::get(op.getContext(), get_round_mode(round_mode)));
}

void CastLowering::LoweringINT8(PatternRewriter &rewriter, top::CastOp op,
                                bool asymmetric) const {
  auto round_mode = op.getRoundModeAttr().str();
  auto newOp = lowering_common<tpu::CastOp>(rewriter, op.getOperation(),
                                            op.getOutput().getType());
  newOp.setRoundModeAttr(
      tpu::RoundModeAttr::get(op.getContext(), get_round_mode(round_mode)));
}

} // namespace bm1684
} // namespace tpu_mlir
