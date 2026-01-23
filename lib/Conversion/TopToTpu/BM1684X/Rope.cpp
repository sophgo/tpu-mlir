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

void RopeLowering::LoweringF32(PatternRewriter &rewriter,
                               top::RopeOp op) const {
  auto rope_mode = get_rope_mode(op.getRopeModeAttr().str());
  module::removeAttr(op, "mul1_round_mode");
  module::removeAttr(op, "mul2_round_mode");
  module::removeAttr(op, "add_round_mode");
  Operation *newOp;
  newOp = lowering_common_f32<tpu::RopeOp>(rewriter, op);
  newOp->setAttr("rope_mode",
                 tpu::RopeModeAttr::get(op.getContext(), rope_mode));
}

void RopeLowering::LoweringINT8(PatternRewriter &rewriter, top::RopeOp op,
                                bool asymmetric) const {
  if (!module::isCV184X() && !module::isSGTPUV8()) {
    LoweringF16(rewriter, op);
  } else {
    LoweringBF16(rewriter, op);
  }
}

void RopeLowering::LoweringINT4(PatternRewriter &rewriter, top::RopeOp op,
                                bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}

void RopeLowering::LoweringBF16(PatternRewriter &rewriter,
                                top::RopeOp op) const {
  if (op.getForceF32()) {
    LoweringF32(rewriter, op);
    return;
  }
  auto rope_mode = get_rope_mode(op.getRopeModeAttr().str());
  module::removeAttr(op, "mul1_round_mode");
  module::removeAttr(op, "mul2_round_mode");
  module::removeAttr(op, "add_round_mode");
  Operation *newOp;
  newOp = lowering_common_bf16<tpu::RopeOp>(rewriter, op);
  newOp->setAttr("rope_mode",
                 tpu::RopeModeAttr::get(op.getContext(), rope_mode));
}

void RopeLowering::LoweringF16(PatternRewriter &rewriter,
                               top::RopeOp op) const {
  if (op.getForceF32()) {
    LoweringF32(rewriter, op);
    return;
  }
  auto rope_mode = get_rope_mode(op.getRopeModeAttr().str());
  module::removeAttr(op, "mul1_round_mode");
  module::removeAttr(op, "mul2_round_mode");
  module::removeAttr(op, "add_round_mode");
  Operation *newOp;
  newOp = lowering_common_f16<tpu::RopeOp>(rewriter, op);
  newOp->setAttr("rope_mode",
                 tpu::RopeModeAttr::get(op.getContext(), rope_mode));
}

void RopeLowering::LoweringF8(PatternRewriter &rewriter, top::RopeOp op) const {
  LoweringF16(rewriter, op);
}

void RopeLowering::LoweringQuantized(PatternRewriter &rewriter,
                                     top::RopeOp op) const {
  // UNREACHABLE_OP("Not Implemented", op);
  auto mul1_round_mode = get_round_mode(op.getMul1RoundModeAttr().str());
  auto mul2_round_mode = get_round_mode(op.getMul2RoundModeAttr().str());
  auto add_round_mode = get_round_mode(op.getAddRoundModeAttr().str());
  auto rope_mode = get_rope_mode(op.getRopeModeAttr().str());
  Operation *newOp;
  // redundant processing to prevent uniform quant input but si8/ui8 output
  auto new_type = (module::isUniformQuantized(op.getInput1()) &&
                   !module::isUniformQuantized(op.getOutput()))
                      ? op.getInput1().getType()
                      : op.getOutput().getType();
  newOp = lowering_common<tpu::RopeOp>(rewriter, op.getOperation(), new_type);
  newOp->setAttr("mul1_round_mode",
                 tpu::RoundModeAttr::get(op.getContext(), mul1_round_mode));
  newOp->setAttr("mul2_round_mode",
                 tpu::RoundModeAttr::get(op.getContext(), mul2_round_mode));
  newOp->setAttr("add_round_mode",
                 tpu::RoundModeAttr::get(op.getContext(), add_round_mode));
  newOp->setAttr("rope_mode",
                 tpu::RopeModeAttr::get(op.getContext(), rope_mode));
}
} // namespace bm1684x
} // namespace tpu_mlir
